# Prefix Codebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantize the Qwen3-TTS prompt (prefill) keys with levels fitted on the prompt itself (k-means or one min-max window) instead of keeping them fp16, so a fully quantized cache can be tested with g32pre on the generated keys.

**Architecture:** A new pure-tensor module `turboquant/fitted_levels.py` fits, assigns and reconstructs per-coordinate levels. `MSECompressor` gains a `levels_mode` codebook stage that uses it; `TurboQuantV3` gains a prefix key compressor and a `prefix=` argument on `compress_kv`/`decompress_kv`. The Qwen cache compresses the prefix span at prefill through that path; the divergence recorder splices the quantized prompt where it used to splice fp16. Naming, scoring, summaries, smoke and run scripts carry the new setting. Everything is default-off so the six existing cells reproduce unchanged.

**Tech Stack:** Python 3.13 / torch (CPU for all tests), pandas, pytest, bash scripts on the SageMaker L4 box.

**Spec:** `docs/superpowers/specs/2026-11-18-prefix-codebook-design.md`

## Global Constraints

- Defaults off: `prefix_codebook=""`, `prefix_bits=0`; with both at default every output (wav names, CSV values, labels, bytes) is identical to today.
- `prefix_codebook` is `""`, `"kmeans"` or `"minmax"`; it requires `protect_prefix == -1`, anything else raises `ValueError`.
- Effective prefix bits = `prefix_bits or key_bits`, capped at 8.
- Metadata accounting: `meta_scalars` fp16 scalars per coordinate per block, `L = 2**bits` for kmeans, `2` for minmax; `compressed_bytes_per_token` is the only formula, new arguments are trailing keywords (VALL-E-X calls it positionally).
- Wav segment `_pq=k<bits>` / `_pq=m<bits>` replaces `_pp=prompt`; fp16 wavs carry no segment. Config label suffix `+pk<bits>` / `+pm<bits>`.
- CSV columns `prefix_codebook` (str) and `prefix_bits` (int, the CLI value) appended after `protect_prefix`.
- Prefix values use the ordinary value compressor; prefix keys are un-RoPEd / re-RoPEd when `pre_rope_keys` is on.
- Recorder keeps quantizing generated keys over the full tensor aligned at position 0 (as `g32pp` rows were), splicing the quantized prompt.
- `uv` only; `ruff format` + `ruff check` clean; functions <= 50 lines; commit messages one sentence, no emojis, no attribution lines; push with `gh auth switch -u pancarnas` (memory note).
- Existing tests must pass unchanged except the two column-list edits named in Task 6.

## Review Focus

1. A prompt shorter than `2**bits` tokens (k-means with more levels than samples): `fit_levels` must not divide by zero or produce NaN levels; reconstruction must be exact. Test in Task 1.
2. A coordinate whose prompt values are all equal (zero range): minmax must not produce NaN; assign/reconstruct returns that value. Test in Task 1.
3. `--prefix-bits 0` with configs K8V8 and K2V2 in one run: effective bits differ per config, so the wav names and the compressor cache key must differ per config; a shared compressor across configs would silently reuse 2-bit levels for K8V8. Test in Task 6.
4. `--resume` on a CSV written before these columns existed, with a codebook flag set: must yield an empty done-set (no sentence skipped wrongly). Test in Task 6.
5. The prefill update with `track_only=True` must leave the buffer pristine and still report codebook bytes for the prefix. Test in Task 5.

---

### Task 1: Fitted levels module

**Files:**
- Create: `turboquant/fitted_levels.py`
- Modify: `models/Qwen3-TTS/benchmarks/keygauss_core.py` (`quant_kmeans` becomes a wrapper)
- Test: `tests/test_fitted_levels.py`

**Interfaces:**
- Produces: `fit_levels(z: Tensor, bits: int, mode: str, iters: int = 20) -> Tensor` with `z` `(N, S, D)` float32, returns `(N, L, D)` float32 sorted ascending along `L`, `L = 2**bits`; `assign(z: Tensor, levels: Tensor) -> Tensor` `(N, S, D)` int32 nearest-level index; `reconstruct(indices: Tensor, levels: Tensor) -> Tensor` `(N, S, D)` float32; `meta_scalars(bits: int, mode: str) -> int` (`2**bits` for `"kmeans"`, `2` for `"minmax"`); `minmax_levels(lo: Tensor, hi: Tensor, bits: int) -> Tensor` `(N, L, D)` evenly spaced from `lo` `(N, 1, D)` to `hi` `(N, 1, D)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_minmax_levels_hit_endpoints_and_are_even():
    z = torch.rand(3, 20, 8) * 4 - 2
    lv = fit_levels(z, 2, "minmax")
    assert torch.allclose(lv[:, 0], z.amin(1)) and torch.allclose(lv[:, -1], z.amax(1))
    assert torch.allclose(lv[:, 1] - lv[:, 0], lv[:, 2] - lv[:, 1])

def test_kmeans_is_deterministic_and_never_worse_than_minmax():
    z = 0.3 * torch.randn(4, 53, 16, generator=torch.Generator().manual_seed(0)) + 1.5
    a, b = fit_levels(z, 2, "kmeans"), fit_levels(z, 2, "kmeans")
    assert torch.equal(a, b)
    err = lambda lv: ((reconstruct(assign(z, lv), lv) - z) ** 2).mean()
    assert err(a) <= err(fit_levels(z, 2, "minmax"))

def test_more_levels_than_samples_is_lossless_and_finite():
    z = torch.randn(2, 5, 4)
    for mode in ("kmeans", "minmax"):
        lv = fit_levels(z, 3, mode)           # 8 levels, 5 samples
        assert torch.isfinite(lv).all()
    lv = fit_levels(z, 3, "kmeans")
    assert torch.allclose(reconstruct(assign(z, lv), lv), z, atol=1e-6)

def test_constant_coordinate_has_no_nan():
    z = torch.ones(1, 10, 3)
    for mode in ("kmeans", "minmax"):
        lv = fit_levels(z, 2, mode)
        assert torch.isfinite(lv).all()
        assert torch.allclose(reconstruct(assign(z, lv), lv), z)

def test_meta_scalars():
    assert meta_scalars(2, "kmeans") == 4 and meta_scalars(3, "minmax") == 2

def test_keygauss_quant_kmeans_still_passes_segment_tests():
    # tests/test_keygauss_segments.py runs unchanged; this test only pins the wrapper shape
    c = _core()  # same loader as tests/test_keygauss_segments.py
    z = torch.randn(40, 6)
    assert c.quant_kmeans(z, 2).shape == z.shape
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fitted_levels.py -q`
Expected: FAIL, `ModuleNotFoundError: turboquant.fitted_levels`

- [ ] **Step 3: Implement `turboquant/fitted_levels.py`**

`fit_levels`: `minmax` → `minmax_levels(z.amin(1, keepdim=True), z.amax(1, keepdim=True), bits)`; `kmeans` → initial levels at the `L` quantiles `(i + 0.5) / L` of each coordinate (`torch.quantile` over dim 1), then `iters` Lloyd steps: `idx = assign(z, levels)`, one-hot `(N, S, D, L)` sums give member means, empty clusters keep their level; sort along `L` at the end. Unknown mode raises `ValueError`. `assign`: `(z.unsqueeze(-1) - levels.unsqueeze(1)).abs().argmin(-1)` with levels broadcast `(N, 1, D, L)` after a transpose; `reconstruct`: `torch.gather` on the `L` axis.

- [ ] **Step 4: Make `keygauss_core.quant_kmeans(z, bits, iters=20)` a wrapper**

Body: `zz = z.unsqueeze(0)` (`(1, S, D)`), `lv = fit_levels(zz, bits, "kmeans", iters)`, return `reconstruct(assign(zz, lv), lv)[0]`. Delete the old one-hot loop.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_fitted_levels.py tests/test_keygauss_segments.py tests/test_keygauss.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add turboquant/fitted_levels.py tests/test_fitted_levels.py models/Qwen3-TTS/benchmarks/keygauss_core.py
git commit -m "Add fitted_levels: per-coordinate k-means and min-max level fitting shared by the compressor and the keygauss diagnostic"
```

---

### Task 2: Compressor codebook stage and accounting

**Files:**
- Modify: `turboquant/compressors_v3.py:28-43` (`compressed_bytes_per_token`), `:46-72` (`MSECompressor.__init__`), `:109-158` (`compress`), `:160-190` (`decompress`), `:192-202` (`memory_bytes`)
- Test: `tests/test_prefix_codebook.py` (new; later tasks add to it)

**Interfaces:**
- Consumes: Task 1 functions.
- Produces: `compressed_bytes_per_token(head_dim, bits, group=0, meta_scalars=0, block=0) -> float`; `MSECompressor(head_dim, bits, seed, device="cpu", group=0, levels_mode="")`; `compress()` output dict gains `"levels"` `(B, H, L, D)` fp16 when `levels_mode == "kmeans"` and `"lo"`, `"hi"` `(B, H, 1, D)` fp16 when `"minmax"`; `MSECompressor.meta_scalars -> int` property (0 when no `levels_mode`); `memory_bytes(B, H, S)` uses `block=S`.

- [ ] **Step 1: Write the failing tests**

```python
def test_bytes_per_token_adds_amortized_metadata_and_keeps_old_signature():
    assert compressed_bytes_per_token(128, 2) == compressed_bytes_per_token(128, 2, 0)
    assert compressed_bytes_per_token(128, 2, 0, meta_scalars=4, block=53) == pytest.approx(
        compressed_bytes_per_token(128, 2) + 4 * 128 * 2 / 53)

@pytest.mark.parametrize("mode", ["kmeans", "minmax"])
def test_levels_compressor_round_trip_is_lossy_but_close(mode):
    c = MSECompressor(32, 3, seed=1, levels_mode=mode)
    x = torch.randn(1, 2, 53, 32)
    out = c.compress(x)
    key = "levels" if mode == "kmeans" else "lo"
    assert out[key].dtype == torch.float16
    y = c.decompress(out)
    assert y.shape == x.shape
    assert torch.nn.functional.cosine_similarity(x.flatten(2), y.flatten(2), dim=-1).mean() > 0.98

def test_levels_compressor_prefix_longer_than_levels_is_near_exact():
    c = MSECompressor(16, 8, seed=1, levels_mode="kmeans")  # 256 levels on 40 tokens
    x = torch.randn(1, 1, 40, 16)
    y = c.decompress(c.compress(x))
    assert torch.allclose(x, y, atol=2e-3)  # fp16 norm + fp16 levels only

def test_levels_mode_off_is_bitwise_old_path():
    a, b = MSECompressor(16, 2, seed=1), MSECompressor(16, 2, seed=1, levels_mode="")
    x = torch.randn(1, 1, 30, 16)
    assert torch.equal(a.decompress(a.compress(x)), b.decompress(b.compress(x)))

def test_memory_bytes_counts_levels_over_the_block():
    c = MSECompressor(32, 2, seed=1, levels_mode="kmeans")
    assert c.meta_scalars == 4
    assert c.memory_bytes(1, 2, 53)["compressed_bytes"] == round(2 * 53 * compressed_bytes_per_token(32, 2, 0, 4, 53))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_codebook.py -q`
Expected: FAIL on the `meta_scalars` keyword / `levels_mode` keyword.

- [ ] **Step 3: Implement**

`compressed_bytes_per_token`: add `+ (meta_scalars * head_dim * 2) / block` when `meta_scalars > 0` (`block` must be > 0, else `ValueError`). `MSECompressor`: store `levels_mode`, reject it together with `group > 0`; in `compress`, when set, `z = rotated.reshape(B, H, S, D)` → fit on `(B*H, S, D)`, round levels to fp16 then back to fp32 before `assign` (same rule as the group path), store `levels` fp16 for kmeans or `lo`/`hi` fp16 for minmax (levels rebuilt with `minmax_levels` on decompress), indices packed as today; `decompress` mirrors; `meta_scalars` property from Task 1; `memory_bytes` passes `meta_scalars` and `block=S`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_prefix_codebook.py tests/test_keyfix.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add turboquant/compressors_v3.py tests/test_prefix_codebook.py
git commit -m "MSECompressor levels_mode: fitted-levels codebook stage with amortized metadata accounting"
```

---

### Task 3: TurboQuantV3 prefix compressor

**Files:**
- Modify: `turboquant/compressors_v3.py:205-355` (`TurboQuantV3`)
- Test: `tests/test_prefix_codebook.py`

**Interfaces:**
- Consumes: Task 2 `MSECompressor(..., levels_mode=)`.
- Produces: `TurboQuantV3(..., key_group=0, prefix_codebook="", prefix_bits=0)`; attribute `prefix_key_compressor: MSECompressor | None` (bits `min(prefix_bits or self.key_bits, 8)`, `seed=seed_base`, `levels_mode=prefix_codebook`); `compress_kv(keys, values, prefix: int = 0) -> tuple[dict, dict]` where with `prefix > 0` each dict carries `"prefix": {"compressed": <MSECompressor output>, "n": prefix}` and the rest describes `keys[:, :, prefix:]` exactly as today; `decompress_kv` returns `[prefix recon | rest]`; `prefix_bits_effective: int` attribute.

- [ ] **Step 1: Write the failing tests**

```python
def test_v3_prefix_zero_is_identical_to_old_output():
    c = TurboQuantV3(16, key_bits=2, value_bits=2, residual_window=0, protected_layers=0, key_group=16)
    k = torch.randn(1, 2, 64, 16); v = torch.randn(1, 2, 64, 16)
    a = c.decompress_kv(*c.compress_kv(k, v)); b = c.decompress_kv(*c.compress_kv(k, v, prefix=0))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

def test_v3_prefix_compressed_with_fitted_levels_rest_unchanged():
    base = TurboQuantV3(16, 2, 2, residual_window=0, protected_layers=0, key_group=16)
    pq = TurboQuantV3(16, 2, 2, residual_window=0, protected_layers=0, key_group=16,
                      prefix_codebook="kmeans", prefix_bits=3)
    assert pq.prefix_bits_effective == 3 and pq.prefix_key_compressor.bits == 3
    k = torch.randn(1, 2, 70, 16); v = torch.randn(1, 2, 70, 16)
    ck, cv = pq.compress_kv(k, v, prefix=22)
    assert ck["prefix"]["n"] == 22 and "levels" in ck["prefix"]["compressed"]
    rk, rv = pq.decompress_kv(ck, cv)
    assert rk.shape == k.shape
    # the prompt is lossy but far better than the fixed 2-bit codebook on the same keys
    fixed = base.decompress_kv(*base.compress_kv(k[:, :, :22], v[:, :, :22]))[0]
    err = lambda a, b: ((a - b) ** 2).sum() / (b ** 2).sum()
    assert err(rk[:, :, :22], k[:, :, :22]) < 0.5 * err(fixed, k[:, :, :22])
    # the rest equals compressing keys[:, :, 22:] alone (groups aligned at the prefix end)
    rest = base.decompress_kv(*base.compress_kv(k[:, :, 22:], v[:, :, 22:]))[0]
    assert torch.allclose(rk[:, :, 22:], rest)

def test_v3_prefix_bits_zero_follows_key_bits():
    assert TurboQuantV3(16, 8, 8, protected_layers=0, prefix_codebook="minmax").prefix_bits_effective == 8

def test_v3_rejects_bad_mode():
    with pytest.raises(ValueError):
        TurboQuantV3(16, 2, 2, prefix_codebook="median")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_codebook.py -q -k v3`
Expected: FAIL, unexpected keyword `prefix_codebook`.

- [ ] **Step 3: Implement**

Build `prefix_key_compressor` in `__init__` when `prefix_codebook`; in `compress_kv` split off `[:prefix]`, compress keys with it and values with `val_compressor`, then run today's body on the remainder; `decompress_kv` decompresses the prefix entry first and concatenates (handle the `compressed is None` short path when the remainder fits the window). Keep `memory_bytes` unchanged (the cache does prefix accounting).

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_prefix_codebook.py tests/test_keyfix.py tests/test_turboquant_cache_readback.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add turboquant/compressors_v3.py tests/test_prefix_codebook.py
git commit -m "TurboQuantV3: optional prefix key compressor and prefix= argument on compress_kv/decompress_kv"
```

---

### Task 4: Config fields and validation

**Files:**
- Modify: `turboquant/config.py:36-52`
- Test: `tests/test_prefix_codebook.py`

**Interfaces:**
- Produces: `TurboQuantConfig.prefix_codebook: str = ""`, `TurboQuantConfig.prefix_bits: int = 0`; construction raises `ValueError` when `prefix_codebook` is set and `protect_prefix != -1`, or when the mode is unknown, or when `prefix_bits < 0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_config_defaults_off_and_validation():
    c = TurboQuantConfig()
    assert c.prefix_codebook == "" and c.prefix_bits == 0
    TurboQuantConfig(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=2)
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_prefix=0, prefix_codebook="kmeans")
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_prefix=-1, prefix_codebook="other")
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_prefix_codebook.py -q -k config` → FAIL (unexpected keyword).

- [ ] **Step 3: Implement** — two fields with the spec's comment block; validation in `__post_init__` (add one if the dataclass has none; keep existing field order).

- [ ] **Step 4: Run** — `pytest tests/test_prefix_codebook.py tests/test_protect_prefix.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add turboquant/config.py tests/test_prefix_codebook.py
git commit -m "TurboQuantConfig: prefix_codebook and prefix_bits fields, validated against protect_prefix"
```

---

### Task 5: Qwen cache prefill compression

**Files:**
- Modify: `models/Qwen3-TTS/qwen_tts/core/models/turboquant_kv_cache.py:248-265` (`_get_compressor`), `:334-398` (`update`), `:428-490` (`memory_report`); new private method `_compress_prefix(layer_idx, template)`
- Test: `tests/test_prefix_codebook.py` (reuse `_cfg`, `_feed_qwen`, `_qwen_cache_mod` from `tests/test_keyfix.py`)

**Interfaces:**
- Consumes: Task 3 `compress_kv(..., prefix=)`, Task 4 config.
- Produces: with a codebook, after the prefill update the buffer's `[0, pfx)` keys and values hold the lossy reconstruction (unless `track_only`); `_chunks_k[layer]` gets a first entry `{"blob": ck, "span": pfx, "prefix": True}` that `update()`'s `prev_compressed` sum ignores; `memory_report()["compressed_bytes"]` counts prefix keys at `compressed_bytes_per_token(D, kb_prefix, 0, meta_scalars, pfx)` and prefix values at `compressed_bytes_per_token(D, vb)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cache_prefix_codebook_quantizes_the_prompt_and_leaves_generation_as_pp():
    mod = _qwen_cache_mod()
    g = torch.Generator().manual_seed(0)
    P, D, H = 40, 64, 2
    k0, v0 = torch.randn(1, H, P, D, generator=g), torch.randn(1, H, P, D, generator=g)
    pq = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3, key_group=16), n_layers=1)
    pp = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, key_group=16), n_layers=1)
    for c in (pq, pp):
        c.update(k0.clone(), v0.clone(), layer_idx=0)
    assert not torch.equal(pq._buf_k[0][:, :, :P], k0)            # prompt keys lossy
    assert not torch.equal(pq._buf_v[0][:, :, :P], v0)            # prompt values quantized too
    assert torch.nn.functional.cosine_similarity(pq._buf_k[0][:, :, :P].flatten(2), k0.flatten(2), dim=-1).mean() > 0.97
    assert pq._chunks_k[0][0].get("prefix") is True and pq._chunks_k[0][0]["span"] == P
    k_in_q, k_out_q = _feed_qwen(pq, steps=4, step_len=8, seed=1)
    k_in_p, k_out_p = _feed_qwen(pp, steps=4, step_len=8, seed=1)
    assert torch.equal(k_out_q[:, :, P:], k_out_p[:, :, P:])        # generated path identical to pp
    assert pq._compressed_upto[0] == pp._compressed_upto[0] == P + 32

def test_cache_prefix_codebook_with_identity_rope_matches_post_rope_path():
    # cos = 1, sin = 0 makes RoPE the identity, so the pre-RoPE path must reproduce the plain path exactly
    mod = _qwen_cache_mod()
    P, D, H = 40, 64, 2
    g = torch.Generator().manual_seed(0)
    k0, v0 = torch.randn(1, H, P, D, generator=g), torch.randn(1, H, P, D, generator=g)
    plain = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3), n_layers=1)
    pre = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3, pre_rope_keys=True), n_layers=1)
    plain.update(k0.clone(), v0.clone(), layer_idx=0)
    pre.update(k0.clone(), v0.clone(), layer_idx=0,
               cache_kwargs={"cos": torch.ones(1, P, D), "sin": torch.zeros(1, P, D)})
    assert torch.allclose(pre._buf_k[0][:, :, :P], plain._buf_k[0][:, :, :P], atol=1e-5)
    assert not torch.equal(pre._buf_k[0][:, :, :P], k0)

def test_cache_prefix_codebook_memory_report():
    mod = _qwen_cache_mod()
    from turboquant.compressors_v3 import compressed_bytes_per_token as bpt
    D, H, P, n = 64, 2, 40, 32
    pq = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, prefix_codebook="minmax", prefix_bits=3, track_only=True), n_layers=1)
    pp = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, track_only=True), n_layers=1)
    for c in (pq, pp):
        c.update(torch.randn(1, H, P, D), torch.randn(1, H, P, D), layer_idx=0)
        c.update(torch.randn(1, H, n, D), torch.randn(1, H, n, D), layer_idx=0)
    diff = pp.memory_report()["compressed_bytes"] - pq.memory_report()["compressed_bytes"]
    assert diff == H * P * (2 * D * 2 - bpt(D, 3, 0, 2, P) - bpt(D, 2))
    assert torch.isfinite(pq._buf_k[0][:, :, :P]).all()  # track_only: buffer untouched

def test_cache_default_config_unchanged():
    mod = _qwen_cache_mod()
    a, b = mod.TurboQuantKVCache(_cfg(), n_layers=1), mod.TurboQuantKVCache(_cfg(prefix_codebook=""), n_layers=1)
    ka, oa = _feed_qwen(a, steps=5, step_len=8); kb, ob = _feed_qwen(b, steps=5, step_len=8)
    assert torch.equal(oa, ob)
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_prefix_codebook.py -q -k cache` → FAIL.

- [ ] **Step 3: Implement**

`_get_compressor` passes `prefix_codebook` / `prefix_bits`. In `update`, right after `_prefix_len` is set on the first call and the buffer slice-write, if `self.config.prefix_codebook and not self.config.track_only` call `_compress_prefix(layer_idx, key_states)`: slice `[0, pfx)`, un-RoPE when `pre_rope_keys` (`_rope_slice(0, pfx)` after `_record_rope`, so move the `_record_rope` call above it), `ck, cv = comp.compress_kv(k, v, prefix=pfx)`, `decompress_kv`, re-RoPE, write back, append the chunk with `"prefix": True`. In `update()`'s `prev_compressed` sum skip chunks with `"prefix"`. In `memory_report`, when a codebook is set, prefix tokens are excluded from `rw` and charged as in the Interfaces block.

- [ ] **Step 4: Run** — `pytest tests/test_prefix_codebook.py tests/test_protect_prefix.py tests/test_keyfix.py tests/test_turboquant_cache_readback.py models/Qwen3-TTS/tests/test_turboquant_cache.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add models/Qwen3-TTS/qwen_tts/core/models/turboquant_kv_cache.py tests/test_prefix_codebook.py
git commit -m "Qwen cache: compress the protected prefix at prefill with the fitted-levels codebook and count it in memory_report"
```

---

### Task 6: Recorder, naming, resume, CLI

**Files:**
- Modify: `models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py:72-95` (`COLUMNS`, `_N_ID`), `:124-151` (`make_config`), `:200-245` (`DivergenceRecorder.__init__`, `_comp`), `:287-361` (`record`), `:460-509` (`_wav_path`, `load_done`), `:517-524` (`audio_done`), `:596-631` (calls), `:710-717` and `:751-770` (CLI, banner, recorder construction)
- Modify: `tests/test_kv_attn_divergence.py` (`COLS` list) and `tests/test_protect_prefix.py:166-167` (id tuples gain two entries: `""`, `0`)
- Test: `tests/test_prefix_codebook.py` (reuse `_recorder_ns`, `_single`, `_cols`, `_talker_module`)

**Interfaces:**
- Consumes: Tasks 3-4.
- Produces: `COLUMNS` has `prefix_codebook`, `prefix_bits` after `protect_prefix`, `_N_ID = 13`; `make_config(kb, vb, rw, protected_layers=2, key_group=0, pre_rope_keys=False, protect_prefix=0, prefix_codebook="", prefix_bits=0)`; `DivergenceRecorder(..., protect_prefix=0, prefix_codebook="", prefix_bits=0)`; `_wav_path(..., protect_prefix=0, prefix_codebook="", prefix_bits=0)` writes `_pq=k<eff>` / `_pq=m<eff>` with `eff = prefix_bits or kb` instead of `_pp=prompt`; `load_done(out_path, protected_layers, key_group=0, pre_rope=False, protect_prefix=0, prefix_codebook="", prefix_bits=0)`; `audio_done` gains the same two; CLI `--prefix-codebook {kmeans,minmax}` (default `""`) and `--prefix-bits` (default 0).

- [ ] **Step 1: Write the failing tests**

```python
def test_recorder_prefix_codebook_splices_quantized_prompt():
    ns = _recorder_ns(); D, groups, P = 64, 2, 24
    rec = ns["DivergenceRecorder"](specs=[(2, 2, 0)], n_layers=4, prot_layers=0, prot_bits=8, stride=1,
                                   protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3)
    pp = ns["DivergenceRecorder"](specs=[(2, 2, 0)], n_layers=4, prot_layers=0, prot_bits=8, stride=1, protect_prefix=-1)
    for r in (rec, pp): r.group, r.idx = "g", 0
    g = torch.Generator().manual_seed(0)
    key = torch.randn(1, 2, P, D, generator=g); q = torch.randn(1, 4, P, D, generator=g)
    for r in (rec, pp): r.record(_talker_module(groups, P, 0, D), q, key, key, None, D**-0.5)
    d, dp = dict(zip(_cols(), rec.rows[0])), dict(zip(_cols(), pp.rows[0]))
    assert d["prefix_codebook"] == "kmeans" and d["prefix_bits"] == 3 and dp["prefix_codebook"] == ""
    assert 0 < d["attn_js"] and d["cos_k"] < 1.0 and d["relmse_k"] > 0      # prompt lossy, measured
    assert dp["attn_js"] < 1e-6                                            # fp16 prompt still exact
    assert rec.errors == 0

def test_recorder_compressor_cache_key_includes_effective_prefix_bits():
    ns = _recorder_ns()
    rec = ns["DivergenceRecorder"](specs=[(8, 8, 0), (2, 2, 0)], n_layers=4, prot_layers=0, prot_bits=8, stride=1,
                                   protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=0)
    a = rec._comp(0, 8, 8, 0, 16, "cpu"); b = rec._comp(0, 2, 2, 0, 16, "cpu")
    assert a.prefix_bits_effective == 8 and b.prefix_bits_effective == 2

def test_wav_path_and_make_config_carry_codebook(tmp_path):
    wav_path = _single("_wav_path", {"os": os})
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, True, -1, "kmeans", 3).endswith("_pl=0_g=32_pre_pq=k3.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 8, 8, 0, 0, 32, True, -1, "minmax", 0).endswith("_pq=m8.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, True, -1).endswith("_pp=prompt.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, None, None, None, 0, 0, False, -1, "kmeans", 2).endswith("_fp16.wav")
    make_config = _single("make_config", {"TurboQuantConfig": TurboQuantConfig})
    c = make_config(2, 2, 0, 0, 32, True, -1, "kmeans", 2)
    assert c.prefix_codebook == "kmeans" and c.prefix_bits == 2

def test_load_done_filters_codebook_and_old_csv(tmp_path):
    pd = pytest.importorskip("pandas"); load_done = _single("load_done", {"pd": pd, "os": os}); cols = _cols()
    rows = [("g", 0, 0, 10, 2, 2, 0, 0, 32, 1, -1, "", 0) + tuple([0.1] * 9),
            ("g", 1, 0, 10, 2, 2, 0, 0, 32, 1, -1, "kmeans", 2) + tuple([0.1] * 9)]
    p = tmp_path / "d.csv"; pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    assert load_done(str(p), 0, 32, True, -1) == {("g", 0)}
    assert load_done(str(p), 0, 32, True, -1, "kmeans", 2) == {("g", 1)}
    old = tmp_path / "old.csv"
    pd.DataFrame(rows, columns=cols).drop(columns=["prefix_codebook", "prefix_bits"]).to_csv(old, index=False)
    assert load_done(str(old), 0, 32, True, -1, "kmeans", 2) == set()
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_prefix_codebook.py -q -k "recorder or wav_path or load_done"` → FAIL.

- [ ] **Step 3: Implement**

Columns and `_N_ID`; recorder stores both fields, `_comp` key becomes `(layer, kb, vb, rw)` still (the effective prefix bits follow `kb` inside `TurboQuantV3`), passes them through; in `record`, when `prefix_codebook` and `pfx > 0`: `pk, pv = comp.compress_kv(key_in[:, :, :pfx], value[:, :, :pfx], prefix=pfx)` → `decompress_kv` → re-RoPE when `rope` → splice `rk[:, :, :pfx] = that` (values likewise) instead of the fp16 splice, and set the recon lower bound to 0 instead of `pfx`; row tuple appends `self.prefix_codebook, self.prefix_bits`. `_wav_path`, `load_done`, `audio_done`, `make_config`, the two call sites in `run_experiment`, CLI flags, banner print, recorder construction. Update the two test column lists.

- [ ] **Step 4: Run** — `pytest tests/test_prefix_codebook.py tests/test_kv_attn_divergence.py tests/test_protect_prefix.py tests/test_keyfix.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py tests/test_prefix_codebook.py tests/test_kv_attn_divergence.py tests/test_protect_prefix.py
git commit -m "Divergence experiment: --prefix-codebook/--prefix-bits with quantized-prompt splice, _pq wav segment, CSV columns and resume filter"
```

---

### Task 7: Scorer, audio check, summaries and analysis

**Files:**
- Modify: `tools/score_wav_dir.py:71-91,116`, `tools/check_audio.py:34-37,66-67`, `tools/summarize_keyfix.py:22-27,113,138`, `tools/reduce_keyfix_div.py:36`, `tools/analyze_keyfix.py:25-33,172,213-235` (cells, pairs, section 8) plus two new sections 9-10
- Test: `tests/test_score_wav_dir.py`, `tests/test_check_audio.py`, `tests/test_prefix_codebook.py`

**Interfaces:**
- Consumes: wav segment `_pq=[km]\d+`, CSV columns from Task 6.
- Produces: `parse_wav_name(...)["config"]` = `K2V2@0+g32+pre+pk3` for `_g=32_pre_pq=k3`; `_keyfix_suffix(kg, pre, pp=None, pq=None)`; `summarize_keyfix._label(kb, vb, rw, kg, pre, pp=0, pq="", pb=0)`; cells `pk2`, `pk3`, `pm3` in `CELLS` of summarize and analyze (labels `kmeans prompt 2b + g32pre`, `kmeans prompt 3b + g32pre`, `minmax prompt 3b + g32pre`; pairs `("g32pre", c)` and `("g32pp", c)` for each); `effective_bits` gains `prefix_bits_per_elem: float | None` (when given, prefix elements cost that instead of 16); analyze section 9 "Runaway classification" (per cell x config: collapses, runaways `dur > 3x fp16`, stopped `< 1.5x`, median spk_sim of collapsed) and section 10 "Per-step maximum vs collapse" (within-cell AUC and Mann-Whitney p of `js_max`, `js_p95`, `l5_max` from `_bystep.csv`, as computed in the 2026-11-18 scratch analysis).

- [ ] **Step 1: Write the failing tests**

```python
def test_score_parses_pq_segment():
    d = sw.parse_wav_name("qwen_librispeech_pc_2_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32_pre_pq=k3.wav")
    assert d["config"] == "K2V2@0+g32+pre+pk3"
    assert sw.parse_wav_name("qwen_librispeech_pc_2_sampling_s0_t0.9_K2_V2_rw=0_pl=0_pq=m3.wav")["config"] == "K2V2@0+pm3"
    assert sw.parse_wav_name("qwen_librispeech_pc_2_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32_pp=prompt.wav")["config"] == "K2V2@0+g32+pp"

def test_check_audio_regex_pq():
    assert ca._NAME_RE.match("qwen_librispeech_pc_2_sampling_s0_t0.9_K2_V2_rw=0_pl=0_pq=k2.wav").group("pq") == "k2"

def test_summarize_label_with_codebook():
    assert sm._label(2, 2, 0, 32, 1, -1, "kmeans", 2) == "K2V2@0+g32+pre+pk2"
    assert sm._label(2, 2, 0, 32, 1, -1) == "K2V2@0+g32+pre+pp"

def test_effective_bits_with_prefix_cost():
    P, S = np.array([53.0]), np.array([303.0])
    assert an.effective_bits(2, 1.0, True, P, S, prefix_bits_per_elem=3.2) == pytest.approx((3.2 * 53 + 3 * 250) / 303)
    assert an.effective_bits(2, 1.0, True, P, S) == pytest.approx((16 * 53 + 3 * 250) / 303)

def test_runaway_classification_counts():
    scores = pd.DataFrame({"idx": [0, 0, 1, 1], "config": ["fp16", "K2V2@0+g32", "fp16", "K2V2@0+g32"],
                           "dur_s": [4.0, 30.0, 5.0, 5.2], "cer": [0.0, 0.9, 0.0, 0.6], "spk_sim": [1, 0.8, 1, 0.9]})
    r = an.runaway_table({"g32": scores})
    row = r[(r.cell == "g32") & (r.config == "K2V2@0")].iloc[0]
    assert row.n_collapse == 2 and row.runaway == 1 and row.stopped == 1
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_score_wav_dir.py tests/test_check_audio.py tests/test_prefix_codebook.py -q -k "pq or label or effective or runaway"` → FAIL.

- [ ] **Step 3: Implement**

Regexes gain `(?:_pq=(?P<pq>[km]\d+))?` after the `_pp` group; `_keyfix_suffix` maps `k3` → `+pk3`, `m3` → `+pm3`; summarize `_label` and the two column loops add `prefix_codebook` (default `""`) and `prefix_bits` (default 0); reducer `CFG` gains both; analyze: cells, labels, two new styles, pairs, `effective_bits(..., prefix_bits_per_elem=None)`, section 8 computes `prefix_bits_per_elem = pb + meta_scalars * 16 / P.mean()` for codebook cells (`meta_scalars` from `turboquant.fitted_levels`), plus `runaway_table(scores: dict[str, DataFrame]) -> DataFrame` and `step_table(div_step: dict[str, DataFrame], scores) -> DataFrame` rendered as sections 9 and 10 (load `_bystep.csv` in `load`).

- [ ] **Step 4: Run** — `pytest tests/test_score_wav_dir.py tests/test_check_audio.py tests/test_prefix_codebook.py -q` → PASS; then `.venv/bin/python tools/analyze_keyfix.py --root results/keyfix --out /tmp/kf` runs on today's archive and prints sections 9 and 10 with the numbers from the 2026-11-18 scratch analysis (g32 K3V3: 14 collapses, 14 runaways; js_max AUC 0.49).

- [ ] **Step 5: Commit**

```bash
git add tools/score_wav_dir.py tools/check_audio.py tools/summarize_keyfix.py tools/reduce_keyfix_div.py tools/analyze_keyfix.py tests/test_score_wav_dir.py tests/test_check_audio.py tests/test_prefix_codebook.py
git commit -m "Tools: pk/pm prefix-codebook cells in scoring, summaries and analysis, with runaway and per-step-maximum sections"
```

---

### Task 8: Smoke, run script, run book

**Files:**
- Modify: `scripts/gpu/03_smoke_keyfix.sh:31,43-49`, `tools/validate_keyfix_smoke.py:20,28-42,113-134,149-153`, `scripts/gpu/14_qwen_keyfix.sh:41-50` and its header comment, `docs/keyfix_experiment.md` (cell table, run commands, outputs, bit accounting), `docs/keyfix_handoff.md` (next steps)
- Test: `tests/test_prefix_codebook.py` (validator helper)

**Interfaces:**
- Consumes: everything above.
- Produces: smoke cell `pk` = `--key-group 32 --pre-rope-keys --protect-prefix -1 --prefix-codebook kmeans --prefix-bits 0`; `validate_keyfix_smoke.CELLS["pk"] = (32, 1, -1, "kmeans", 0)` (tuple grows by two for every cell, old cells `("", 0)`), `_wav(wavdir, idx, cfg)` writes `_g=32_pre_pq=k<kb>` for `pk`; checks: columns `prefix_codebook`/`prefix_bits` present and equal on every row; K2V2 prefill rows `cos_k < 1` and mean `relmse_k` below plain's K2V2 prefill rows; K8V8 prefill rows `cos_k > 0.999`; K2V2 wav differs from plain and pp. Run cells `pk2`, `pk3`, `pm3` in `cell_flags`.

- [ ] **Step 1: Write the failing test**

```python
def test_validator_wav_name_for_pk():
    v = load("tools/validate_keyfix_smoke.py", "validate_pk")
    assert v._wav("w", 1, (2, 2, "pk")).endswith("_K2_V2_rw=0_pl=0_g=32_pre_pq=k2.wav")
    assert v._wav("w", 1, (8, 8, "pk")).endswith("_pq=k8.wav")
    assert v._wav("w", 1, (2, 2, "pp")).endswith("_pp=prompt.wav")
    assert v.CELLS["pk"] == (32, 1, -1, "kmeans", 0) and v.CELLS["plain"] == (0, 0, 0, "", 0)
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_prefix_codebook.py -q -k validator` → FAIL.

- [ ] **Step 3: Implement the validator, both scripts and the docs**

Validator per the Interfaces block (the `pp` zero-divergence check stays as is). `03_smoke_keyfix.sh`: `ALL_CELLS="plain g32 pre pp pk"` and the case line. `14_qwen_keyfix.sh`: three cases, header comment lists them, default `CELLS` unchanged. Run book: new rows in the cell table (`pk2`, `pk3`, `pm3` with their flags and prompt cost 3.2 / 5.4 / 3.6 bits per element), the run line `CELLS="pk2 pk3 pm3" CFGS="fp16,K2V2@0" MAXPG=50 nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix_pq.log 2>&1 &`, the `_pq` wav segment and `+pk`/`+pm` labels in the outputs table, the prefix metadata rule in "Bit accounting". Handoff: replace step 2b's "still to do" with the new cells.

- [ ] **Step 4: Run the whole CPU suite and lint**

Run: `.venv/bin/python -m pytest -x -q` and `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
Expected: all PASS, clean.

- [ ] **Step 5: Commit and push**

```bash
git add scripts/gpu/03_smoke_keyfix.sh scripts/gpu/14_qwen_keyfix.sh tools/validate_keyfix_smoke.py docs/keyfix_experiment.md docs/keyfix_handoff.md tests/test_prefix_codebook.py
git commit -m "Smoke cell pk and run cells pk2/pk3/pm3 for the prefix codebook experiment, with run book and handoff updates"
gh auth switch -u pancarnas && git push origin exp/keyfix-group-prerope; gh auth switch -u vildan-huseynov-m10
```

Then on the box: `git pull`, `bash scripts/gpu/03_smoke_keyfix.sh` must print `KEYFIX SMOKE PASS` before the full run.

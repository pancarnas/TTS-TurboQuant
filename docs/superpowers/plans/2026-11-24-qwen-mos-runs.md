# Qwen MOS objective runs: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and score the audio for the Qwen listening study on the 195-sentence, 39-speaker librispeech_pc list: pm4 at fp16 / K2V2 / K3V2 / K3V3, g32pre and g32pp at fp16 / K2V2.

**Architecture:** The Qwen recorder gets `--idx-file` (filter the enumerated sentences by index, as the VALL-E benchmark does) and a per-config wav skip in audio-only resume. The keyfix run script exposes both as env knobs. The sentence list is produced on the box by the existing selector and committed under `docs/`. No library code changes.

**Tech Stack:** Python 3.12, pytest (CPU, model-free), bash. Box: SageMaker L4, `scripts/gpu/14_qwen_keyfix.sh`.

**Spec:** `docs/superpowers/specs/2026-11-24-qwen-mos-sentence-set-design.md`

## Global Constraints

- No edits under `turboquant/`.
- Laptop tests must pass without torch: load recorder functions with `tests.test_keyfix._single`.
- Commits one sentence, no Claude attribution; push via the `pancarnas` gh account.
- Size limits: functions 50 lines, files 300 lines (the recorder is already over; add, do not grow functions past 50).
- Box pitfalls (handoff): no curly quotes, no `=====` echo, 24 MiB download limit.

## Review Focus

1. An idx file with blank lines or `#` comments must not crash the loader (Task 1 test).
2. `--idx-file` with the default `--max-per-group 100` would silently drop idxs above 99; with an idx file the enumeration must cover the whole group (Task 1 test on the resolved slice).
3. Audio-only resume re-generates every config of a sentence when one wav is missing (the first 50 pm4 sentences lack K3V2); the per-config skip must keep existing wavs byte-identical and only fill the gap (Task 2, verified on the smoke).
4. `IDXFILE` unset must leave the script's behaviour exactly as today (Task 3, `bash -n` plus a dry echo).
5. The scorer resumes on the cell's scores CSV, so the 195-list scores land in the same file as the first-50 ones; the summary and the thesis table must filter on the idx list (noted for the test-prep plan, not this one).

---

### Task 1: `--idx-file` in the Qwen recorder

**Files:**
- Modify: `models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py` (near `pending_items`, line 622; the items list, line 691; argparse, line 909)
- Test: `tests/test_keyfix.py`

**Interfaces:**
- Produces: `load_idx_file(path: str | None) -> set[int] | None` (None when path is falsy; ints, one per line, blank lines and `#` comments skipped) and `select_idxs(items: list, allowed: set[int] | None) -> list` (identity when allowed is None, else keep tuples whose `t[1]` is in allowed).

- [ ] **Step 1: Write the failing tests**

```python
def test_load_idx_file_skips_blanks_and_comments(tmp_path):
    load_idx_file = _single("load_idx_file", {"os": os})
    p = tmp_path / "idx.txt"
    p.write_text("# head\n3\n\n7\n 11 \n")
    assert load_idx_file(str(p)) == {3, 7, 11}
    assert load_idx_file(None) is None


def test_select_idxs_filters_on_index():
    select_idxs = _single("select_idxs", {})
    items = [("g", 0, "a"), ("g", 1, "b"), ("g", 2, "c")]
    assert select_idxs(items, {1}) == [("g", 1, "b")]
    assert select_idxs(items, None) == items
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./.venv/bin/pytest tests/test_keyfix.py -k "idx_file or select_idxs" -v`
Expected: FAIL with `ValueError: substring not found` from `_single` (function absent).

- [ ] **Step 3: Implement both functions next to `pending_items`**

`load_idx_file` is the VALL-E `_load_idx_file` body (10 lines). `select_idxs` is one comprehension. In `main`, add `parser.add_argument("--idx-file", default=None, help="Restrict to the sentence idxs listed in this file (one per line, # comments); implies the whole group is enumerated, --max-per-group ignored.")`, then build `items` with `iter_eval_items([g], None if args.idx_file else args.max_per_group, args.data_dir)` and wrap it in `select_idxs(items, load_idx_file(args.idx_file))`. Print `restricting to N idxs from <path>` when set.

- [ ] **Step 4: Run the tests and the full suite**

Run: `./.venv/bin/pytest tests/ -x -q`
Expected: all pass (215 + 2).

- [ ] **Step 5: Commit**

```bash
git add models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py tests/test_keyfix.py
git commit -m "Add --idx-file to the Qwen recorder so a run can target a sentence list"
```

---

### Task 2: per-config wav skip in audio-only resume

**Files:**
- Modify: `models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py:756-800` (the `for kb, vb, rw in specs` loop)
- Test: `tests/test_keyfix.py`

**Interfaces:**
- Produces: `wav_present(path: str) -> bool` (exists and non-empty), used in the loop as `if resume and no_div and wav_present(path): continue` after computing the wav path once before `run_generation`.

- [ ] **Step 1: Write the failing test**

```python
def test_wav_present_requires_nonempty_file(tmp_path):
    wav_present = _single("wav_present", {"os": os})
    empty, full = tmp_path / "e.wav", tmp_path / "f.wav"
    empty.write_bytes(b"")
    full.write_bytes(b"RIFF")
    assert not wav_present(str(tmp_path / "missing.wav"))
    assert not wav_present(str(empty))
    assert wav_present(str(full))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/bin/pytest tests/test_keyfix.py -k wav_present -v`
Expected: FAIL, function absent.

- [ ] **Step 3: Implement**

Add `wav_present` next to `audio_done`. In the loop, compute `path = _wav_path(...)` once (the same 14 arguments as the existing `sf.write` call), add the skip guard, and pass `path` to `sf.write`. Net change: the `_wav_path(...)` call moves up, three lines are added.

- [ ] **Step 4: Run the suite**

Run: `./.venv/bin/pytest tests/ -x -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py tests/test_keyfix.py
git commit -m "Skip existing wavs per config in audio-only resume so a new config fills only its gap"
```

---

### Task 3: `IDXFILE` and `NODIV` knobs in the keyfix script

**Files:**
- Modify: `scripts/gpu/14_qwen_keyfix.sh` (knob block lines 39-44, the recorder call lines 65-72, the header comment)

**Interfaces:**
- Consumes: `--idx-file`, `--no-divergence` from Tasks 1 and 2.
- Produces: env knobs `IDXFILE=""` (path; when set, passed as `--idx-file "$IDXFILE"`) and `NODIV=0` (when `1`, passes `--no-divergence` and skips the `measurement errors` grep, since no divergence pass runs).

- [ ] **Step 1: Add the knobs**

Default `IDXFILE=""`, `NODIV=0`. Build `EXTRA=""`; append `--idx-file "$IDXFILE"` when non-empty and `--no-divergence` when `NODIV=1`; add `$EXTRA` to the recorder call next to `$FLAGS`. Guard the `measurement errors` check with `[ "$NODIV" != "1" ]`. Document both in the header comment with the MOS run line from Step 3.

- [ ] **Step 2: Syntax check**

Run: `bash -n scripts/gpu/14_qwen_keyfix.sh && grep -c -- '--idx-file "\$IDXFILE"\|--no-divergence' scripts/gpu/14_qwen_keyfix.sh`
Expected: no syntax error and the count is 2.

- [ ] **Step 3: Add the MOS run lines to `docs/keyfix_experiment.md`**

New section "MOS objective run (39 speakers, 195 sentences)" after "Getting the results off the box", with: the selector command on the box (`$PY tools/select_mos_sentences.py --data-dir data --per-speaker 5 --min-dur 4 --max-dur 9 --out docs/qwen_mos_idx_list.txt`, then commit the file), a 2-sentence smoke (`head -2 docs/qwen_mos_idx_list.txt > /tmp/idx2.txt; CELLS="pm4" CFGS="fp16,K3V2@0" IDXFILE=/tmp/idx2.txt NODIV=1 SCORE=0 bash scripts/gpu/14_qwen_keyfix.sh` and the two expected wav names), the full run (`CELLS="pm4" CFGS="fp16,K2V2@0,K3V2@0,K3V3@0" IDXFILE=docs/qwen_mos_idx_list.txt NODIV=1 nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix_mos_pm4.log 2>&1 &` then `CELLS="g32pre g32pp" CFGS="fp16,K2V2@0"` the same way), the progress check (`ls outputs/keyfix_pm4 | grep -c _pq=m4` reaches 585), and what to download (`results/qwen_keyfix_{pm4,g32pre,g32pp}_scores.csv` plus the three score logs, all under 1 MB).

- [ ] **Step 4: Commit**

```bash
git add scripts/gpu/14_qwen_keyfix.sh docs/keyfix_experiment.md
git commit -m "Expose IDXFILE and NODIV in the keyfix run script with the MOS objective run lines"
```

---

### Task 4: sentence list committed from the box

**Files:**
- Create: `docs/qwen_mos_idx_list.txt` (on the box, by the selector; committed and pushed from there, or pasted here)

- [ ] **Step 1: On the box, after `git pull`, run the selector**

Run: `$PY tools/select_mos_sentences.py --data-dir data --per-speaker 5 --min-dur 4 --max-dur 9 --out docs/qwen_mos_idx_list.txt`
Expected: `wrote 195 idxs across 39 speakers` (fewer only if a speaker has no 4-9 s sentence; the tool prints which).

- [ ] **Step 2: Commit the list**

```bash
git add docs/qwen_mos_idx_list.txt
git commit -m "Add the 39-speaker librispeech_pc sentence list for the Qwen MOS runs"
```

- [ ] **Step 3: Smoke, then launch the two runs as in the run book; report `tail -2` of each log and the wav counts.**

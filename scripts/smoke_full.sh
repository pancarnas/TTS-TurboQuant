#!/bin/bash
# Full pre-flight smoke: generate (+divergence) -> score -> AUTO-VALIDATE.
# Exercises both bit axes (K4V4/K4V2/K3V3/K2V2 + fp16) at 2 seeds with the
# English preset, then asserts the metrics are internally correct. Prints a
# single PASS/FAIL verdict. Run on a GPU machine (direct, no SGE) before
# launching the heavy Eddie batch.
#
#   bash scripts/smoke_full.sh [N_SENTENCES] [PRESET]
#   bash scripts/smoke_full.sh 4 librispeech_1.npz
set -euo pipefail

N="${1:-4}"
PRESET="${2:-librispeech_1.npz}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs results

TAG=smoke_full
SUB=smoke_full
DIV="results/${TAG}_div.csv"
SCORES="results/${TAG}_scores.csv"
CFGS="fp16,K4V4@0,K4V2@0,K3V3@0,K2V2@0"

echo "### 1/3  generate + divergence  (preset=$PRESET, $N sents x 5 configs x 2 seeds) ###"
python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device cuda \
  --groups librispeech_pc --max-per-group "$N" --data-dir data --no-quality \
  --configs "$CFGS" --protected-layers 0 --output-subdir "$SUB" \
  --seeds 0,1 --decode sampling --preset "$PRESET" \
  --record-divergence --divergence-out "$DIV" --divergence-stride 16 \
  --num-shards 1 --shard-id 0 --run-tag "$TAG"

echo -e "\n### 2/3  score (Whisper large-v3 CER/WER) ###"
python tools/score_wav_dir.py \
  --audio-dir "models/VALL-E-X/benchmarks/outputs/$SUB" \
  --data-dir data --device cuda --out "$SCORES"

echo -e "\n### 3/3  validate ###"
python3 - "$DIV" "$SCORES" "models/VALL-E-X/benchmarks/outputs/$SUB" "$N" <<'PY'
import sys, glob, os, re, pandas as pd
div, sc, wavdir, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ok = True
def check(name, cond):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# --- wavs ---
nwav = len(glob.glob(os.path.join(wavdir, "*.wav")))
check(f"wav count == {N}x5x2 ({N*10})", nwav == N*10)
seeds = {int(m.group(1)) for f in glob.glob(os.path.join(wavdir, "*.wav"))
         if (m := re.search(r"_s(\d+)_", os.path.basename(f)))}
check("both seeds saved (0 and 1)", {0, 1} <= seeds)

# --- divergence correctness (both bit axes) ---
d = pd.read_csv(div)
d['cfg'] = 'K'+d.key_bits.astype(str)+'V'+d.value_bits.astype(str)
g = d.groupby('cfg')[['cos_k','cos_v','attn_js','attn_top1','relmse_k','relmse_v']].mean()
print(g.round(4).to_string())
check("cos_k: K4V4 > K2V2   (key bits drive cos_k)", g.loc['K4V4','cos_k'] > g.loc['K2V2','cos_k'] + 0.01)
check("cos_k: K4V4 ~= K4V2  (same keys, |d|<0.02)",  abs(g.loc['K4V4','cos_k']-g.loc['K4V2','cos_k']) < 0.02)
check("cos_v: K4V4 > K4V2   (value bits drive cos_v)", g.loc['K4V4','cos_v'] > g.loc['K4V2','cos_v'] + 0.01)
check("cos_v: K4V2 ~= K2V2  (same values, |d|<0.02)", abs(g.loc['K4V2','cos_v']-g.loc['K2V2','cos_v']) < 0.02)
check("attn_js: K2V2 > K4V4 (worse keys shift attn)", g.loc['K2V2','attn_js'] > g.loc['K4V4','attn_js'])
check("relmse_k: K2V2 > K4V4", g.loc['K2V2','relmse_k'] > g.loc['K4V4','relmse_k'])
check("all cos in [0.5,1]", ((g[['cos_k','cos_v']] >= 0.5) & (g[['cos_k','cos_v']] <= 1.0)).all().all())

# --- audio scores ---
s = pd.read_csv(sc)
check("both seeds present in scores", {0,1} <= set(s.seed.unique()))
fp = s[s.config == 'fp16'].wer.mean()
check(f"fp16 WER < 0.20 (English preset; got {fp:.3f})", fp < 0.20)
w = s.groupby('config').wer.mean()
check("K2V2@0 WER >= fp16 WER", w.get('K2V2@0', 9) >= fp)
print("\nper-config CER/WER:")
print(s.groupby('config')[['cer','wer']].mean().round(3).to_string())

print("\n===== SMOKE " + ("PASS — safe to launch the heavy Eddie batch"
      if ok else "FAIL — DO NOT launch heavy yet; inspect above") + " =====")
sys.exit(0 if ok else 1)
PY

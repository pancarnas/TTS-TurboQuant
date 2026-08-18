#!/bin/bash
# Listening demo: ONE sentence, cloned from its real reference voice, generated
# under ALL 28 configs (fp16 + K{4,3,2}xV{4,3,2} x rw{0,64,128}) — so a human
# (e.g. the thesis supervisor) can hear the quality ladder directly.
#
#   bash scripts/gpu/40_demo_samples.sh            # sentence idx 0
#   IDX=17 bash scripts/gpu/40_demo_samples.sh     # a different sentence
#   PL=2  bash scripts/gpu/40_demo_samples.sh      # with protected layers
#
# Needs data/ (01_fetch_data.sh). ~28 generations of one sentence: 10-25 min.
#
# Output: demo_listen_idx<IDX>/
#   00_reference_prompt.wav   the real voice being cloned (the prompt)
#   00_ground_truth.wav       the real recording of the target sentence
#   fp16.wav, K4V4@0.wav ...  one wav per config
#   listen.txt                target text + reference text + file map
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs results

IDX="${IDX:-0}"
PL="${PL:-0}"
SUB="demo_idx${IDX}_pl${PL}"
LISTEN_DIR="demo_listen_idx${IDX}"

GRID="fp16"
for K in 4 3 2; do for V in 4 3 2; do for RW in 0 64 128; do
  GRID="$GRID,K${K}V${V}@${RW}"
done; done; done

IDXFILE="$(mktemp)"
echo "$IDX" > "$IDXFILE"

echo "### generating sentence idx=$IDX under 28 configs (pl=$PL, clone mode) ###"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device cuda \
  --groups librispeech_pc --data-dir data --no-quality \
  --configs "$GRID" --protected-layers "$PL" --output-subdir "$SUB" \
  --seeds 0 --decode sampling --preset librispeech_1.npz --voice-mode clone \
  --idx-file "$IDXFILE" --num-shards 1 --shard-id 0 --run-tag "$SUB"
rm -f "$IDXFILE"

echo "### packaging $LISTEN_DIR ###"
python - "$IDX" "models/VALL-E-X/benchmarks/outputs/$SUB" "$LISTEN_DIR" <<'EOF'
import glob, os, re, shutil, sys

idx, wavdir, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, os.getcwd())
from turboquant.eval_sentences import load_librispeech_pc

item = load_librispeech_pc("data")[idx]
os.makedirs(out, exist_ok=True)
shutil.copy(item.ref_audio, os.path.join(out, "00_reference_prompt.wav"))
if item.ground_truth_audio and os.path.exists(item.ground_truth_audio):
    shutil.copy(item.ground_truth_audio, os.path.join(out, "00_ground_truth.wav"))

pat = re.compile(rf"vallex_librispeech_pc_{idx}_sampling_s0(?:_t[\d.]+)?_(.+)\.wav$")
copied = []
for f in sorted(glob.glob(os.path.join(wavdir, "*.wav"))):
    m = pat.search(os.path.basename(f))
    if m:
        cfg = m.group(1)
        shutil.copy(f, os.path.join(out, f"{cfg}.wav"))
        copied.append(cfg)

with open(os.path.join(out, "listen.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"Sentence idx {idx} (LibriSpeech-PC, zero-shot clone)\n\n")
    fh.write(f"TARGET TEXT   : {item.text}\n")
    fh.write(f"REFERENCE TEXT: {item.ref_text}\n\n")
    fh.write("00_reference_prompt.wav = the real voice being cloned\n")
    fh.write("00_ground_truth.wav     = the real recording of the target text\n")
    fh.write("fp16.wav                = uncompressed VALL-E-X baseline\n")
    fh.write("K<k>V<v>@<rw>.wav       = key/value bits + residual window\n\n")
    fh.write(f"configs generated ({len(copied)}): {', '.join(copied)}\n")

print(f"{out}/: {len(copied)} config wavs + reference + ground truth + listen.txt")
EOF

echo "done — zip and share: zip -r ${LISTEN_DIR}.zip ${LISTEN_DIR}"

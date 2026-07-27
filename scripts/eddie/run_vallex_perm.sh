#!/bin/bash
# Sharded VALL-E-X AR/NAR/both permutation run (one GPU, N parallel workers).
#
# Submit via the generic runner; args = shard count (default 3), max sentences
# per group (default 50), groups (default the three standard sets), run tag,
# and an optional 5th arg overriding the config list:
#   qsub -N vx_kv_perm -l h_rt=12:00:00 -pe sharedmem 4 scripts/eddie/eddie_run.sh \
#     bash scripts/eddie/run_vallex_perm.sh 3 100
#   qsub -N vx_long -l h_rt=24:00:00 -pe sharedmem 4 scripts/eddie/eddie_run.sh \
#     bash scripts/eddie/run_vallex_perm.sh 3 50 libritts_long vallex_long
#   # custom config grid (arg 5); quote it — it contains commas/@:
#   bash scripts/eddie/run_vallex_perm.sh 6 100 seedtts_en,librispeech_pc,ellav_hard \
#     vallex_grid "fp16,K4V4@0,K4V2@64,K2V2@128"
#
# Each worker takes every N-th work item (--num-shards/--shard-id round-robin),
# writes its own log (logs/<tag>_shard<i>.log) and trials CSV (shared run-tag).
# Rerunning with a larger max-per-group regenerates earlier sentences
# byte-identically (paired seeds) and adds the new ones; the post-hoc scorer's
# --resume skips already-scored rows. No inline metrics.
set -euo pipefail

N="${1:-3}"
MAXPG="${2:-50}"
# Reduce CUDA fragmentation — long-sequence NAR passes allocate/free big
# transient buffers; without this the allocator OOMs well below capacity.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: not named GROUPS — that's a bash builtin (user's group ids) and
# assignments to it are silently ignored.
EVAL_GROUPS="${3:-seedtts_en,librispeech_pc,ellav_hard}"
TAG="${4:-vallex_kv_perm_pl2}"
export OMP_NUM_THREADS=2

# Default config list (arg 5 overrides it wholesale — quote the override).
DEFAULT_CFGS="fp16,K4V4@0,K4V4@64,K4V4@128,K4V3@0,K3V3@64,K3V3@128"
DEFAULT_CFGS="$DEFAULT_CFGS,nar:K4V4@0,nar:K4V4@64,nar:K4V4@128,nar:K3V3@128"
DEFAULT_CFGS="$DEFAULT_CFGS,both:K4V4@0,both:K4V4@64,both:K4V4@128,both:K3V3@128"
CFGS="${5:-$DEFAULT_CFGS}"
# arg 6: protected layers (default 2). arg 7: output subdir — REQUIRED when
# rerunning the same configs at a different pl, since wav names don't encode pl
# (else the pl=0 run overwrites the pl=2 wavs). arg 8: seeds (default 0) —
# e.g. "0,1,2" for multi-seed error bars (each seed is a paired draw per config).
PL="${6:-2}"
SUBDIR="${7:-}"
SEEDS="${8:-0}"
# arg 9: voice preset. DEFAULT IS NOW librispeech_1.npz (English) — alan.npz is
# Japanese and synthesizes English text cross-lingually, which inflated WER ~5x.
PRESET="${9:-librispeech_1.npz}"

for i in $(seq 0 $((N - 1))); do
  python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device cuda \
    --groups "$EVAL_GROUPS" --max-per-group "$MAXPG" \
    --data-dir data --no-quality --configs "$CFGS" \
    --protected-layers "$PL" ${SUBDIR:+--output-subdir "$SUBDIR"} \
    --seeds "$SEEDS" --decode sampling --preset "$PRESET" \
    --num-shards "$N" --shard-id "$i" --run-tag "$TAG" \
    > "logs/${TAG}_shard$i.log" 2>&1 &
done
wait
echo "all $N shards done"

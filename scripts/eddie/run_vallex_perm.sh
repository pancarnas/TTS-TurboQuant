#!/bin/bash
# Sharded VALL-E-X AR/NAR/both permutation run (one GPU, N parallel workers).
#
# Submit via the generic runner; args = shard count (default 3) and
# max sentences per group (default 50):
#   qsub -N vx_kv_perm -l h_rt=12:00:00 -pe sharedmem 4 scripts/eddie/eddie_run.sh \
#     bash scripts/eddie/run_vallex_perm.sh 3 100
#
# Each worker takes every N-th work item (--num-shards/--shard-id round-robin),
# writes its own log (logs/vx_shard<i>.log) and trials CSV (shared run-tag).
# Rerunning with a larger max-per-group regenerates earlier sentences
# byte-identically (paired seeds) and adds the new ones; the post-hoc scorer's
# --resume skips already-scored rows. No inline metrics.
set -euo pipefail

N="${1:-3}"
MAXPG="${2:-50}"
export OMP_NUM_THREADS=2

CFGS="fp16,K4V4@0,K4V4@64,K4V4@128,K4V3@0,K3V3@64,K3V3@128"
CFGS="$CFGS,nar:K4V4@0,nar:K4V4@64,nar:K4V4@128,nar:K3V3@128"
CFGS="$CFGS,both:K4V4@0,both:K4V4@64,both:K4V4@128,both:K3V3@128"

for i in $(seq 0 $((N - 1))); do
  python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device cuda \
    --groups seedtts_en,librispeech_pc,ellav_hard --max-per-group "$MAXPG" \
    --data-dir data --no-quality --configs "$CFGS" \
    --protected-layers 2 --seeds 0 --decode sampling \
    --num-shards "$N" --shard-id "$i" --run-tag vallex_kv_perm_pl2 \
    > "logs/vx_shard$i.log" 2>&1 &
done
wait
echo "all $N shards done"

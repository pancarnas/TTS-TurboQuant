#!/bin/bash
# Aggregate scores + divergence into the thesis tables and figures.
# CPU-only (pandas/scipy/matplotlib) — no GPU or torch needed, safe to run on
# a laptop with just the results/ CSVs copied over.
#
#   bash scripts/gpu/30_analyze.sh
#
# Outputs (results/): clone_seed_errorbars.md, master_cl_pl0.csv,
#   cross_model_ci.md, clone_rw_significance.md, combined_clone_summary.csv,
#   vallex_layer_summary_cl.csv, qwen_layer_summary.csv, figures/*.png
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p results/figures

concat_shards() {  # concat_shards OUT SHARD1 [SHARD2 ...] — header once
  local out="$1"; shift
  head -n 1 "$1" > "$out"
  for f in "$@"; do tail -n +2 "$f" >> "$out"; done
  echo "  $out <- $# shard(s)"
}

echo "### concat divergence shards ###"
concat_shards results/vallex_grid_cl_pl0_div.csv results/grid_cl_pl0_div_shard*.csv
concat_shards results/vallex_seed_cl_pl0_div.csv results/seed_cl_pl0_div_shard*.csv

echo "### per-layer divergence summaries (chunked, big CSVs ok) ###"
python tools/layer_summary.py \
  --divergence results/vallex_grid_cl_pl0_div.csv \
  --out results/vallex_layer_summary_cl.csv
python tools/layer_summary.py \
  --divergence results/qwen_div_seed0.csv \
  --out results/qwen_layer_summary.csv

echo "### multi-seed error bars (pl0 vs pl2) ###"
python tools/aggregate_seeds.py \
  --scores pl0=results/seed_cl_pl0_scores.csv pl2=results/seed_cl_pl2_scores.csv \
  --exclude-groups ellav_hard \
  --out results/clone_seed_errorbars.md \
  --img results/figures/clone_seed_errorbars.png

echo "### master per-config table (audio + PPL + attention distance) ###"
python tools/merge_metrics.py \
  --scores results/grid_cl_pl0_scores.csv \
  --ppl results/vallex_ppl_cl.csv \
  --divergence results/vallex_ppl_div_cl.csv \
  --group librispeech_pc \
  --out results/master_cl_pl0.csv

echo "### cross-model table: WER +/- sd + collapse Wilson CIs ###"
python tools/cross_model_ci.py --group librispeech_pc \
  --qwen results/qwen_seed_scores.csv \
  --valle results/seed_cl_pl0_scores.csv \
  --out results/cross_model_ci.md

echo "### residual-window significance (paired Wilcoxon, Holm) ###"
python tools/rw_analysis.py \
  --scores results/grid_cl_pl0_scores.csv \
  --out results/clone_rw_significance.md \
  --img-dir results/figures

echo "### cross-model combined summary (raw divergence CSVs, streamed) ###"
python tools/combined_summary.py \
  --model "qwen:results/qwen_seed_scores.csv:results/qwen_div_seed0.csv" \
  --model "valle:results/seed_cl_pl0_scores.csv:results/vallex_seed_cl_pl0_div.csv" \
  --out results/combined_clone_summary.csv \
  --markdown results/combined_clone_summary.md

echo "### figures ###"
for M in cer wer; do
  python tools/plot_kv_pl.py \
    --scores pl0=results/grid_cl_pl0_scores.csv pl2=results/grid_cl_pl2_scores.csv \
    --metric "$M" --out "results/figures/clone_kv_pl_${M}.png"
  python tools/plot_tradeoff.py \
    --scores pl0=results/grid_cl_pl0_scores.csv=0 pl2=results/grid_cl_pl2_scores.csv=2 \
    --metric "$M" --out "results/figures/clone_tradeoff_${M}.png"
done
python tools/plot_layer_profile.py \
  --valle results/vallex_layer_summary_cl.csv \
  --qwen results/qwen_layer_summary.csv \
  --config K4V4@0 --out results/figures/layer_profile_attnjs.png
python tools/plot_cross_model_collapse.py \
  --summary results/combined_clone_summary.csv --group librispeech_pc \
  --out results/figures/cross_model_collapse.png

echo "analysis done — see results/ and results/figures/"

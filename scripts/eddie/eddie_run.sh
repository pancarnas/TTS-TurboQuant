#!/bin/bash
# Generic Eddie (ECDF / Grid Engine) batch runner for TTS-TurboQuant.
#
# Everything you pass after the script name is run verbatim as the command, e.g.
#   qsub scripts/eddie/eddie_run.sh make eval-qwen DEVICE=cuda METRICS_DEVICE=cuda
#   qsub scripts/eddie/eddie_run.sh make smoke-qwen DEVICE=cuda
#   qsub scripts/eddie/eddie_run.sh python models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
#        --residual-windows 24,0 --out results/divergence_eddie.csv
#
# Per-submit overrides take precedence over the #$ defaults below, e.g.
#   qsub -N qwen_eval -l h_rt=8:00:00 scripts/eddie/eddie_run.sh make eval-qwen DEVICE=cuda
#
# Grid Engine options (lines prefixed with #$)
#$ -cwd
#$ -l h_rt=6:00:00            # max runtime — override: qsub -l h_rt=8:00:00 ...
#$ -q gpu
#$ -l gpu=1                   # IMPORTANT: exactly one GPU (do not request more)
#$ -P ppls_ssgpu             # priority project code
#$ -N tq_run                  # job name — override: qsub -N my_name ...
#$ -M sXXXXXXX@ed.ac.uk      # EDIT: your University email address
#$ -m beas                    # email on begin/end/abort/suspend
#$ -o logs/                   # stdout dir (created by setup_eddie.sh)
#$ -e logs/                   # stderr dir

set -euo pipefail

# Initialise the environment modules and activate the shared (read-only) env.
. /etc/profile.d/modules.sh
module load anaconda
conda activate py312torch27cuda118

# Repo-local user-site for the pip --user deps installed by setup_eddie.sh.
# Kept inside the project dir so it lives on group storage, not the home quota.
export PYTHONUSERBASE="${TQ_USERBASE:-$PWD/.eddie-userbase}"
export PATH="$PYTHONUSERBASE/bin:$PATH"

# Expose the repo's pure-Python packages (turboquant, qwen_tts, VALL-E-X) without
# installing into the shared env. turboquant lives at the repo root.
export PYTHONPATH="$PWD:$PWD/models/Qwen3-TTS:$PWD/models/VALL-E-X:${PYTHONPATH:-}"

# Persistent HuggingFace cache so multi-GB model/data downloads survive between
# jobs and don't fill the home quota.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

echo "=== node: $(hostname)  date: $(date) ==="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

if [ "$#" -eq 0 ]; then
    echo "ERROR: no command given. Example:" >&2
    echo "  qsub scripts/eddie/eddie_run.sh make eval-qwen DEVICE=cuda" >&2
    exit 2
fi

echo "=== running: $* ==="
exec "$@"

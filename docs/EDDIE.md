# Running TTS-TurboQuant on Eddie (ECDF / Grid Engine)

This runbook covers running the benchmarks on the Edinburgh **Eddie** compute
cluster using **Grid Engine (`qsub`)** and the **shared conda env**
`py312torch27cuda118` (Python 3.12 / torch 2.7 / CUDA 11.8).

The local `uv`-based workflow in the main `README.md` is unchanged. This is an
additive launch layer:

| File | Purpose |
| --- | --- |
| `scripts/eddie/setup_eddie.sh` | One-time setup (run on a login node). |
| `scripts/eddie/eddie_run.sh` | Generic `qsub` job script — runs whatever command you pass it. |
| `make eddie-setup` / `make eddie-submit` | Convenience wrappers. |

## Design notes (why it works this way)

- The shared env is **read-only**, so we never `pip install` the repo's deps into
  it. The repo's pure-Python packages (`turboquant`, `qwen_tts`, VALL-E-X) are put
  on `PYTHONPATH`; only the *missing* third-party deps go into a repo-local
  user-site (`.eddie-userbase`). **torch is never reinstalled** — the shared env's
  GPU build is used as-is.
- **One GPU per job** (`#$ -l gpu=1`). Do not request more — it burns GPU hours
  faster and isn't needed (decode is latency-bound).
- HuggingFace downloads (model + eval data) go to a persistent `HF_HOME`
  (`.hf_cache` in the repo) so they survive between jobs and don't fill your home
  quota.

## 1. Get the repo onto Eddie

Pick a location on group storage (not your small home dir):

```bash
EXP_DIR=/exports/chss/eddie/ppls/groups/slpgpustorage/users/$USER/tts_cw
mkdir -p "$EXP_DIR" && cd "$EXP_DIR"

# Option A — clone from your remote:
git clone <your-repo-url> TTS-TurboQuant

# Option B — copy from your laptop (run on your laptop):
#   rsync -av --exclude .git --exclude .venv ./TTS-TurboQuant/ \
#     <user>@eddie.ecdf.ed.ac.uk:$EXP_DIR/TTS-TurboQuant/

cd TTS-TurboQuant
```

## 2. One-time setup (on a login node)

Login nodes have internet; compute nodes are for the GPU jobs.

```bash
module load anaconda && conda activate py312torch27cuda118
bash scripts/eddie/setup_eddie.sh        # or: make eddie-setup
```

This installs the missing deps into `.eddie-userbase` and verifies
`import torch, turboquant`. Then **edit the `#$ -M` email line** in
`scripts/eddie/eddie_run.sh` to your `@ed.ac.uk` address.

## 3. Interactive debugging (test before you submit)

Grab a short GPU session, then run a smoke check with the same environment the
batch job uses:

```bash
qlogin -q gpu -l gpu=1 -l h_rt=0:30:00 -P ppls_ssgpu
# ... wait for allocation ...
module load anaconda && conda activate py312torch27cuda118
cd $EXP_DIR/TTS-TurboQuant

export PYTHONUSERBASE=$PWD/.eddie-userbase
export PYTHONPATH=$PWD:$PWD/models/Qwen3-TTS:$PWD/models/VALL-E-X
export HF_HOME=$PWD/.hf_cache
make smoke-qwen DEVICE=cuda

logout      # release the GPU as soon as you're done (Ctrl-D also works)
```

## 4. Submit a batch job

The job script runs whatever command follows it. Examples:

```bash
# Quick smoke (short, ~30 min slot):
qsub -N tq_smoke -l h_rt=0:30:00 scripts/eddie/eddie_run.sh make smoke-qwen DEVICE=cuda

# Full Qwen quality eval:
qsub -N qwen_eval -l h_rt=8:00:00 scripts/eddie/eddie_run.sh \
    make eval-qwen DEVICE=cuda METRICS_DEVICE=cuda

# KV attention-divergence experiment:
qsub -N tq_divergence scripts/eddie/eddie_run.sh \
    python models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
    --residual-windows 24,0 --out results/divergence_eddie.csv

# VALL-E-X eval:
qsub -N vallex_eval -l h_rt=8:00:00 scripts/eddie/eddie_run.sh make eval-vallex DEVICE=cuda
```

Or via the Makefile wrapper:

```bash
make eddie-submit CMD="make eval-qwen DEVICE=cuda METRICS_DEVICE=cuda" \
     EDDIE_NAME=qwen_eval EDDIE_RT=8:00:00
```

`qsub` command-line options (`-N`, `-l h_rt=...`, `-M`) override the `#$` defaults
baked into `eddie_run.sh`.

## 5. Monitor

```bash
qstat                 # your jobs: qw = queued/waiting, r = running
qstat -j <jobid>      # detailed status / why a job is still waiting
qdel <jobid>          # cancel a job

# Live logs (named <jobname>.o<jobid> / .e<jobid> in logs/):
tail -f logs/qwen_eval.o<jobid>
```

The job prints `nvidia-smi` and the torch/CUDA line at the top of its log, so you
can confirm it landed on a GPU and the env is correct.

## 6. Outputs and copying back

- Trial CSVs / result logs: `results/`
- Generated audio: `models/Qwen3-TTS/benchmarks/outputs/`,
  `models/VALL-E-X/benchmarks/outputs/`
- Analysis: `make eddie-submit CMD="make analyze"` (or run `make analyze` in an
  interactive session).

Copy audio off ECDF to your laptop:

```bash
rsync -av <user>@eddie.ecdf.ed.ac.uk:$EXP_DIR/TTS-TurboQuant/models/Qwen3-TTS/benchmarks/outputs/ ./outputs/
```

## Troubleshooting

- **`cuda_available False` in the log** → the job didn't get a GPU (check
  `#$ -q gpu -l gpu=1`) or the shared env's torch can't see it; re-check via an
  interactive `qlogin` GPU session.
- **A `--user` dep dragged in a different torch** → re-run that one package with
  `pip install --user --no-deps <pkg>`. Nothing is installed into the shared env,
  so this is always reversible (delete `.eddie-userbase` to start over).
- **Version friction with the shared env** (torch 2.7/cu118 vs the repo's tested
  2.6/cu124) → fallback is to clone the shared env and customize it:
  `conda create --clone py312torch27cuda118 -p $EXP_DIR/envs/tq`, activate it, then
  `pip install -e . -e models/Qwen3-TTS/ -e models/VALL-E-X/`.
- **Compute nodes turn out to be offline** → prefetch on a login node (run the
  model once so it caches into `HF_HOME`), then add `export HF_HUB_OFFLINE=1` to
  `eddie_run.sh`.

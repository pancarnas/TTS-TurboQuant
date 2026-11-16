# TTS-TurboQuant

KV-cache quantization (TurboQuant) for TTS models: Qwen3-TTS (talker, GQA,
multimodal RoPE) and VALL-E-X. Library in `turboquant/`, model patches under
`models/`, GPU scripts in `scripts/gpu/`, analysis tools in `tools/`, thesis
in `thesis/`, paper in `paper/`.

## Working rules

- No torch on the laptop: tests must be model-free CPU (`pytest tests/`), GPU
  steps go to the user as a command checklist for the remote SageMaker box.
- Do not edit `turboquant/` library code without asking first.
- Commits: no Claude co-author trailer.
- Thesis prose: no colons, plain human style, no result numbers in sections 1-3.
- Any claim about a "layer-5 hotspot" from old divergence CSVs is an artifact
  of code-predictor rows; re-derive talker-only (pos >= 33).

## Current work

Branch `exp/keyfix-group-prerope`: OmniKVQuant follow-up (group key scaling,
pre-RoPE keys). State, results, next steps and box pitfalls:
`docs/keyfix_handoff.md`. Run book: `docs/keyfix_experiment.md`.

# Evaluation datasets

Distinct sentences synthesized for inference. The same sentences are used
under every quantization config (paired design).

| Dataset | Sentences | Notes |
|---|---|---|
| seedtts_en | 100 | Seed-TTS-eval English |
| librispeech_pc | 100 | LibriSpeech-PC cross-sentence; has ground-truth audio |
| ellav_hard | 20 | tongue-twister stress set (reconstruction) |
| libritts_long | 100 | VALL-E only; 20–160 s passages |

Totals: **Qwen 220** sentences (no `libritts_long`), **VALL-E 320** sentences.

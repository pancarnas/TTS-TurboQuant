# Qwen pm4 listening study: sentence set and run design

Date 2026-11-24. Branch `exp/keyfix-group-prerope`. Short design.

## Intent

Rate the pm4 cache (4-bit min-max prompt keys over g32pre) by ear, with MOS
(naturalness) and SMOS (similarity to the cloning prompt), on sentences whose
objective scores (CER, WER, speaker similarity) exist for the same clips, and
give the objective side a larger speaker-diverse set that stands on its own.

## Goals, non-goals

- Goals: one sentence list for both tests, the listening set a strict subset
  of the objective set; speaker-diverse (39 LibriSpeech-PC voices); the
  listening draw stratified on Qwen's own fp16 WER so it spans easy to hard;
  same survey size and block design as the VALL-E study so listener load and
  the Qualtrics builder carry over.
- Non-goals: attention divergence (settled on the 50-sentence cells); seeds
  beyond 0; comparison with the VALL-E listening set on identical sentences
  (Qwen gets its own draw, decided 2026-11-24).
- Anti-goal: a listening set where the same few voices recur, which would
  confound SMOS with voice.

## Systems

Listening test, 6 per sentence, the same load as the VALL-E study: natural
LibriSpeech recording, fp16, pm4 at K3V3@0, pm4 at K3V2@0, pm4 at K2V2@0,
and g32pre at K2V2@0 (the same cache without the prompt fix: the low anchor
and attention check, in the role plain K2V2 had in the VALL-E study). The
reference prompt (a different utterance by the same speaker) is played for
SMOS only.

Objective run adds g32pp at K2V2@0 (fp16 prompt, same generated key bits) as
the reference row for the thesis table. Not in the listening test.

## Sentence set

- Objective: `mos_idx_list.txt` rebuilt on the box with
  `tools/select_mos_sentences.py --per-speaker 5 --min-dur 4 --max-dur 9`
  (195 sentences, 39 speakers). Committed to the repo this time so the set
  is reproducible without the box.
- Listening: 20 sentences, one per speaker, drawn by
  `tools/prepare_mos_study.py --with-similarity --unique-speakers
  --stratify-by wer --strata "0:0.05:8,0.05:0.15:6,0.15:0.30:4,0.30:9:2"`
  from the objective run's fp16 scores, all 6 systems present. 120 stimuli
  plus 20 references, 10 blocks x 2 sentences, as before. Strata counts are
  my call and can be rebalanced after the fp16 scores are in.

## Subjective setup (my calls, same frame as the VALL-E study)

- Stimuli: 20 sentences x 6 systems = 120 clips, plus 20 reference prompts.
  Clips RMS-normalised to one level, 24 kHz, anonymised ids (c001..c120),
  references r001..r020. Built by `tools/prepare_mos_study.py`, hosted at a
  static URL, survey imported into Qualtrics from `tools/make_qualtrics.py`.
- Blocks: 10 blocks x 2 sentences. One block per listener, assigned evenly
  by the Qualtrics randomiser, so a listener rates 12 stimuli (2 sentences x
  6 systems), about 7 minutes, identical to the VALL-E study's load. Order
  shuffled at build time, fixed per block.
- Per stimulus page: play the clip, rate naturalness on the 5-point MOS
  scale (Bad to Excellent); then, for the five synthesised systems, the
  reference prompt is revealed and the listener rates speaker similarity on
  the 5-point SMOS scale (Not at all similar to Identical speaker).
  Naturalness first, so the reference does not anchor it.
- Instructions page: headphones, quiet room, play each clip fully once
  before rating, no replay limit; a short consent line; language is English
  and the listener should be a fluent English listener.
- Attention checks: the natural clip must score 4 or 5 on MOS and the
  g32pre K2V2 clip 1 or 2 (it collapses in every sentence). Drop a listener
  who fails both, as in the VALL-E study.
- Listeners: target 15 ratings per clip, so 15 listeners per block, 150 in
  all. Prolific at about 1.5 GBP per 7-minute session is about 250 GBP;
  colleagues cover fewer blocks but cost nothing. Minimum for a thesis
  claim: 8 per block, 80 listeners.
- Analysis: MOS and SMOS mean with 95 % confidence intervals per system;
  within-listener paired comparisons (Wilcoxon) of each pm4 config against
  fp16 and against each other; per-clip MOS against CER and SMOS against
  WavLM speaker similarity from the objective run, to report how well the
  objective metrics track the ratings. Export joins back to the manifest
  through the question ids.

## Approaches considered

- A (chosen): full 39-speaker 195-sentence list, audio only, ~7 GPU h.
  Strong standalone objective set and enough candidates per WER band.
- B: 2 per speaker (78 sentences), ~2 GPU h. Thin hard band for the draw.
- C: reuse the existing first-50 run plus K3V2. 8 voices, SMOS confounded.

## How it gets made (my calls)

1. Port `--idx-file` from `models/VALL-E-X/benchmarks/benchmark_vallex_real.py`
   to `models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py` (filter
   the enumerated items by idx, resume unchanged), expose it as `IDXFILE` in
   `scripts/gpu/14_qwen_keyfix.sh`, with a model-free test.
2. Run on the box: `CELLS="pm4 g32pre g32pp" IDXFILE=mos_idx_list.txt`
   with `--no-divergence`, seed 0, into `outputs/keyfix_<cell>` (resume
   skips the first-50 wavs already there). pm4 at fp16,K2V2@0,K3V2@0,K3V3@0;
   g32pre and g32pp at fp16,K2V2@0 only. Score with `tools/score_wav_dir.py`.
   About 7 GPU h, measured on the smoke.
3. `tools/prepare_mos_study.py` learns the Qwen wav name
   (`qwen_{group}_{idx}_sampling_s0_t0.9_{cfg}.wav`, cfg from the recorder's
   `_wav_path`) and the system list above, behind a `--model qwen` switch so
   the VALL-E study stays reproducible. `tools/make_qualtrics.py` unchanged.
4. Analysis: MOS and SMOS per system with listener-paired tests, and per clip
   against CER / WER / spk_sim from the objective run.

## Left to the builder

Exact strata counts after seeing the fp16 WER distribution; whether the
natural clip is band-limited the same way as in the VALL-E study (keep the
README note); smoke with `MAXPG=2` before the full run.

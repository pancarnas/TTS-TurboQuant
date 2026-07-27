"""Decode a VALL-E-X voice preset (.npz of EnCodec tokens) to a listenable wav.

The presets store EnCodec audio_tokens, not audio. This decodes them with the
same Vocos vocoder the generation pipeline uses (decode_audio's vocos path), so
you can hear what a preset voice actually sounds like.

  python tools/decode_preset.py --preset models/VALL-E-X/presets/alan.npz \
      --out alan_preset.wav --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SAMPLE_RATE = 24000
N_Q = 8  # EnCodec 24 kHz @ 6 kbps


def _to_b_t_q(arr: np.ndarray) -> torch.Tensor:
    """Coerce stored audio_tokens to (B=1, T, n_q) for the vocos decode path."""
    a = np.squeeze(np.asarray(arr))
    if a.ndim != 2:
        raise SystemExit(f"unexpected audio_tokens shape {np.asarray(arr).shape}")
    # make it (T, n_q): the codebook axis has length N_Q
    if a.shape[0] == N_Q and a.shape[1] != N_Q:
        a = a.T
    return torch.tensor(a, dtype=torch.long).unsqueeze(0)  # (1, T, n_q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="models/VALL-E-X/presets/alan.npz")
    ap.add_argument("--out", default="alan_preset.wav")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from vocos import Vocos

    data = np.load(args.preset)
    print("keys:", list(data.keys()))
    print("audio_tokens shape:", np.asarray(data["audio_tokens"]).shape)
    codes = _to_b_t_q(data["audio_tokens"]).to(args.device)  # (1, T, n_q)

    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(args.device)
    frames = codes.permute(2, 0, 1)  # (n_q, B, T)
    features = vocos.codes_to_features(frames).float()
    samples = vocos.decode(
        features, bandwidth_id=torch.tensor([2], device=args.device)
    ).squeeze().detach().cpu().numpy()

    sf.write(args.out, samples, SAMPLE_RATE)
    print(f"wrote {args.out}  ({len(samples) / SAMPLE_RATE:.1f} s @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()

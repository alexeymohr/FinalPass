"""Local inference for the Paderborn SHEET SSL-MOS Re+CNN frame-level model.

Model: `kuhlmannm/is25-ssl-mos-cnn` (MIT), the released checkpoint of
"Towards Frame-level Quality Predictions of Synthetic Speech" (Interspeech
2025), evaluated with `fgnt/frame-level-mos` (MIT).

This module builds the architecture explicitly instead of resolving the
`factory:` paths in the model's `config.yaml`. That is deliberate: the upstream
loader (`padertorch.Module.from_storage_dir`) imports and constructs arbitrary
dotted paths named by a YAML file, and its `WAV2VEC2_BASE` encoder is
instantiated through `torchaudio.pipelines...get_model()`, which downloads
pretrained weights over Torch Hub. Neither is acceptable inside a
client-audio run.

Two audited facts make the explicit construction safe:

* the released checkpoint carries the COMPLETE fine-tuned encoder (210 tensors
  matching `torchaudio`'s Wav2Vec2 base state dict exactly), so no pretrained
  weights need fetching — the architecture is built uninitialised and every
  parameter is overwritten;
* `load_state_dict(..., strict=True)` over all 219 tensors is what proves this
  reconstruction matches the trained model, so a structural mistake fails the
  load rather than silently producing plausible numbers.

Semantics below follow the upstream sources line by line; the deviations are
listed in `PREPROCESSING_NOTES`.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn

from .config import FRAME_STRIDE, MODEL_SR, RECEPTIVE_FIELD

# `layer: -1` in config.yaml is resolved by the upstream wrapper to
# `len(transformer.layers) - 1`, and the features taken are `x[-1]` of the
# returned intermediates. The released model therefore uses the output of the
# 11th transformer layer and never runs the 12th.
ENCODER_NUM_LAYERS = 11
CONV_IN, CONV_HIDDEN, CONV_OUT, CONV_KERNEL, CONV_LAYERS = 768, 512, 512, 3, 3
LAYER_WEIGHTS = 12
TARGET_DBFS = -18.0

PREPROCESSING_NOTES = (
    "equal_loudness follows upstream exactly (paderbox int16 WAV round trip, "
    "pydub RMS gain to -18 dBFS), except that a digitally silent block is left "
    "untouched: upstream would compute an infinite gain there. Standardisation "
    "removes any constant gain, so the audible effect of the loudness stage is "
    "confined to 16-bit requantisation.",
)


def num_frames_for(num_samples: int) -> int:
    """Frames the encoder emits for a 16 kHz input of `num_samples`.

    Derived from the upstream STFT bookkeeping (window 400, shift 320,
    pad=True, fading='half'), which reduces exactly to ceil(n / 320).
    """
    if num_samples <= 0:
        return 0
    return -(-num_samples // FRAME_STRIDE)


def _encoder_padding(num_samples: int) -> tuple[int, int]:
    """Left/right zero padding the upstream wrapper applies before the encoder."""
    left = (RECEPTIVE_FIELD - FRAME_STRIDE) // 2
    right = math.ceil((RECEPTIVE_FIELD - FRAME_STRIDE) / 2)
    length = num_samples + left + right
    extra = 0
    if length < RECEPTIVE_FIELD:
        extra = RECEPTIVE_FIELD - length
    elif (length + FRAME_STRIDE - RECEPTIVE_FIELD) % FRAME_STRIDE:
        extra = FRAME_STRIDE - ((length + FRAME_STRIDE - RECEPTIVE_FIELD) % FRAME_STRIDE)
    return left, right + extra


def build_wav2vec2_base() -> nn.Module:
    """Wav2Vec2 BASE architecture with no weights and no network access.

    `torchaudio.pipelines.WAV2VEC2_BASE.get_model()` builds this same module and
    then downloads `wav2vec2_fairseq_base_ls960.pth` through Torch Hub. Only the
    builder is used here; the weights come from the Paderborn checkpoint.
    """
    from torchaudio.pipelines._wav2vec2 import utils as _w2v_utils

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    if bundle.sample_rate != MODEL_SR:
        raise RuntimeError(f"unexpected bundle sample rate {bundle.sample_rate}")
    return _w2v_utils._get_model(bundle._model_type, bundle._params)


class _Encoder(nn.Module):
    """Mirrors the `encoder.model.*` prefix of the released state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.model = build_wav2vec2_base()


class _ConvLayer(nn.Module):
    """One padertorch `Conv1d` block: pre-activation, then symmetric padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 activation: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self._activation = activation
        total = kernel_size - 1
        self._pad = (total // 2, math.ceil(total / 2))

    def forward(self, x: Tensor) -> Tensor:
        x = self._activation(x)
        x = nn.functional.pad(x, self._pad)
        return self.conv(x)


class _ConvNet(nn.Module):
    """`build_cnn1d` with this model's config: 3 layers, k=3, stride 1, no norm.

    `input_layer=True` forces the first block's pre-activation to identity and
    `output_layer=True` does the same for the last, so the stack is
    conv -> LeakyReLU -> conv -> LeakyReLU -> conv with length preserved.
    """

    def __init__(self) -> None:
        super().__init__()
        widths = [CONV_IN] + [CONV_HIDDEN] * (CONV_LAYERS - 1) + [CONV_OUT]
        activations = [nn.Identity()] + [
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
            for _ in range(CONV_LAYERS - 1)
        ]
        self.convs = nn.ModuleList([
            _ConvLayer(widths[i], widths[i + 1], CONV_KERNEL, activations[i])
            for i in range(CONV_LAYERS)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


class SSLMOS(nn.Module):
    """Frame-level MOS predictor. Scores are MOS-like: higher is better."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.conv_net = _ConvNet()
        self.out_proj = nn.Linear(CONV_OUT, 1)
        # Unused when a single encoder layer is selected, but present in the
        # trained state dict; registered so strict loading stays strict.
        self.weights = nn.Parameter(torch.ones(LAYER_WEIGHTS))

    @torch.no_grad()
    def frame_scores(self, wav: Tensor) -> Tensor:
        """Per-frame MOS for one 16 kHz mono waveform of shape (samples,)."""
        if wav.ndim != 1:
            raise ValueError(f"expected a 1-D waveform, got shape {tuple(wav.shape)}")
        n = int(wav.shape[0])
        frames = num_frames_for(n)
        if frames == 0:
            return torch.empty(0)

        left, right = _encoder_padding(n)
        padded = nn.functional.pad(wav[None, :], (left, right))
        latents, _ = self.encoder.model.feature_extractor(
            padded, torch.tensor([padded.shape[-1]], dtype=torch.long)
        )
        latents = latents[..., :frames, :]

        lengths = torch.tensor([frames], dtype=torch.long)
        x = self.encoder.model.encoder.extract_features(
            latents, lengths=lengths, num_layers=ENCODER_NUM_LAYERS
        )[-1]

        # l2_normalization: divide by the norm of the time-averaged embedding.
        mean = x.sum(dim=1, keepdim=True) / (frames + 1e-6)
        norm = torch.linalg.norm(mean, ord=2, dim=-1, keepdim=True)
        x = x / (norm + 1e-6)

        y = self.conv_net(x.mT).mT
        preds = torch.tanh(self.out_proj(y)).squeeze(-1)
        return ((preds + 1) * 2 + 1)[0]


def normalize_loudness(audio: np.ndarray, sampling_rate: int,
                       target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Upstream `equal_loudness`: int16 WAV round trip with an RMS gain.

    Reproduces `frame_level_mos.utils.normalize_loudness`, with paderbox's
    `dumps_audio` replaced by the soundfile call it makes. A digitally silent
    block is returned untouched — upstream's gain would be infinite there.
    """
    import soundfile as sf
    from pydub import AudioSegment

    dtype = audio.dtype
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, subtype="PCM_16", format="WAV")
    sound = AudioSegment(data=buf.getvalue())
    if not math.isfinite(sound.dBFS):
        return audio
    sound = sound.apply_gain(target_dbfs - sound.dBFS)
    out = io.BytesIO()
    sound.export(out, format="wav")
    out.seek(0)
    array = sf.read(out, dtype=str(dtype))[0]
    while array.ndim < audio.ndim:
        array = np.expand_dims(array, axis=0)
    return array


def prepare_waveform(audio: np.ndarray, sampling_rate: int = MODEL_SR,
                     equal_loudness: bool = True,
                     standardize: bool = True) -> np.ndarray:
    """`SSLMOS.prepare_example` for one block: equal loudness, then standardise."""
    out = audio
    if equal_loudness:
        out = normalize_loudness(out, sampling_rate)
    if standardize:
        out = (out - np.mean(out, axis=-1, keepdims=True)) / (
            np.std(out, axis=-1, keepdims=True) + 1e-7
        )
    return out


def load_model(state_path: Path, device: str = "cpu") -> SSLMOS:
    """Build the model and load the extracted inference artifact strictly.

    `weights_only=True` with no allowlist: the artifact holds nothing but
    tensors. A structural mismatch raises here rather than later.
    """
    payload = torch.load(str(state_path), map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    non_tensor = [k for k, v in state.items() if not torch.is_tensor(v)]
    if non_tensor:
        raise RuntimeError(f"non-tensor entries in model state: {non_tensor[:5]}")
    model = SSLMOS()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

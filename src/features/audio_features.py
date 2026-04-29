"""Audio feature extraction modules: log-Mel and wav2vec2."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torchaudio
from torch import nn
from transformers import AutoFeatureExtractor, Wav2Vec2Model

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureOutput:
    features: torch.Tensor  # [B, T, D]
    lengths: torch.Tensor  # [B]
    frame_hz: float


class LogMelFrontend(nn.Module):
    """Extract log-Mel frame features from raw waveforms."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    @property
    def output_dim(self) -> int:
        return self.n_mels

    @property
    def frame_hz(self) -> float:
        return float(self.sample_rate) / float(self.hop_length)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> FeatureOutput:
        feats: list[torch.Tensor] = []
        frame_lengths: list[int] = []
        for i in range(waveforms.size(0)):
            wav = waveforms[i, : int(lengths[i].item())]
            mel = self.mel(wav.unsqueeze(0))  # [1, n_mels, T]
            mel = torch.log(torch.clamp(mel, min=1e-6))
            mel = mel.transpose(1, 2).squeeze(0)  # [T, n_mels]
            feats.append(mel)
            frame_lengths.append(int(mel.size(0)))

        max_t = max(frame_lengths)
        batch = torch.zeros((len(feats), max_t, self.n_mels), device=waveforms.device)
        for i, feat in enumerate(feats):
            batch[i, : feat.size(0)] = feat.to(waveforms.device)

        return FeatureOutput(
            features=batch,
            lengths=torch.tensor(frame_lengths, dtype=torch.long, device=waveforms.device),
            frame_hz=self.frame_hz,
        )


class Wav2Vec2Frontend(nn.Module):
    """Extract frame embeddings from wav2vec2."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        sample_rate: int = 16000,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.use_safetensors = True
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        LOGGER.info(
            "Loading wav2vec2 model '%s' with use_safetensors=%s",
            model_name,
            self.use_safetensors,
        )
        try:
            self.model = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load wav2vec2 with safetensors. "
                "Install safetensors and retry: pip install safetensors"
            ) from exc
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def frame_hz(self) -> float:
        ratio = int(getattr(self.model.config, "inputs_to_logits_ratio", 320))
        return float(self.sample_rate) / float(ratio)

    def _build_input_mask(self, lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        ids = torch.arange(max_len, device=device).unsqueeze(0)
        return (ids < lengths.unsqueeze(1)).long()

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> FeatureOutput:
        device = waveforms.device
        max_len = waveforms.size(1)
        attention_mask = self._build_input_mask(lengths, max_len, device)

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            out = self.model(input_values=waveforms, attention_mask=attention_mask)
            features = out.last_hidden_state

        feat_lengths = self.model._get_feat_extract_output_lengths(lengths.cpu()).to(device)  # type: ignore[attr-defined]
        return FeatureOutput(features=features, lengths=feat_lengths, frame_hz=self.frame_hz)


def build_frontend(config: dict) -> nn.Module:
    feat_cfg = config["features"]
    feat_type = feat_cfg["type"].lower()
    if feat_type == "logmel":
        return LogMelFrontend(
            sample_rate=int(feat_cfg.get("sample_rate", 16000)),
            n_mels=int(feat_cfg.get("n_mels", 80)),
            n_fft=int(feat_cfg.get("n_fft", 400)),
            hop_length=int(feat_cfg.get("hop_length", 160)),
            win_length=int(feat_cfg.get("win_length", 400)),
        )
    if feat_type == "wav2vec2":
        return Wav2Vec2Frontend(
            model_name=str(feat_cfg.get("model_name", "facebook/wav2vec2-base")),
            sample_rate=int(feat_cfg.get("sample_rate", 16000)),
            freeze=bool(feat_cfg.get("freeze", True)),
        )
    raise ValueError(f"Unsupported feature type: {feat_type}")

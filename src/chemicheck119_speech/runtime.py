"""Thin faster-whisper adapter that preserves model-native quality signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str
    avg_log_probability: float
    no_speech_probability: float
    compression_ratio: float


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: tuple[TranscriptSegment, ...]
    audio_seconds: float
    voiced_seconds: float


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript: ...


class FasterWhisperTranscriber:
    """Pinned inference configuration for comparable CPU/GPU experiments."""

    def __init__(
        self,
        *,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        download_root: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
            local_files_only=local_files_only,
        )

    def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript:
        generated, info = self._model.transcribe(
            str(audio_path),
            language="ko",
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            word_timestamps=False,
            hotwords=hotwords,
        )
        segments = tuple(
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=segment.text,
                avg_log_probability=float(segment.avg_logprob),
                no_speech_probability=float(segment.no_speech_prob),
                compression_ratio=float(segment.compression_ratio),
            )
            for segment in generated
        )
        return Transcript(
            text=" ".join(segment.text.strip() for segment in segments).strip(),
            segments=segments,
            audio_seconds=float(info.duration),
            voiced_seconds=float(info.duration_after_vad),
        )

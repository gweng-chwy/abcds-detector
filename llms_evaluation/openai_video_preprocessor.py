"""Local video preprocessing for OpenAI evaluation."""

import hashlib
import os
from pathlib import Path
import subprocess

import models


_SUBPROCESS_TIMEOUT_SECONDS = 300


def build_cache_key(uri: str) -> str:
  """Build a stable short cache key from an input URI."""
  return hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]


class VideoPreprocessor:
  """Prepares local frame and transcript evidence for OpenAI."""

  def __init__(
      self,
      cache_dir: str,
      max_frames: int,
      frame_sample_rate: float,
      openai_service: object,
      refresh_cache: bool = False,
      transcription_model: str = "gpt-4o-transcribe",
  ):
    self.cache_dir = Path(cache_dir)
    self.max_frames = max_frames
    self.frame_sample_rate = frame_sample_rate
    self.openai_service = openai_service
    self.refresh_cache = refresh_cache
    self.transcription_model = transcription_model

  def preprocess(self, source: models.VideoSource) -> models.VideoPreprocessResult:
    """Preprocess a video source for OpenAI evaluation."""
    local_source = self._ensure_local(source)
    video_dir = self.cache_dir / build_cache_key(local_source.original_uri)
    full_frames_dir = video_dir / "frames" / "full"
    first_frames_dir = video_dir / "frames" / "first_5s"
    audio_path = video_dir / "audio.mp3"
    first_5s_audio_path = video_dir / "audio_first_5s.mp3"
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    first_frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    for frames_dir in (full_frames_dir, first_frames_dir):
      for stale_frame in frames_dir.glob("*.jpg"):
        if stale_frame.is_file():
          stale_frame.unlink()
    for stale_audio_path in (audio_path, first_5s_audio_path):
      if stale_audio_path.exists():
        stale_audio_path.unlink()

    duration_seconds = self._probe_duration(local_source.local_path)
    full_video_frame_evidence = self._extract_frames_at_timestamps(
        local_source.local_path,
        full_frames_dir,
        self._full_video_timestamps(duration_seconds),
    )
    full_video_frames = self._list_frames(full_frames_dir)
    if not full_video_frames:
      raise RuntimeError(f"No frames extracted for full video: {full_frames_dir}")
    first_5_seconds_frame_evidence = self._extract_frames_at_timestamps(
        local_source.local_path,
        first_frames_dir,
        self._first_5s_timestamps(duration_seconds),
    )
    first_5_seconds_frames = self._list_frames(first_frames_dir)
    if not first_5_seconds_frames:
      raise RuntimeError(
          f"No frames extracted for first 5 seconds: {first_frames_dir}"
      )
    try:
      self._extract_audio(local_source.local_path, audio_path)
      audio_available = audio_path.exists()
    except RuntimeError:
      for failed_audio_path in (audio_path, first_5s_audio_path):
        if failed_audio_path.exists():
          failed_audio_path.unlink()
      audio_available = False

    transcript = ""
    transcript_available = False
    result_audio_path = None
    first_5_seconds_transcript = ""
    first_5_seconds_transcript_available = False
    result_first_5s_audio_path = None
    if audio_available:
      result_audio_path = str(audio_path)
      try:
        transcript = self.openai_service.transcribe_audio(
            str(audio_path), model_name=self.transcription_model
        )
        transcript_available = bool(transcript)
      except Exception:
        transcript = ""
        transcript_available = False
      try:
        self._extract_first_5s_audio(local_source.local_path, first_5s_audio_path)
        if first_5s_audio_path.exists():
          result_first_5s_audio_path = str(first_5s_audio_path)
          try:
            first_5_seconds_transcript = self.openai_service.transcribe_audio(
                str(first_5s_audio_path), model_name=self.transcription_model
            )
            first_5_seconds_transcript_available = bool(
                first_5_seconds_transcript
            )
          except Exception:
            first_5_seconds_transcript = ""
            first_5_seconds_transcript_available = False
      except RuntimeError:
        if first_5s_audio_path.exists():
          first_5s_audio_path.unlink()
        first_5_seconds_transcript = ""
        first_5_seconds_transcript_available = False
        result_first_5s_audio_path = None

    return models.VideoPreprocessResult(
        source=local_source,
        duration_seconds=duration_seconds,
        full_video_frames=full_video_frames,
        first_5_seconds_frames=first_5_seconds_frames,
        audio_path=result_audio_path,
        transcript=transcript,
        transcript_available=transcript_available,
        full_video_frame_evidence=full_video_frame_evidence,
        first_5_seconds_frame_evidence=first_5_seconds_frame_evidence,
        full_video_transcript=transcript,
        first_5_seconds_audio_path=result_first_5s_audio_path,
        first_5_seconds_transcript=first_5_seconds_transcript,
        first_5_seconds_transcript_available=(
            first_5_seconds_transcript_available
        ),
    )

  def _ensure_local(self, source: models.VideoSource) -> models.VideoSource:
    if source.source_type == models.CreativeProviderType.YOUTUBE.value:
      video_dir = self.cache_dir / build_cache_key(source.original_uri)
      video_dir.mkdir(parents=True, exist_ok=True)
      for stale_source in video_dir.glob("source.*"):
        if stale_source.is_file():
          stale_source.unlink()
      output_template = str(video_dir / "source.%(ext)s")
      self._run([
          "yt-dlp",
          "-f",
          "mp4/best",
          "-o",
          output_template,
          source.original_uri,
      ])
      downloaded = sorted(video_dir.glob("source.*"))
      if not downloaded:
        raise FileNotFoundError(f"Downloaded video not found for {source.original_uri}")
      if len(downloaded) != 1:
        raise RuntimeError(f"Expected one downloaded video, found {len(downloaded)}")
      return models.VideoSource(
          original_uri=source.original_uri,
          local_path=str(downloaded[0]),
          source_type=source.source_type,
      )

    if not os.path.exists(source.local_path):
      raise FileNotFoundError(f"Video file not found: {source.local_path}")
    return source

  def _probe_duration(self, video_path: str) -> float:
    completed = self._run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ])
    return float(completed.stdout.strip())

  def _full_video_timestamps(self, duration_seconds: float) -> list[float]:
    """Return evenly spaced frame timestamps across the full video."""
    if duration_seconds <= 0 or self.max_frames <= 1:
      return [0.0]
    step = duration_seconds / (self.max_frames - 1)
    return [round(step * index, 2) for index in range(self.max_frames)]

  def _first_5s_timestamps(self, duration_seconds: float) -> list[float]:
    """Return sampled frame timestamps within the first five seconds."""
    if duration_seconds <= 0 or self.max_frames <= 1:
      return [0.0]
    end_seconds = min(duration_seconds, 5.0)
    if self.frame_sample_rate <= 0:
      return [0.0, round(end_seconds, 2)][:self.max_frames]

    step = 1 / self.frame_sample_rate
    timestamps = []
    timestamp = 0.0
    while timestamp < end_seconds:
      timestamps.append(round(timestamp, 2))
      timestamp += step
    if not timestamps or timestamps[-1] != round(end_seconds, 2):
      timestamps.append(round(end_seconds, 2))
    if len(timestamps) <= self.max_frames:
      return timestamps
    return timestamps[: self.max_frames - 1] + [round(end_seconds, 2)]

  def _extract_frames_at_timestamps(
      self,
      video_path: str,
      output_dir: Path,
      timestamps: list[float],
  ) -> list[models.VideoFrameEvidence]:
    evidence = []
    for index, timestamp in enumerate(timestamps, start=1):
      output_path = output_dir / f"frame_{index:04d}.jpg"
      try:
        self._run([
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            str(output_path),
        ])
      except RuntimeError:
        if timestamp <= 0:
          raise
      if not output_path.exists() and timestamp > 0:
        retry_timestamp = max(0.0, timestamp - 0.05)
        self._run([
            "ffmpeg",
            "-y",
            "-ss",
            f"{retry_timestamp:.2f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            str(output_path),
        ])
      if output_path.exists():
        evidence.append(models.VideoFrameEvidence(
            path=str(output_path),
            timestamp_seconds=timestamp,
        ))
    return evidence

  def _extract_first_5s_audio(self, video_path: str, audio_path: Path) -> None:
    self._run([
        "ffmpeg",
        "-y",
        "-t",
        "5",
        "-i",
        video_path,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(audio_path),
    ])

  def _extract_audio(self, video_path: str, audio_path: Path) -> None:
    self._run([
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(audio_path),
    ])

  def _list_frames(self, frames_dir: Path) -> list[str]:
    return [str(path) for path in sorted(frames_dir.glob("*.jpg"))]

  def _run(self, cmd: list[str]):
    try:
      return subprocess.run(
          cmd,
          check=True,
          capture_output=True,
          text=True,
          timeout=_SUBPROCESS_TIMEOUT_SECONDS,
      )
    except subprocess.TimeoutExpired as exc:
      raise RuntimeError(
          "Command timed out after "
          f"{_SUBPROCESS_TIMEOUT_SECONDS}s: {' '.join(cmd)}; "
          f"stderr: {exc.stderr or ''}"
      ) from exc
    except subprocess.CalledProcessError as exc:
      raise RuntimeError(
          f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}; "
          f"stderr: {exc.stderr or ''}"
      ) from exc

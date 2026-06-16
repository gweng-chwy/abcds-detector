"""Local video preprocessing for OpenAI evaluation."""

import hashlib
import os
from pathlib import Path
import subprocess

import models


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
  ):
    self.cache_dir = Path(cache_dir)
    self.max_frames = max_frames
    self.frame_sample_rate = frame_sample_rate
    self.openai_service = openai_service

  def preprocess(self, source: models.VideoSource) -> models.VideoPreprocessResult:
    """Preprocess a video source for OpenAI evaluation."""
    local_source = self._ensure_local(source)
    video_dir = self.cache_dir / build_cache_key(local_source.original_uri)
    full_frames_dir = video_dir / "frames" / "full"
    first_frames_dir = video_dir / "frames" / "first_5s"
    audio_path = video_dir / "audio.mp4"
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    first_frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    duration_seconds = self._probe_duration(local_source.local_path)
    self._extract_frames(local_source.local_path, full_frames_dir)
    self._extract_first_5s_frames(local_source.local_path, first_frames_dir)
    self._extract_audio(local_source.local_path, audio_path)

    transcript = ""
    transcript_available = False
    result_audio_path = None
    if audio_path.exists():
      result_audio_path = str(audio_path)
      try:
        transcript = self.openai_service.transcribe_audio(str(audio_path))
        transcript_available = bool(transcript)
      except Exception:
        transcript = ""
        transcript_available = False

    return models.VideoPreprocessResult(
        source=local_source,
        duration_seconds=duration_seconds,
        full_video_frames=self._list_frames(full_frames_dir),
        first_5_seconds_frames=self._list_frames(first_frames_dir),
        audio_path=result_audio_path,
        transcript=transcript,
        transcript_available=transcript_available,
    )

  def _ensure_local(self, source: models.VideoSource) -> models.VideoSource:
    if source.source_type == models.CreativeProviderType.YOUTUBE.value:
      video_dir = self.cache_dir / build_cache_key(source.original_uri)
      video_dir.mkdir(parents=True, exist_ok=True)
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

  def _extract_frames(self, video_path: str, output_dir: Path) -> None:
    self._run([
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={self.frame_sample_rate}",
        "-frames:v",
        str(self.max_frames),
        str(output_dir / "frame_%04d.jpg"),
    ])

  def _extract_first_5s_frames(self, video_path: str, output_dir: Path) -> None:
    self._run([
        "ffmpeg",
        "-y",
        "-t",
        "5",
        "-i",
        video_path,
        "-vf",
        f"fps={self.frame_sample_rate}",
        "-frames:v",
        str(self.max_frames),
        str(output_dir / "frame_%04d.jpg"),
    ])

  def _extract_audio(self, video_path: str, audio_path: Path) -> None:
    self._run([
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-c:a",
        "copy",
        str(audio_path),
    ])

  def _list_frames(self, frames_dir: Path) -> list[str]:
    return [str(path) for path in sorted(frames_dir.glob("*.jpg"))]

  def _run(self, cmd: list[str]):
    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

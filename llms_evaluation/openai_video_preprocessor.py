"""Local video preprocessing for OpenAI evaluation."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import models


_SUBPROCESS_TIMEOUT_SECONDS = 300
_MANIFEST_SCHEMA_VERSION = 1
_EXTRACTION_STRATEGY_VERSION = "openai-evidence-v2"
_MANIFEST_FILENAME = "preprocess_manifest.json"


def build_cache_key(uri: str) -> str:
  """Build a stable short cache key from an input URI."""
  return hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]


def _manifest_path(video_dir: Path) -> Path:
  return video_dir / _MANIFEST_FILENAME


def _source_fingerprint(source: models.VideoSource) -> dict:
  source_stat = Path(source.local_path).stat()
  return {
      "original_uri": source.original_uri,
      "local_path": source.local_path,
      "source_type": source.source_type,
      "size": source_stat.st_size,
      "mtime_ns": source_stat.st_mtime_ns,
  }


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
    manifest_path = _manifest_path(video_dir)
    if not self.refresh_cache:
      cached_result = self._load_manifest_result(manifest_path, local_source)
      if cached_result is not None:
        return cached_result

    full_frames_dir = video_dir / "frames" / "full"
    first_frames_dir = video_dir / "frames" / "first_5s"
    audio_path = video_dir / "audio.mp3"
    first_5s_audio_path = video_dir / "audio_first_5s.mp3"
    transcript_path = video_dir / "transcript.txt"
    first_5s_transcript_path = video_dir / "transcript_first_5s.txt"
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
    transcript_error = False
    first_5_seconds_transcript = ""
    first_5_seconds_transcript_available = False
    result_first_5s_audio_path = None
    first_5_seconds_transcript_error = False
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
        transcript_error = True
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
            first_5_seconds_transcript_error = True
      except RuntimeError:
        if first_5s_audio_path.exists():
          first_5s_audio_path.unlink()
        first_5_seconds_transcript = ""
        first_5_seconds_transcript_available = False
        result_first_5s_audio_path = None

    transcript_path.write_text(transcript, encoding="utf-8")
    first_5s_transcript_path.write_text(
        first_5_seconds_transcript, encoding="utf-8"
    )

    result = models.VideoPreprocessResult(
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
        preprocess_manifest_path=str(manifest_path),
    )
    self._write_manifest(
        manifest_path,
        result,
        transcript_path,
        first_5s_transcript_path,
        transcript_error=transcript_error,
        first_5_seconds_transcript_error=first_5_seconds_transcript_error,
    )
    return result

  def _ensure_local(self, source: models.VideoSource) -> models.VideoSource:
    if source.source_type == models.CreativeProviderType.YOUTUBE.value:
      video_dir = self.cache_dir / build_cache_key(source.original_uri)
      video_dir.mkdir(parents=True, exist_ok=True)
      cached_sources = sorted(video_dir.glob("source.*"))
      if not self.refresh_cache and cached_sources:
        if len(cached_sources) != 1:
          raise RuntimeError(
              f"Expected one downloaded video, found {len(cached_sources)}"
          )
        return models.VideoSource(
            original_uri=source.original_uri,
            local_path=str(cached_sources[0]),
            source_type=source.source_type,
        )
      if self.refresh_cache:
        for stale_source in cached_sources:
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

  def _settings_fingerprint(self) -> dict:
    return {
        "max_frames": self.max_frames,
        "frame_sample_rate": self.frame_sample_rate,
        "transcription_model": self.transcription_model,
    }

  def _load_manifest_result(
      self,
      manifest_path: Path,
      source: models.VideoSource,
  ) -> models.VideoPreprocessResult | None:
    if not manifest_path.exists():
      return None
    try:
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      source_fingerprint = _source_fingerprint(source)
    except (OSError, json.JSONDecodeError):
      return None
    if not isinstance(manifest, dict):
      return None
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
      return None
    if manifest.get("strategy_version") != _EXTRACTION_STRATEGY_VERSION:
      return None
    if manifest.get("source") != source_fingerprint:
      return None
    if manifest.get("settings") != self._settings_fingerprint():
      return None
    if manifest.get("transcript_error"):
      return None
    if manifest.get("first_5_seconds_transcript_error"):
      return None

    full_video_frame_evidence = self._manifest_frame_evidence(
        manifest.get("full_video_frame_evidence")
    )
    first_5_seconds_frame_evidence = self._manifest_frame_evidence(
        manifest.get("first_5_seconds_frame_evidence")
    )
    if not full_video_frame_evidence or not first_5_seconds_frame_evidence:
      return None

    audio_path = manifest.get("audio_path")
    first_5_seconds_audio_path = manifest.get("first_5_seconds_audio_path")
    if not self._manifest_optional_artifact_exists(audio_path):
      return None
    if not self._manifest_optional_artifact_exists(first_5_seconds_audio_path):
      return None

    transcript_path = manifest.get("transcript_path")
    first_5_seconds_transcript_path = manifest.get(
        "first_5_seconds_transcript_path"
    )
    if not isinstance(transcript_path, str):
      return None
    if not isinstance(first_5_seconds_transcript_path, str):
      return None
    try:
      duration_seconds = float(manifest.get("duration_seconds"))
      transcript = Path(transcript_path).read_text(encoding="utf-8")
      first_5_seconds_transcript = Path(
          first_5_seconds_transcript_path
      ).read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError):
      return None
    transcript_available = bool(manifest.get("transcript_available"))
    first_5_seconds_transcript_available = bool(
        manifest.get("first_5_seconds_transcript_available")
    )
    if transcript_available != bool(transcript):
      return None
    if first_5_seconds_transcript_available != bool(
        first_5_seconds_transcript
    ):
      return None

    return models.VideoPreprocessResult(
        source=source,
        duration_seconds=duration_seconds,
        full_video_frames=[
            frame.path for frame in full_video_frame_evidence
        ],
        first_5_seconds_frames=[
            frame.path for frame in first_5_seconds_frame_evidence
        ],
        audio_path=audio_path,
        transcript=transcript,
        transcript_available=transcript_available,
        full_video_frame_evidence=full_video_frame_evidence,
        first_5_seconds_frame_evidence=first_5_seconds_frame_evidence,
        full_video_transcript=transcript,
        first_5_seconds_audio_path=first_5_seconds_audio_path,
        first_5_seconds_transcript=first_5_seconds_transcript,
        first_5_seconds_transcript_available=(
            first_5_seconds_transcript_available
        ),
        preprocess_manifest_path=str(manifest_path),
    )

  def _write_manifest(
      self,
      manifest_path: Path,
      result: models.VideoPreprocessResult,
      transcript_path: Path,
      first_transcript_path: Path,
      transcript_error: bool = False,
      first_5_seconds_transcript_error: bool = False,
  ) -> None:
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "strategy_version": _EXTRACTION_STRATEGY_VERSION,
        "source": _source_fingerprint(result.source),
        "settings": self._settings_fingerprint(),
        "duration_seconds": result.duration_seconds,
        "full_video_frame_evidence": [
            {
                "path": frame.path,
                "timestamp_seconds": frame.timestamp_seconds,
            }
            for frame in result.full_video_frame_evidence
        ],
        "first_5_seconds_frame_evidence": [
            {
                "path": frame.path,
                "timestamp_seconds": frame.timestamp_seconds,
            }
            for frame in result.first_5_seconds_frame_evidence
        ],
        "audio_path": result.audio_path,
        "first_5_seconds_audio_path": result.first_5_seconds_audio_path,
        "transcript_path": str(transcript_path),
        "first_5_seconds_transcript_path": str(first_transcript_path),
        "transcript_available": result.transcript_available,
        "first_5_seconds_transcript_available": (
            result.first_5_seconds_transcript_available
        ),
        "transcript_error": transcript_error,
        "first_5_seconds_transcript_error": (
            first_5_seconds_transcript_error
        ),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  def _manifest_frame_evidence(
      self,
      manifest_evidence,
  ) -> list[models.VideoFrameEvidence] | None:
    if not isinstance(manifest_evidence, list):
      return None
    evidence = []
    for frame in manifest_evidence:
      if not isinstance(frame, dict):
        return None
      if set(frame) != {"path", "timestamp_seconds"}:
        return None
      path = frame["path"]
      if not isinstance(path, str) or not Path(path).is_file():
        return None
      try:
        timestamp_seconds = float(frame["timestamp_seconds"])
      except (TypeError, ValueError):
        return None
      evidence.append(models.VideoFrameEvidence(
          path=path,
          timestamp_seconds=timestamp_seconds,
      ))
    return evidence

  def _manifest_optional_artifact_exists(self, artifact_path) -> bool:
    if artifact_path is None:
      return True
    if not isinstance(artifact_path, str) or not artifact_path:
      return False
    return Path(artifact_path).is_file()

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
    """Return sampled frame timestamps across the full video."""
    return self._sampled_timestamps(duration_seconds)

  def _first_5s_timestamps(self, duration_seconds: float) -> list[float]:
    """Return sampled frame timestamps within the first five seconds."""
    if self.max_frames <= 1:
      return [0.0]
    end_seconds = min(duration_seconds, 5.0)
    return self._sampled_timestamps(end_seconds, max_count=self.max_frames)

  def _sampled_timestamps(
      self,
      end_seconds: float,
      max_count: int | None = None,
  ) -> list[float]:
    """Return sample-rate timestamps from zero through end_seconds."""
    if end_seconds <= 0:
      return [0.0]
    if self.frame_sample_rate <= 0:
      timestamps = [0.0, round(end_seconds, 2)]
    else:
      step = 1 / self.frame_sample_rate
      timestamps = []
      timestamp = 0.0
      while timestamp < end_seconds:
        timestamps.append(round(timestamp, 2))
        timestamp += step
      if not timestamps or timestamps[-1] != round(end_seconds, 2):
        timestamps.append(round(end_seconds, 2))

    if max_count is None or len(timestamps) <= max_count:
      return timestamps
    return timestamps[: max_count - 1] + [round(end_seconds, 2)]

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

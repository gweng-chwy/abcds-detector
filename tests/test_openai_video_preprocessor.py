"""Tests for OpenAI video preprocessing."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import models
import llms_evaluation.openai_video_preprocessor as preprocessor_module
from llms_evaluation.openai_video_preprocessor import (
    VideoPreprocessor,
    build_cache_key,
)


def test_build_cache_key_returns_stable_sha256_prefix():
  """Cache keys are stable 16-character sha256 prefixes."""
  uri = "https://www.youtube.com/watch?v=test"

  assert build_cache_key(uri) == hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
  assert len(build_cache_key(uri)) == 16


def test_local_source_requires_existing_path(tmp_path):
  """Missing local videos fail before subprocess work starts."""
  source = models.VideoSource(
      original_uri="missing.mp4",
      local_path=str(tmp_path / "missing.mp4"),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=3,
      frame_sample_rate=1.0,
      openai_service=object(),
  )

  with pytest.raises(FileNotFoundError, match="Video file not found"):
    preprocessor.preprocess(source)


def test_local_preprocessing_extracts_media_and_returns_transcript(
    tmp_path, monkeypatch
):
  """Local preprocessing returns duration, frames, audio path, and transcript."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path):
      assert audio_path == str(
          cache_dir / build_cache_key(str(video_path)) / "audio.mp3"
      )
      return "hello there"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="12.5\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      output_pattern = Path(cmd[-1])
      (output_pattern.parent / "frame_0001.jpg").write_bytes(b"frame")
      (output_pattern.parent / "frame_0002.jpg").write_bytes(b"frame")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=4,
      frame_sample_rate=0.5,
      openai_service=FakeOpenAIService(),
  ).preprocess(source)

  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  assert commands == [
      [
          "ffprobe",
          "-v",
          "error",
          "-show_entries",
          "format=duration",
          "-of",
          "default=noprint_wrappers=1:nokey=1",
          str(video_path),
      ],
      [
          "ffmpeg",
          "-y",
          "-i",
          str(video_path),
          "-vf",
          "fps=0.5",
          "-frames:v",
          "4",
          str(video_cache_dir / "frames" / "full" / "frame_%04d.jpg"),
      ],
      [
          "ffmpeg",
          "-y",
          "-t",
          "5",
          "-i",
          str(video_path),
          "-vf",
          "fps=0.5",
          "-frames:v",
          "4",
          str(video_cache_dir / "frames" / "first_5s" / "frame_%04d.jpg"),
      ],
      [
          "ffmpeg",
          "-y",
          "-i",
          str(video_path),
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
          str(video_cache_dir / "audio.mp3"),
      ],
  ]
  assert result == models.VideoPreprocessResult(
      source=source,
      duration_seconds=12.5,
      full_video_frames=[
          str(video_cache_dir / "frames" / "full" / "frame_0001.jpg"),
          str(video_cache_dir / "frames" / "full" / "frame_0002.jpg"),
      ],
      first_5_seconds_frames=[
          str(video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg"),
          str(video_cache_dir / "frames" / "first_5s" / "frame_0002.jpg"),
      ],
      audio_path=str(video_cache_dir / "audio.mp3"),
      transcript="hello there",
      transcript_available=True,
  )


def test_youtube_preprocessing_downloads_before_extracting(tmp_path, monkeypatch):
  """YouTube preprocessing downloads into cache and returns the cached source."""
  cache_dir = tmp_path / "cache"
  youtube_uri = "https://www.youtube.com/watch?v=test"
  source = models.VideoSource(
      original_uri=youtube_uri,
      local_path="",
      source_type=models.CreativeProviderType.YOUTUBE.value,
  )
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path):
      return "downloaded transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "yt-dlp":
      output_template = Path(cmd[4])
      (output_template.parent / "source.mp4").write_bytes(b"downloaded")
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="8.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      output_pattern = Path(cmd[-1])
      (output_pattern.parent / "frame_0001.jpg").write_bytes(b"frame")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=2,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
  ).preprocess(source)

  video_cache_dir = cache_dir / build_cache_key(youtube_uri)
  downloaded_path = video_cache_dir / "source.mp4"
  assert commands[0] == [
      "yt-dlp",
      "-f",
      "mp4/best",
      "-o",
      str(video_cache_dir / "source.%(ext)s"),
      youtube_uri,
  ]
  assert commands[1][0] == "ffprobe"
  assert commands[1][-1] == str(downloaded_path)
  assert result.source == models.VideoSource(
      original_uri=youtube_uri,
      local_path=str(downloaded_path),
      source_type=models.CreativeProviderType.YOUTUBE.value,
  )
  assert result.duration_seconds == 8.0
  assert result.transcript == "downloaded transcript"


def test_youtube_preprocessing_ignores_stale_hash_scoped_sources(
    tmp_path, monkeypatch
):
  """Stale source files in the video cache are removed before download."""
  cache_dir = tmp_path / "cache"
  youtube_uri = "https://www.youtube.com/watch?v=stale"
  video_cache_dir = cache_dir / build_cache_key(youtube_uri)
  video_cache_dir.mkdir(parents=True)
  stale_path = video_cache_dir / "source.avi"
  stale_path.write_bytes(b"old")
  unrelated_path = cache_dir / "source.avi"
  unrelated_path.write_bytes(b"outside hash dir")
  source = models.VideoSource(
      original_uri=youtube_uri,
      local_path="",
      source_type=models.CreativeProviderType.YOUTUBE.value,
  )

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path):
      return "fresh transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "yt-dlp":
      assert not stale_path.exists()
      output_template = Path(cmd[4])
      (output_template.parent / "source.mp4").write_bytes(b"fresh")
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="9.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      output_pattern = Path(cmd[-1])
      (output_pattern.parent / "frame_0001.jpg").write_bytes(b"frame")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
  ).preprocess(source)

  assert result.source.local_path == str(video_cache_dir / "source.mp4")
  assert not stale_path.exists()
  assert unrelated_path.exists()


def test_transcription_failure_does_not_fail_preprocessing(tmp_path, monkeypatch):
  """Audio transcription failures produce a result without transcript evidence."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )

  class FailingOpenAIService:

    def transcribe_audio(self, audio_path):
      raise RuntimeError("transcription failed")

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      output_pattern = Path(cmd[-1])
      (output_pattern.parent / "frame_0001.jpg").write_bytes(b"frame")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FailingOpenAIService(),
  ).preprocess(source)

  assert result.audio_path == str(
      cache_dir / build_cache_key(str(video_path)) / "audio.mp3"
  )
  assert result.transcript == ""
  assert result.transcript_available is False


def test_audio_extraction_failure_does_not_fail_preprocessing(tmp_path, monkeypatch):
  """Videos without extractable audio still return frame evidence."""
  video_path = tmp_path / "silent.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  stale_audio_path = cache_dir / build_cache_key(str(video_path)) / "audio.mp3"
  stale_audio_path.parent.mkdir(parents=True)
  stale_audio_path.write_bytes(b"stale audio")
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )

  class UnexpectedOpenAIService:

    def transcribe_audio(self, audio_path):
      raise AssertionError("transcription should not run without audio")

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      output_pattern = Path(cmd[-1])
      (output_pattern.parent / "frame_0001.jpg").write_bytes(b"frame")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      raise preprocessor_module.subprocess.CalledProcessError(
          returncode=1,
          cmd=cmd,
          stderr="Stream map '0:a:0' matches no streams.",
      )
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=UnexpectedOpenAIService(),
  ).preprocess(source)

  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  assert result.full_video_frames == [
      str(video_cache_dir / "frames" / "full" / "frame_0001.jpg")
  ]
  assert result.first_5_seconds_frames == [
      str(video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg")
  ]
  assert result.audio_path is None
  assert result.transcript == ""
  assert result.transcript_available is False


def test_frame_extraction_without_jpegs_fails_preprocessing(tmp_path, monkeypatch):
  """Frame extraction must produce JPEG evidence."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  stale_full_frame = video_cache_dir / "frames" / "full" / "frame_0001.jpg"
  stale_first_frame = video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg"
  stale_full_frame.parent.mkdir(parents=True)
  stale_first_frame.parent.mkdir(parents=True)
  stale_full_frame.write_bytes(b"stale")
  stale_first_frame.write_bytes(b"stale")
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="4.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  with pytest.raises(RuntimeError, match="No frames extracted.*full"):
    VideoPreprocessor(
        cache_dir=str(cache_dir),
        max_frames=1,
        frame_sample_rate=1.0,
        openai_service=object(),
    ).preprocess(source)


def test_subprocess_failure_includes_command_and_stderr(tmp_path, monkeypatch):
  """Subprocess errors include enough context to diagnose the failure."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    raise preprocessor_module.subprocess.CalledProcessError(
        returncode=2,
        cmd=cmd,
        stderr="probe failed",
    )

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  with pytest.raises(RuntimeError, match="ffprobe.*probe failed"):
    VideoPreprocessor(
        cache_dir=str(tmp_path / "cache"),
        max_frames=1,
        frame_sample_rate=1.0,
        openai_service=object(),
    ).preprocess(source)

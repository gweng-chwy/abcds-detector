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


def _write_exact_frame_output(cmd):
  output_path = Path(cmd[-1])
  assert "%" not in output_path.name
  output_path.write_bytes(b"frame")


def test_build_cache_key_returns_stable_sha256_prefix():
  """Cache keys are stable 16-character sha256 prefixes."""
  uri = "https://www.youtube.com/watch?v=test"

  assert build_cache_key(uri) == hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
  assert len(build_cache_key(uri)) == 16


def test_full_video_frame_timestamps_are_uniform_across_duration(tmp_path):
  """Full-video frame timestamps are evenly distributed across the video."""
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=4,
      frame_sample_rate=1.0,
      openai_service=object(),
  )

  assert preprocessor._full_video_timestamps(10.0) == [0.0, 3.33, 6.67, 10.0]


def test_first_5s_frame_timestamps_are_capped_by_duration(tmp_path):
  """First-5-second frame timestamps stop at the shorter source duration."""
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=24,
      frame_sample_rate=2.0,
      openai_service=object(),
  )

  assert preprocessor._first_5s_timestamps(3.0) == [
      0.0,
      0.5,
      1.0,
      1.5,
      2.0,
      2.5,
      3.0,
  ]


def test_frame_timestamps_fall_back_to_zero_for_empty_or_single_frame(tmp_path):
  """Timestamp helpers always return zero for empty videos or single-frame caps."""
  zero_duration_preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=4,
      frame_sample_rate=2.0,
      openai_service=object(),
  )
  single_frame_preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=1,
      frame_sample_rate=2.0,
      openai_service=object(),
  )

  assert zero_duration_preprocessor._full_video_timestamps(0.0) == [0.0]
  assert zero_duration_preprocessor._first_5s_timestamps(-1.0) == [0.0]
  assert single_frame_preprocessor._full_video_timestamps(10.0) == [0.0]
  assert single_frame_preprocessor._first_5s_timestamps(10.0) == [0.0]


def test_frame_extraction_retries_slightly_before_missing_endpoint(
    tmp_path, monkeypatch
):
  """Endpoint frame extraction retries earlier while preserving evidence time."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  output_dir = tmp_path / "frames"
  output_dir.mkdir()
  commands = []

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[cmd.index("-ss") + 1] == "9.95":
      Path(cmd[-1]).write_bytes(b"frame")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  evidence = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=object(),
  )._extract_frames_at_timestamps(str(video_path), output_dir, [10.0])

  assert [cmd[cmd.index("-ss") + 1] for cmd in commands] == ["10.00", "9.95"]
  assert evidence == [
      models.VideoFrameEvidence(
          path=str(output_dir / "frame_0001.jpg"),
          timestamp_seconds=10.0,
      )
  ]


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


def test_local_preprocessing_extracts_first_5s_audio_and_transcript(
    tmp_path, monkeypatch
):
  """Local preprocessing returns timestamped frame and transcript evidence."""
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

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      assert model_name == "gpt-4o-transcribe"
      if audio_path.endswith("audio_first_5s.mp3"):
        return "first five words"
      if audio_path.endswith("audio.mp3"):
        return "full transcript"
      raise AssertionError(f"unexpected audio path: {audio_path}")

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="12.5\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
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
  ffprobe_commands = [cmd for cmd in commands if cmd[0] == "ffprobe"]
  frame_commands = [
      cmd for cmd in commands if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg")
  ]
  audio_commands = [
      cmd for cmd in commands if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3")
  ]
  assert ffprobe_commands == [[
      "ffprobe",
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      str(video_path),
  ]]
  assert len(frame_commands) == 8
  assert all("-ss" in cmd for cmd in frame_commands)
  assert all(cmd[-1].endswith(".jpg") for cmd in frame_commands)
  assert audio_commands == [
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
      [
          "ffmpeg",
          "-y",
          "-t",
          "5",
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
          str(video_cache_dir / "audio_first_5s.mp3"),
      ],
  ]
  assert result.full_video_frames == [
      str(video_cache_dir / "frames" / "full" / "frame_0001.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0002.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0003.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0004.jpg"),
  ]
  assert result.first_5_seconds_frames == [
      str(video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg"),
      str(video_cache_dir / "frames" / "first_5s" / "frame_0002.jpg"),
      str(video_cache_dir / "frames" / "first_5s" / "frame_0003.jpg"),
      str(video_cache_dir / "frames" / "first_5s" / "frame_0004.jpg"),
  ]
  assert result.audio_path == str(video_cache_dir / "audio.mp3")
  assert result.transcript == "full transcript"
  assert result.full_video_transcript == "full transcript"
  assert result.transcript_available is True
  assert result.first_5_seconds_audio_path == str(
      video_cache_dir / "audio_first_5s.mp3"
  )
  assert result.first_5_seconds_transcript == "first five words"
  assert result.first_5_seconds_transcript_available is True
  assert [
      frame.timestamp_seconds for frame in result.full_video_frame_evidence
  ] == [0.0, 4.17, 8.33, 12.5]
  assert [
      frame.timestamp_seconds for frame in result.first_5_seconds_frame_evidence
  ] == [0.0, 2.0, 4.0, 5.0]


def test_local_preprocessing_uses_configured_transcription_model(
    tmp_path, monkeypatch
):
  """Transcription uses the model configured by the CLI path."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  transcription_calls = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name):
      transcription_calls.append((audio_path, model_name))
      return "configured transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="12.5\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=4,
      frame_sample_rate=0.5,
      openai_service=FakeOpenAIService(),
      transcription_model="gpt-custom-transcribe",
  ).preprocess(source)

  assert transcription_calls == [
      (
          str(cache_dir / build_cache_key(str(video_path)) / "audio.mp3"),
          "gpt-custom-transcribe",
      ),
      (
          str(
              cache_dir
              / build_cache_key(str(video_path))
              / "audio_first_5s.mp3"
          ),
          "gpt-custom-transcribe",
      ),
  ]
  assert result.transcript == "configured transcript"


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

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      assert model_name == "gpt-4o-transcribe"
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
      _write_exact_frame_output(cmd)
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

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      assert model_name == "gpt-4o-transcribe"
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
      _write_exact_frame_output(cmd)
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

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      raise RuntimeError("transcription failed")

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
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


def test_first_5s_transcription_failure_keeps_extracted_audio_path(
    tmp_path, monkeypatch
):
  """A first-5 transcription failure keeps the extracted first-5 audio."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )

  class First5TranscriptionFailingService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        raise RuntimeError("first-5 transcription failed")
      return "full transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=First5TranscriptionFailingService(),
  ).preprocess(source)

  first_5s_audio_path = cache_dir / build_cache_key(
      str(video_path)
  ) / "audio_first_5s.mp3"
  assert result.audio_path == str(
      cache_dir / build_cache_key(str(video_path)) / "audio.mp3"
  )
  assert result.transcript == "full transcript"
  assert result.transcript_available is True
  assert result.first_5_seconds_audio_path == str(first_5s_audio_path)
  assert first_5s_audio_path.exists()
  assert result.first_5_seconds_transcript == ""
  assert result.first_5_seconds_transcript_available is False


def test_empty_first_5s_transcript_keeps_extracted_audio_path(
    tmp_path, monkeypatch
):
  """An empty first-5 transcript still returns the extracted audio path."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )

  class EmptyFirst5TranscriptService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return ""
      return "full transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=EmptyFirst5TranscriptService(),
  ).preprocess(source)

  first_5s_audio_path = cache_dir / build_cache_key(
      str(video_path)
  ) / "audio_first_5s.mp3"
  assert result.first_5_seconds_audio_path == str(first_5s_audio_path)
  assert first_5s_audio_path.exists()
  assert result.first_5_seconds_transcript == ""
  assert result.first_5_seconds_transcript_available is False


def test_first_5s_audio_extraction_failure_removes_partial_audio(
    tmp_path, monkeypatch
):
  """A failed first-5 extraction removes partial audio and omits its path."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  transcription_calls = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      transcription_calls.append(audio_path)
      return "full transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith("audio_first_5s.mp3"):
      Path(cmd[-1]).write_bytes(b"partial audio")
      raise preprocessor_module.subprocess.CalledProcessError(
          returncode=1,
          cmd=cmd,
          stderr="first-5 extraction failed",
      )
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

  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  assert result.audio_path == str(video_cache_dir / "audio.mp3")
  assert result.transcript == "full transcript"
  assert result.transcript_available is True
  assert result.first_5_seconds_audio_path is None
  assert not (video_cache_dir / "audio_first_5s.mp3").exists()
  assert result.first_5_seconds_transcript == ""
  assert result.first_5_seconds_transcript_available is False
  assert transcription_calls == [str(video_cache_dir / "audio.mp3")]


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

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      raise AssertionError("transcription should not run without audio")

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
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

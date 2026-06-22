"""Tests for OpenAI video preprocessing."""

import hashlib
import json
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


def _write_manifest(
    cache_dir,
    source,
    max_frames,
    frame_sample_rate,
    transcription_model="gpt-4o-transcribe",
    duration_seconds=12.5,
    transcript="cached full transcript",
    first_5_seconds_transcript="cached first transcript",
):
  video_cache_dir = cache_dir / build_cache_key(source.original_uri)
  full_frame_path = video_cache_dir / "frames" / "full" / "frame_0001.jpg"
  first_frame_path = (
      video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg"
  )
  audio_path = video_cache_dir / "audio.mp3"
  first_audio_path = video_cache_dir / "audio_first_5s.mp3"
  transcript_path = video_cache_dir / "transcript.txt"
  first_transcript_path = video_cache_dir / "transcript_first_5s.txt"
  for artifact_path in (
      full_frame_path,
      first_frame_path,
      audio_path,
      first_audio_path,
      transcript_path,
      first_transcript_path,
  ):
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
  full_frame_path.write_bytes(b"cached full frame")
  first_frame_path.write_bytes(b"cached first frame")
  audio_path.write_bytes(b"cached audio")
  first_audio_path.write_bytes(b"cached first audio")
  transcript_path.write_text(transcript, encoding="utf-8")
  first_transcript_path.write_text(
      first_5_seconds_transcript, encoding="utf-8"
  )
  source_path = Path(source.local_path)
  manifest = {
      "schema_version": 1,
      "strategy_version": preprocessor_module._EXTRACTION_STRATEGY_VERSION,
      "source": {
          "original_uri": source.original_uri,
          "local_path": source.local_path,
          "source_type": source.source_type,
          "size": source_path.stat().st_size,
          "mtime_ns": source_path.stat().st_mtime_ns,
      },
      "settings": {
          "max_frames": max_frames,
          "frame_sample_rate": frame_sample_rate,
          "transcription_model": transcription_model,
      },
      "duration_seconds": duration_seconds,
      "full_video_frame_evidence": [
          {
              "path": str(full_frame_path),
              "timestamp_seconds": 0.0,
          }
      ],
      "first_5_seconds_frame_evidence": [
          {
              "path": str(first_frame_path),
              "timestamp_seconds": 0.0,
          }
      ],
      "audio_path": str(audio_path),
      "first_5_seconds_audio_path": str(first_audio_path),
      "transcript_path": str(transcript_path),
      "first_5_seconds_transcript_path": str(first_transcript_path),
      "transcript_available": bool(transcript),
      "first_5_seconds_transcript_available": bool(
          first_5_seconds_transcript
      ),
  }
  manifest_path = video_cache_dir / "preprocess_manifest.json"
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  return manifest_path


def test_build_cache_key_returns_stable_sha256_prefix():
  """Cache keys are stable 16-character sha256 prefixes."""
  uri = "https://www.youtube.com/watch?v=test"

  assert build_cache_key(uri) == hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
  assert len(build_cache_key(uri)) == 16


def test_full_video_frame_timestamps_cover_duration_at_sample_rate(tmp_path):
  """Full-video frame timestamps sample the whole video at the configured rate."""
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=4,
      frame_sample_rate=1.0,
      openai_service=object(),
  )

  assert preprocessor._full_video_timestamps(10.0) == [
      0.0,
      1.0,
      2.0,
      3.0,
      4.0,
      5.0,
      6.0,
      7.0,
      8.0,
      9.0,
      10.0,
  ]


def test_full_video_frame_timestamps_are_not_capped_by_max_frames(tmp_path):
  """The first-5-second max frame cap does not truncate full-video evidence."""
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=2,
      frame_sample_rate=0.5,
      openai_service=object(),
  )

  assert preprocessor._full_video_timestamps(5.0) == [0.0, 2.0, 4.0, 5.0]


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


def test_frame_timestamps_fall_back_to_zero_for_empty_videos(tmp_path):
  """Timestamp helpers always return zero for empty videos."""
  zero_duration_preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=4,
      frame_sample_rate=2.0,
      openai_service=object(),
  )

  assert zero_duration_preprocessor._full_video_timestamps(0.0) == [0.0]
  assert zero_duration_preprocessor._first_5s_timestamps(-1.0) == [0.0]


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
    seek_time = cmd[cmd.index("-ss") + 1]
    if seek_time == "10.00":
      raise preprocessor_module.subprocess.CalledProcessError(
          returncode=1,
          cmd=cmd,
          stderr="Output file is empty",
      )
    if seek_time == "9.95":
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


def test_frame_extraction_failure_at_zero_timestamp_still_raises(
    tmp_path, monkeypatch
):
  """Zero timestamp extraction failures are not treated as EOF retries."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  output_dir = tmp_path / "frames"
  output_dir.mkdir()

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    raise preprocessor_module.subprocess.CalledProcessError(
        returncode=1,
        cmd=cmd,
        stderr="decode failed",
    )

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  with pytest.raises(RuntimeError, match="decode failed"):
    VideoPreprocessor(
        cache_dir=str(tmp_path / "cache"),
        max_frames=1,
        frame_sample_rate=1.0,
        openai_service=object(),
    )._extract_frames_at_timestamps(str(video_path), output_dir, [0.0])


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


def test_preprocessing_reuses_valid_manifest_cache(tmp_path, monkeypatch):
  """Valid manifest cache avoids subprocess and transcription work."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  manifest_path = _write_manifest(
      cache_dir,
      source,
      max_frames=4,
      frame_sample_rate=0.5,
      transcript="cached full",
      first_5_seconds_transcript="cached first",
  )

  class UnexpectedOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      raise AssertionError("transcription should not run on cache hit")

  def fake_run(cmd, check, capture_output, text, timeout):
    raise AssertionError(f"subprocess should not run on cache hit: {cmd}")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=4,
      frame_sample_rate=0.5,
      openai_service=UnexpectedOpenAIService(),
  ).preprocess(source)

  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  assert result.source == source
  assert result.duration_seconds == 12.5
  assert result.full_video_frames == [
      str(video_cache_dir / "frames" / "full" / "frame_0001.jpg")
  ]
  assert result.first_5_seconds_frames == [
      str(video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg")
  ]
  assert result.audio_path == str(video_cache_dir / "audio.mp3")
  assert result.first_5_seconds_audio_path == str(
      video_cache_dir / "audio_first_5s.mp3"
  )
  assert result.transcript == "cached full"
  assert result.full_video_transcript == "cached full"
  assert result.first_5_seconds_transcript == "cached first"
  assert result.transcript_available is True
  assert result.first_5_seconds_transcript_available is True
  assert result.preprocess_manifest_path == str(manifest_path)


def test_refresh_cache_rebuilds_valid_manifest_cache(tmp_path, monkeypatch):
  """Refresh ignores valid manifest cache and writes fresh outputs."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  manifest_path = _write_manifest(
      cache_dir,
      source,
      max_frames=1,
      frame_sample_rate=1.0,
      transcript="cached full",
      first_5_seconds_transcript="cached first",
  )
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return "fresh first"
      return "fresh full"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"fresh audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
      refresh_cache=True,
  ).preprocess(source)

  assert [cmd[0] for cmd in commands].count("ffprobe") == 1
  assert result.duration_seconds == 3.0
  assert result.transcript == "fresh full"
  assert result.first_5_seconds_transcript == "fresh first"
  assert result.preprocess_manifest_path == str(manifest_path)
  assert (cache_dir / build_cache_key(str(video_path)) / "transcript.txt").read_text(
      encoding="utf-8"
  ) == "fresh full"


def test_manifest_setting_mismatch_rebuilds_cache(tmp_path, monkeypatch):
  """Changed preprocessing settings invalidate manifest cache."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  _write_manifest(
      cache_dir,
      source,
      max_frames=1,
      frame_sample_rate=1.0,
      transcript="cached full",
      first_5_seconds_transcript="cached first",
  )
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return "rebuilt first"
      return "rebuilt full"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="4.0\n")
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

  assert [cmd[0] for cmd in commands].count("ffprobe") == 1
  assert result.transcript == "rebuilt full"
  assert len(result.full_video_frame_evidence) == 5


def test_missing_manifest_transcript_rebuilds_cache(tmp_path, monkeypatch):
  """Missing referenced transcript artifact invalidates manifest cache."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  manifest_path = _write_manifest(
      cache_dir,
      source,
      max_frames=1,
      frame_sample_rate=1.0,
      transcript="cached full",
      first_5_seconds_transcript="cached first",
  )
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  Path(manifest["transcript_path"]).unlink()
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return "rebuilt first"
      return "rebuilt full"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"fresh audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
  ).preprocess(source)

  assert [cmd[0] for cmd in commands].count("ffprobe") == 1
  assert result.transcript == "rebuilt full"
  assert Path(manifest["transcript_path"]).read_text(
      encoding="utf-8"
  ) == "rebuilt full"


def test_manifest_transcript_availability_mismatch_rebuilds_cache(
    tmp_path, monkeypatch
):
  """Transcript availability flags must match cached transcript file content."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  manifest_path = _write_manifest(
      cache_dir,
      source,
      max_frames=1,
      frame_sample_rate=1.0,
      transcript="cached full",
      first_5_seconds_transcript="cached first",
  )
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  Path(manifest["transcript_path"]).write_text("", encoding="utf-8")
  manifest["transcript_available"] = True
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return "rebuilt first"
      return "rebuilt full"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"fresh audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
  ).preprocess(source)

  assert [cmd[0] for cmd in commands].count("ffprobe") == 1
  assert result.transcript == "rebuilt full"
  assert result.transcript_available is True


def test_preprocessing_reuses_no_audio_manifest_cache(tmp_path, monkeypatch):
  """Manifest cache without audio artifacts loads empty transcripts as unavailable."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  manifest_path = _write_manifest(
      cache_dir,
      source,
      max_frames=1,
      frame_sample_rate=1.0,
      transcript="",
      first_5_seconds_transcript="",
  )
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  Path(manifest["audio_path"]).unlink()
  Path(manifest["first_5_seconds_audio_path"]).unlink()
  manifest["audio_path"] = None
  manifest["first_5_seconds_audio_path"] = None
  manifest["transcript_available"] = False
  manifest["first_5_seconds_transcript_available"] = False
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  class UnexpectedOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      raise AssertionError("transcription should not run on cache hit")

  def fake_run(cmd, check, capture_output, text, timeout):
    raise AssertionError(f"subprocess should not run on cache hit: {cmd}")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=UnexpectedOpenAIService(),
  ).preprocess(source)

  assert result.audio_path is None
  assert result.first_5_seconds_audio_path is None
  assert result.transcript == ""
  assert result.full_video_transcript == ""
  assert result.first_5_seconds_transcript == ""
  assert result.transcript_available is False
  assert result.first_5_seconds_transcript_available is False
  assert result.preprocess_manifest_path == str(manifest_path)


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
  assert len(frame_commands) == 12
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
      str(video_cache_dir / "frames" / "full" / "frame_0005.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0006.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0007.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0008.jpg"),
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
  ] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.5]
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


def test_youtube_preprocessing_reuses_cached_source_without_refresh(
    tmp_path, monkeypatch
):
  """YouTube preprocessing reuses an existing cached source when not refreshing."""
  cache_dir = tmp_path / "cache"
  youtube_uri = "https://www.youtube.com/watch?v=cached"
  video_cache_dir = cache_dir / build_cache_key(youtube_uri)
  video_cache_dir.mkdir(parents=True)
  cached_source_path = video_cache_dir / "source.mp4"
  cached_source_path.write_bytes(b"cached source")
  source = models.VideoSource(
      original_uri=youtube_uri,
      local_path="",
      source_type=models.CreativeProviderType.YOUTUBE.value,
  )
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      assert model_name == "gpt-4o-transcribe"
      return "cached source transcript"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    assert cmd[0] != "yt-dlp"
    if cmd[0] == "ffprobe":
      assert cmd[-1] == str(cached_source_path)
      return SimpleNamespace(stdout="8.0\n")
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

  assert [cmd[0] for cmd in commands] == ["ffprobe"] + ["ffmpeg"] * 12
  assert result.source.local_path == str(cached_source_path)
  assert result.transcript == "cached source transcript"


def test_youtube_preprocessing_ignores_stale_hash_scoped_sources(
    tmp_path, monkeypatch
):
  """Refresh removes stale source files in the video cache before download."""
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
      refresh_cache=True,
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


def test_transcription_failure_manifest_is_not_reused_on_next_run(
    tmp_path, monkeypatch
):
  """Transient full transcription failures are retried instead of cached."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  commands = []

  class FailingOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      raise RuntimeError("transcription failed")

  class RetryingOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return "retry first"
      return "retry full"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  first_result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FailingOpenAIService(),
  ).preprocess(source)
  manifest_path = Path(first_result.preprocess_manifest_path)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

  second_result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=RetryingOpenAIService(),
  ).preprocess(source)

  assert first_result.audio_path == str(
      cache_dir / build_cache_key(str(video_path)) / "audio.mp3"
  )
  assert first_result.transcript == ""
  assert first_result.transcript_available is False
  assert manifest["transcript_error"] is True
  assert [cmd[0] for cmd in commands].count("ffprobe") == 2
  assert second_result.transcript == "retry full"
  assert second_result.transcript_available is True
  assert second_result.first_5_seconds_transcript == "retry first"
  assert second_result.first_5_seconds_transcript_available is True


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


def test_first_5s_transcription_failure_manifest_is_not_reused_on_next_run(
    tmp_path, monkeypatch
):
  """Transient first-5 transcription failures are retried instead of cached."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(
      original_uri=str(video_path),
      local_path=str(video_path),
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  commands = []

  class First5TranscriptionFailingService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        raise RuntimeError("first-5 transcription failed")
      return "full transcript"

  class RetryingOpenAIService:

    def transcribe_audio(self, audio_path, model_name="gpt-4o-transcribe"):
      if audio_path.endswith("audio_first_5s.mp3"):
        return "retry first"
      return "retry full"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="3.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      _write_exact_frame_output(cmd)
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp3"):
      Path(cmd[-1]).write_bytes(b"audio")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  first_result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=First5TranscriptionFailingService(),
  ).preprocess(source)
  manifest_path = Path(first_result.preprocess_manifest_path)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

  second_result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=RetryingOpenAIService(),
  ).preprocess(source)

  assert first_result.transcript == "full transcript"
  assert first_result.first_5_seconds_audio_path == str(
      cache_dir / build_cache_key(str(video_path)) / "audio_first_5s.mp3"
  )
  assert first_result.first_5_seconds_transcript == ""
  assert manifest["first_5_seconds_transcript_error"] is True
  assert [cmd[0] for cmd in commands].count("ffprobe") == 2
  assert second_result.transcript == "retry full"
  assert second_result.first_5_seconds_transcript == "retry first"
  assert second_result.first_5_seconds_transcript_available is True


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
      str(video_cache_dir / "frames" / "full" / "frame_0001.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0002.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0003.jpg"),
      str(video_cache_dir / "frames" / "full" / "frame_0004.jpg"),
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

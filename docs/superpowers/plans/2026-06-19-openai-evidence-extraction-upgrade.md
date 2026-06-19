# OpenAI Evidence Extraction Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the OpenAI path so local/YouTube videos reuse cached evidence, sample frames across the full runtime, route first-5 feature groups to first-5 evidence, and produce a post-implementation evidence review notebook.

**Architecture:** Add explicit evidence metadata to `models.VideoPreprocessResult`, then make `llms_evaluation/openai_video_preprocessor.py` produce manifest-backed full and first-5 evidence packs. `OpenAIDetector` selects the right evidence pack, `OpenAIAPIService` includes frame timestamp labels in the prompt, and a notebook helper renders 16:9 frame/transcript/checklist review figures after a live validation run.

**Tech Stack:** Python 3.11, pytest, ffmpeg/ffprobe, OpenAI Python SDK, yt-dlp, matplotlib, notebook JSON.

---

## File Structure

- Modify `models.py`: add `VideoFrameEvidence`; extend `VideoPreprocessResult` with frame metadata, first-5 audio/transcript fields, and manifest path.
- Modify `configuration.py`: add `refresh_cache` config field and `set_parameters` arg.
- Modify `utils.py`: add `--refresh_cache` CLI flag and config mapping.
- Modify `main.py`: pass `refresh_cache` into `VideoPreprocessor`.
- Modify `llms_evaluation/openai_video_preprocessor.py`: manifest cache, uniform full-video frame extraction, first-5 audio/transcript, cache reuse.
- Modify `llms_evaluation/openai_detector.py`: select frame paths, frame metadata, and transcript by feature group.
- Modify `llms_evaluation/openai_api_service.py`: include frame evidence text and selected transcript in Responses API input.
- Modify `requirements.txt`: add notebook visualization dependency.
- Create `notebooks/20260619_openai_evidence_review/README.md`: notebook usage and output contract.
- Create `notebooks/20260619_openai_evidence_review/evidence_review.py`: sample discovery, optional live-run command builder, transcript snippet extraction, 16:9 matplotlib rendering.
- Create `notebooks/20260619_openai_evidence_review/openai_evidence_review.ipynb`: notebook that calls helper functions.
- Test files: update existing OpenAI tests plus create `tests/test_openai_evidence_review.py`.

Do not edit `docs/feature_inventory/*`. Do not edit `features_repository/*` unless a later explicit user instruction unblocks that.

## Task 1: Models And CLI Cache Flag

**Files:**
- Modify: `models.py`
- Modify: `configuration.py`
- Modify: `utils.py`
- Modify: `main.py`
- Test: `tests/test_openai_cli_config.py`

- [ ] **Step 1: Write failing model and CLI tests**

Append/modify these tests in `tests/test_openai_cli_config.py`:

```python
def test_openai_cli_refresh_cache_flag_defaults_false():
  """Refresh cache is opt-in so repeated OpenAI runs can reuse artifacts."""
  args = utils.parse_args([])

  config = utils.build_abcd_params_config(args)

  assert config.refresh_cache is False


def test_openai_cli_refresh_cache_flag_sets_config():
  """Refresh cache can be requested from the CLI."""
  args = utils.parse_args(["--refresh_cache"])

  config = utils.build_abcd_params_config(args)

  assert config.refresh_cache is True


def test_video_preprocess_result_tracks_timestamped_evidence():
  """Video preprocessing results can carry frame metadata and first-5 transcript."""
  source = models.VideoSource(
      original_uri="ad.mp4",
      local_path="ad.mp4",
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  full_frame = models.VideoFrameEvidence(
      path="frames/full/frame_0001.jpg",
      timestamp_seconds=0.0,
      segment=models.VideoSegment.FULL_VIDEO.value,
  )
  first_frame = models.VideoFrameEvidence(
      path="frames/first_5s/frame_0001.jpg",
      timestamp_seconds=0.0,
      segment=models.VideoSegment.FIRST_5_SECS_VIDEO.value,
  )

  result = models.VideoPreprocessResult(
      source=source,
      duration_seconds=12.5,
      full_video_frames=[full_frame.path],
      first_5_seconds_frames=[first_frame.path],
      audio_path="audio.mp3",
      transcript="full transcript",
      transcript_available=True,
      full_video_frame_evidence=[full_frame],
      first_5_seconds_frame_evidence=[first_frame],
      first_5_seconds_audio_path="audio_first_5s.mp3",
      first_5_seconds_transcript="first five transcript",
      first_5_seconds_transcript_available=True,
      preprocess_manifest_path=".cache/key/preprocess_manifest.json",
  )

  assert result.full_video_frame_evidence == [full_frame]
  assert result.first_5_seconds_frame_evidence == [first_frame]
  assert result.first_5_seconds_transcript == "first five transcript"
  assert result.preprocess_manifest_path == ".cache/key/preprocess_manifest.json"
```

Also update `test_openai_local_youtube_cli_config` to include and assert:

```python
      "--refresh_cache",
```

```python
  assert config.refresh_cache is True
```

Update `test_default_cli_config_uses_gemini_gcs_and_safe_optional_values`:

```python
  assert config.refresh_cache is False
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
pytest tests/test_openai_cli_config.py -q
```

Expected: fails because `models.VideoFrameEvidence` and `config.refresh_cache` do not exist, and argparse does not know `--refresh_cache`.

- [ ] **Step 3: Implement model and config fields**

In `models.py`, add after `VideoSource`:

```python
@dataclass
class VideoFrameEvidence:
  """Timestamped frame evidence used by OpenAI evaluation."""

  path: str
  timestamp_seconds: float
  segment: str
```

Extend `VideoPreprocessResult`:

```python
@dataclass
class VideoPreprocessResult:
  """Class that represents video preprocessing outputs"""

  source: VideoSource
  duration_seconds: float
  full_video_frames: list[str]
  first_5_seconds_frames: list[str]
  audio_path: str | None
  transcript: str
  transcript_available: bool
  full_video_frame_evidence: list[VideoFrameEvidence] = field(
      default_factory=list
  )
  first_5_seconds_frame_evidence: list[VideoFrameEvidence] = field(
      default_factory=list
  )
  first_5_seconds_audio_path: str | None = None
  first_5_seconds_transcript: str = ""
  first_5_seconds_transcript_available: bool = False
  preprocess_manifest_path: str | None = None
```

In `configuration.py`, add default:

```python
    self.refresh_cache = False
```

Add `refresh_cache` to `set_parameters`:

```python
      refresh_cache: bool | None = None,
```

Set it after `frame_sample_rate`:

```python
    self.refresh_cache = bool(refresh_cache)
```

In `utils.py`, pass it through:

```python
      refresh_cache=getattr(args, "refresh_cache", None),
```

Add parser flag after `--frame_sample_rate`:

```python
  parser.add_argument(
      "--refresh_cache",
      help="Rebuild OpenAI local preprocessing cache for this run",
      action="store_true",
      default=False,
  )
```

In `main.py`, pass to preprocessor:

```python
      refresh_cache=config.refresh_cache,
```

- [ ] **Step 4: Run tests and verify green**

Run:

```bash
pytest tests/test_openai_cli_config.py -q
```

Expected: all tests in file pass.

- [ ] **Step 5: Commit**

```bash
git add models.py configuration.py utils.py main.py tests/test_openai_cli_config.py
git commit -m "feat: add OpenAI cache refresh config"
```

## Task 2: Timestamped Frame And First-5 Audio Extraction

**Files:**
- Modify: `llms_evaluation/openai_video_preprocessor.py`
- Test: `tests/test_openai_video_preprocessor.py`

- [ ] **Step 1: Write failing extraction tests**

Append these tests to `tests/test_openai_video_preprocessor.py`:

```python
def test_full_video_frame_timestamps_are_uniform_across_duration(tmp_path):
  """Full-video frame timestamps cover the full video duration."""
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=4,
      frame_sample_rate=1.0,
      openai_service=object(),
  )

  assert preprocessor._full_video_timestamps(10.0) == [0.0, 3.33, 6.67, 10.0]


def test_first_5s_frame_timestamps_are_capped_by_duration(tmp_path):
  """First-5 evidence never asks for frames beyond available duration."""
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=24,
      frame_sample_rate=2.0,
      openai_service=object(),
  )

  assert preprocessor._first_5s_timestamps(3.0) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_local_preprocessing_extracts_first_5s_audio_and_transcript(
    tmp_path, monkeypatch
):
  """Preprocessing creates separate full and first-5 transcript evidence."""
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
      transcription_calls.append(Path(audio_path).name)
      if Path(audio_path).name == "audio_first_5s.mp3":
        return "first five words"
      return "full words after five seconds"

  def fake_run(cmd, check, capture_output, text, timeout):
    assert check is True
    assert capture_output is True
    assert text is True
    assert timeout == 300
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="10.0\n")
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".jpg"):
      Path(cmd[-1]).write_bytes(b"frame")
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

  assert transcription_calls == ["audio.mp3", "audio_first_5s.mp3"]
  assert result.transcript == "full words after five seconds"
  assert result.first_5_seconds_transcript == "first five words"
  assert result.first_5_seconds_transcript_available is True
  assert result.first_5_seconds_audio_path == str(
      cache_dir / build_cache_key(str(video_path)) / "audio_first_5s.mp3"
  )
  assert [frame.timestamp_seconds for frame in result.full_video_frame_evidence] == [
      0.0,
      10.0,
  ]
  assert [frame.timestamp_seconds for frame in result.first_5_seconds_frame_evidence] == [
      0.0,
      1.0,
      2.0,
      3.0,
      4.0,
      5.0,
  ]
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
pytest tests/test_openai_video_preprocessor.py::test_full_video_frame_timestamps_are_uniform_across_duration tests/test_openai_video_preprocessor.py::test_first_5s_frame_timestamps_are_capped_by_duration tests/test_openai_video_preprocessor.py::test_local_preprocessing_extracts_first_5s_audio_and_transcript -q
```

Expected: fails because timestamp helpers and first-5 transcript fields are not implemented.

- [ ] **Step 3: Implement timestamp helpers and extraction**

In `llms_evaluation/openai_video_preprocessor.py`, add imports:

```python
import json
```

Add constructor arg and field:

```python
      refresh_cache: bool = False,
```

```python
    self.refresh_cache = refresh_cache
```

Replace `_extract_frames` and `_extract_first_5s_frames` with timestamp-based extraction:

```python
  def _full_video_timestamps(self, duration_seconds: float) -> list[float]:
    if self.max_frames <= 1:
      return [0.0]
    if duration_seconds <= 0:
      return [0.0]
    step = duration_seconds / (self.max_frames - 1)
    return [round(step * index, 2) for index in range(self.max_frames)]

  def _first_5s_timestamps(self, duration_seconds: float) -> list[float]:
    segment_duration = min(max(duration_seconds, 0.0), 5.0)
    if segment_duration == 0:
      return [0.0]
    frame_count = min(
        self.max_frames,
        int(segment_duration * self.frame_sample_rate) + 1,
    )
    if frame_count <= 1:
      return [0.0]
    step = segment_duration / (frame_count - 1)
    return [round(step * index, 2) for index in range(frame_count)]

  def _extract_frames_at_timestamps(
      self,
      video_path: str,
      output_dir: Path,
      timestamps: list[float],
      segment: str,
  ) -> list[models.VideoFrameEvidence]:
    frame_evidence = []
    for index, timestamp in enumerate(timestamps, start=1):
      frame_path = output_dir / f"frame_{index:04d}.jpg"
      self._run([
          "ffmpeg",
          "-y",
          "-ss",
          f"{timestamp:.2f}",
          "-i",
          video_path,
          "-frames:v",
          "1",
          str(frame_path),
      ])
      frame_evidence.append(
          models.VideoFrameEvidence(
              path=str(frame_path),
              timestamp_seconds=timestamp,
              segment=segment,
          )
      )
    return frame_evidence
```

Change `preprocess` to use:

```python
    full_frame_evidence = self._extract_frames_at_timestamps(
        local_source.local_path,
        full_frames_dir,
        self._full_video_timestamps(duration_seconds),
        models.VideoSegment.FULL_VIDEO.value,
    )
    full_video_frames = self._list_frames(full_frames_dir)
```

```python
    first_5_seconds_frame_evidence = self._extract_frames_at_timestamps(
        local_source.local_path,
        first_frames_dir,
        self._first_5s_timestamps(duration_seconds),
        models.VideoSegment.FIRST_5_SECS_VIDEO.value,
    )
    first_5_seconds_frames = self._list_frames(first_frames_dir)
```

Add first-5 audio:

```python
    first_5_audio_path = video_dir / "audio_first_5s.mp3"
```

Delete stale first-5 audio when rebuilding:

```python
    if first_5_audio_path.exists():
      first_5_audio_path.unlink()
```

Add helper:

```python
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
```

Transcribe both audio files:

```python
    first_5_transcript = ""
    first_5_transcript_available = False
    result_first_5_audio_path = None
    if audio_available:
      try:
        self._extract_first_5s_audio(local_source.local_path, first_5_audio_path)
        result_first_5_audio_path = str(first_5_audio_path)
        first_5_transcript = self.openai_service.transcribe_audio(
            str(first_5_audio_path), model_name=self.transcription_model
        )
        first_5_transcript_available = bool(first_5_transcript)
      except Exception:
        if first_5_audio_path.exists():
          first_5_audio_path.unlink()
        result_first_5_audio_path = None
        first_5_transcript = ""
        first_5_transcript_available = False
```

Return new fields:

```python
        full_video_frame_evidence=full_frame_evidence,
        first_5_seconds_frame_evidence=first_5_seconds_frame_evidence,
        first_5_seconds_audio_path=result_first_5_audio_path,
        first_5_seconds_transcript=first_5_transcript,
        first_5_seconds_transcript_available=first_5_transcript_available,
```

- [ ] **Step 4: Update existing tests for new ffmpeg frame command shape**

Current tests assert the exact old `ffmpeg -vf fps=... -frames:v ...` command. Replace those exact command expectations with command-family assertions:

```python
  ffmpeg_frame_commands = [
      command
      for command in commands
      if command[0] == "ffmpeg" and command[-1].endswith(".jpg")
  ]
  assert len(ffmpeg_frame_commands) == 4
  assert all("-ss" in command for command in ffmpeg_frame_commands)
```

Keep assertions for `ffprobe`, audio extraction, transcript content, and result dataclass equality. Add the new dataclass fields to expected results where equality is used.

- [ ] **Step 5: Run tests and verify green**

Run:

```bash
pytest tests/test_openai_video_preprocessor.py -q
```

Expected: all preprocessor tests pass.

- [ ] **Step 6: Commit**

```bash
git add llms_evaluation/openai_video_preprocessor.py tests/test_openai_video_preprocessor.py
git commit -m "feat: extract timestamped OpenAI evidence"
```

## Task 3: Manifest Cache Reuse And Invalidation

**Files:**
- Modify: `llms_evaluation/openai_video_preprocessor.py`
- Test: `tests/test_openai_video_preprocessor.py`

- [ ] **Step 1: Write failing manifest cache tests**

Append these tests:

```python
def _write_manifest(video_cache_dir, source, video_path, duration=10.0):
  full_frame = video_cache_dir / "frames" / "full" / "frame_0001.jpg"
  first_frame = video_cache_dir / "frames" / "first_5s" / "frame_0001.jpg"
  audio = video_cache_dir / "audio.mp3"
  first_audio = video_cache_dir / "audio_first_5s.mp3"
  transcript = video_cache_dir / "transcript.txt"
  first_transcript = video_cache_dir / "transcript_first_5s.txt"
  for path in (full_frame, first_frame, audio, first_audio, transcript, first_transcript):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"text" if path.suffix == ".txt" else b"asset")
  transcript.write_text("cached full", encoding="utf-8")
  first_transcript.write_text("cached first", encoding="utf-8")
  stat = video_path.stat()
  manifest = {
      "schema_version": 1,
      "strategy_version": "openai-evidence-v1",
      "source": {
          "original_uri": source.original_uri,
          "local_path": source.local_path,
          "source_type": source.source_type,
          "size": stat.st_size,
          "mtime_ns": stat.st_mtime_ns,
      },
      "settings": {
          "max_frames": 1,
          "frame_sample_rate": 1.0,
          "transcription_model": "gpt-4o-transcribe",
      },
      "duration_seconds": duration,
      "full_video_frame_evidence": [{
          "path": str(full_frame),
          "timestamp_seconds": 0.0,
          "segment": models.VideoSegment.FULL_VIDEO.value,
      }],
      "first_5_seconds_frame_evidence": [{
          "path": str(first_frame),
          "timestamp_seconds": 0.0,
          "segment": models.VideoSegment.FIRST_5_SECS_VIDEO.value,
      }],
      "audio_path": str(audio),
      "first_5_seconds_audio_path": str(first_audio),
      "transcript_path": str(transcript),
      "first_5_seconds_transcript_path": str(first_transcript),
      "transcript_available": True,
      "first_5_seconds_transcript_available": True,
  }
  (video_cache_dir / "preprocess_manifest.json").write_text(
      json.dumps(manifest),
      encoding="utf-8",
  )


def test_preprocessing_reuses_valid_manifest_cache(tmp_path, monkeypatch):
  """Valid manifest cache avoids subprocess and transcription work."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(str(video_path), str(video_path), "LOCAL")
  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  _write_manifest(video_cache_dir, source, video_path)

  def fail_run(*args, **kwargs):
    raise AssertionError("subprocess should not run on cache hit")

  class UnexpectedOpenAIService:

    def transcribe_audio(self, audio_path, model_name):
      raise AssertionError("transcription should not run on cache hit")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fail_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=UnexpectedOpenAIService(),
  ).preprocess(source)

  assert result.transcript == "cached full"
  assert result.first_5_seconds_transcript == "cached first"
  assert result.preprocess_manifest_path == str(video_cache_dir / "preprocess_manifest.json")


def test_refresh_cache_rebuilds_valid_manifest_cache(tmp_path, monkeypatch):
  """refresh_cache ignores an otherwise valid manifest."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(str(video_path), str(video_path), "LOCAL")
  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  _write_manifest(video_cache_dir, source, video_path)
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name):
      return "fresh"

  def fake_run(cmd, check, capture_output, text, timeout):
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="10.0\n")
    if cmd[0] == "ffmpeg":
      Path(cmd[-1]).write_bytes(b"fresh")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=1,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
      refresh_cache=True,
  ).preprocess(source)

  assert commands
  assert result.transcript == "fresh"


def test_manifest_setting_mismatch_rebuilds_cache(tmp_path, monkeypatch):
  """Changing frame settings invalidates the manifest."""
  video_path = tmp_path / "video.mp4"
  video_path.write_bytes(b"video")
  cache_dir = tmp_path / "cache"
  source = models.VideoSource(str(video_path), str(video_path), "LOCAL")
  video_cache_dir = cache_dir / build_cache_key(str(video_path))
  _write_manifest(video_cache_dir, source, video_path)
  commands = []

  class FakeOpenAIService:

    def transcribe_audio(self, audio_path, model_name):
      return "rebuilt"

  def fake_run(cmd, check, capture_output, text, timeout):
    commands.append(cmd)
    if cmd[0] == "ffprobe":
      return SimpleNamespace(stdout="10.0\n")
    if cmd[0] == "ffmpeg":
      Path(cmd[-1]).write_bytes(b"fresh")
    return SimpleNamespace(stdout="")

  monkeypatch.setattr(preprocessor_module.subprocess, "run", fake_run)

  result = VideoPreprocessor(
      cache_dir=str(cache_dir),
      max_frames=2,
      frame_sample_rate=1.0,
      openai_service=FakeOpenAIService(),
  ).preprocess(source)

  assert commands
  assert result.transcript == "rebuilt"
```

Add `import json` at top of test file.

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
pytest tests/test_openai_video_preprocessor.py::test_preprocessing_reuses_valid_manifest_cache tests/test_openai_video_preprocessor.py::test_refresh_cache_rebuilds_valid_manifest_cache tests/test_openai_video_preprocessor.py::test_manifest_setting_mismatch_rebuilds_cache -q
```

Expected: fails because manifest load/save is missing.

- [ ] **Step 3: Implement manifest helpers**

In `llms_evaluation/openai_video_preprocessor.py`, add constants:

```python
_MANIFEST_SCHEMA_VERSION = 1
_EXTRACTION_STRATEGY_VERSION = "openai-evidence-v1"
_MANIFEST_FILENAME = "preprocess_manifest.json"
```

Add helper methods:

```python
  def _manifest_path(self, video_dir: Path) -> Path:
    return video_dir / _MANIFEST_FILENAME

  def _source_fingerprint(self, source: models.VideoSource) -> dict:
    path = Path(source.local_path)
    stat = path.stat()
    return {
        "original_uri": source.original_uri,
        "local_path": source.local_path,
        "source_type": source.source_type,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

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
    try:
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      return None

    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
      return None
    if manifest.get("strategy_version") != _EXTRACTION_STRATEGY_VERSION:
      return None
    if manifest.get("source") != self._source_fingerprint(source):
      return None
    if manifest.get("settings") != self._settings_fingerprint():
      return None

    full_evidence = [
        models.VideoFrameEvidence(**item)
        for item in manifest.get("full_video_frame_evidence", [])
    ]
    first_evidence = [
        models.VideoFrameEvidence(**item)
        for item in manifest.get("first_5_seconds_frame_evidence", [])
    ]
    artifact_paths = [
        *(frame.path for frame in full_evidence),
        *(frame.path for frame in first_evidence),
        manifest.get("audio_path"),
        manifest.get("first_5_seconds_audio_path"),
        manifest.get("transcript_path"),
        manifest.get("first_5_seconds_transcript_path"),
    ]
    if any(path and not Path(path).exists() for path in artifact_paths):
      return None
    if not full_evidence or not first_evidence:
      return None

    transcript_path = manifest.get("transcript_path")
    first_transcript_path = manifest.get("first_5_seconds_transcript_path")
    return models.VideoPreprocessResult(
        source=source,
        duration_seconds=float(manifest["duration_seconds"]),
        full_video_frames=[frame.path for frame in full_evidence],
        first_5_seconds_frames=[frame.path for frame in first_evidence],
        audio_path=manifest.get("audio_path"),
        transcript=Path(transcript_path).read_text(encoding="utf-8")
        if transcript_path and Path(transcript_path).exists()
        else "",
        transcript_available=bool(manifest.get("transcript_available")),
        full_video_frame_evidence=full_evidence,
        first_5_seconds_frame_evidence=first_evidence,
        first_5_seconds_audio_path=manifest.get("first_5_seconds_audio_path"),
        first_5_seconds_transcript=Path(first_transcript_path).read_text(
            encoding="utf-8"
        )
        if first_transcript_path and Path(first_transcript_path).exists()
        else "",
        first_5_seconds_transcript_available=bool(
            manifest.get("first_5_seconds_transcript_available")
        ),
        preprocess_manifest_path=str(manifest_path),
    )

  def _write_manifest(
      self,
      manifest_path: Path,
      result: models.VideoPreprocessResult,
      transcript_path: Path,
      first_transcript_path: Path,
  ) -> None:
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "strategy_version": _EXTRACTION_STRATEGY_VERSION,
        "source": self._source_fingerprint(result.source),
        "settings": self._settings_fingerprint(),
        "duration_seconds": result.duration_seconds,
        "full_video_frame_evidence": [
            frame.__dict__ for frame in result.full_video_frame_evidence
        ],
        "first_5_seconds_frame_evidence": [
            frame.__dict__ for frame in result.first_5_seconds_frame_evidence
        ],
        "audio_path": result.audio_path,
        "first_5_seconds_audio_path": result.first_5_seconds_audio_path,
        "transcript_path": str(transcript_path),
        "first_5_seconds_transcript_path": str(first_transcript_path),
        "transcript_available": result.transcript_available,
        "first_5_seconds_transcript_available": (
            result.first_5_seconds_transcript_available
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Wire cache into `preprocess`**

At start of `preprocess`, after `local_source` and `video_dir`:

```python
    manifest_path = self._manifest_path(video_dir)
    if not self.refresh_cache:
      cached_result = self._load_manifest_result(manifest_path, local_source)
      if cached_result is not None:
        return cached_result
```

When transcripts are produced, write them:

```python
    transcript_path = video_dir / "transcript.txt"
    first_transcript_path = video_dir / "transcript_first_5s.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    first_transcript_path.write_text(first_5_transcript, encoding="utf-8")
```

Before returning, set manifest path and write manifest:

```python
    result = models.VideoPreprocessResult(
        ...
        preprocess_manifest_path=str(manifest_path),
    )
    self._write_manifest(manifest_path, result, transcript_path, first_transcript_path)
    return result
```

- [ ] **Step 5: Keep stale cleanup scoped**

Preserve the current stale-frame cleanup, but only on rebuild. Do not unlink frames/audio before a cache-hit check.

For YouTube downloads, reuse `source.*` when `refresh_cache=False` and one file already exists:

```python
      downloaded = sorted(video_dir.glob("source.*"))
      if downloaded and not self.refresh_cache:
        return models.VideoSource(
            original_uri=source.original_uri,
            local_path=str(downloaded[0]),
            source_type=source.source_type,
        )
```

- [ ] **Step 6: Run tests and verify green**

Run:

```bash
pytest tests/test_openai_video_preprocessor.py -q
```

Expected: all preprocessor tests pass.

- [ ] **Step 7: Commit**

```bash
git add llms_evaluation/openai_video_preprocessor.py tests/test_openai_video_preprocessor.py
git commit -m "feat: reuse OpenAI preprocessing cache"
```

## Task 4: Evidence Selection And Prompt Timestamp Labels

**Files:**
- Modify: `llms_evaluation/openai_detector.py`
- Modify: `llms_evaluation/openai_api_service.py`
- Test: `tests/test_openai_detector.py`
- Test: `tests/test_openai_api_service.py`

- [ ] **Step 1: Write failing detector evidence-selection tests**

Update `OpenAIServiceStub.evaluate_features` in `tests/test_openai_detector.py`:

```python
  def evaluate_features(
      self,
      prompt_config,
      preprocess_result,
      model_name,
      schema,
      frame_paths,
      transcript,
      transcript_available,
      frame_evidence,
  ):
    self.calls.append({
        "prompt_config": prompt_config,
        "preprocess_result": preprocess_result,
        "model_name": model_name,
        "schema": schema,
        "frame_paths": frame_paths,
        "transcript": transcript,
        "transcript_available": transcript_available,
        "frame_evidence": frame_evidence,
    })
    return self.response
```

Update `_preprocess_result()` to include frame evidence and first-5 transcript:

```python
      full_video_frame_evidence=[
          models.VideoFrameEvidence("full-1.jpg", 0.0, "FULL_VIDEO"),
          models.VideoFrameEvidence("full-2.jpg", 12.0, "FULL_VIDEO"),
      ],
      first_5_seconds_frame_evidence=[
          models.VideoFrameEvidence("first-1.jpg", 2.0, "FIRST_5_SECS_VIDEO"),
      ],
      first_5_seconds_audio_path="audio_first_5s.mp3",
      first_5_seconds_transcript="Shop now.",
      first_5_seconds_transcript_available=True,
```

Append:

```python
def test_detector_selects_first_5s_transcript_and_frame_evidence():
  """First-five-second feature groups use first-five evidence pack."""
  service = OpenAIServiceStub({"features": []})

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=_preprocess_result(),
      feature_configs=[_feature("brand-early", models.VideoSegment.FIRST_5_SECS_VIDEO)],
  )

  assert service.calls[0]["transcript"] == "Shop now."
  assert service.calls[0]["transcript_available"] is True
  assert service.calls[0]["frame_evidence"] == [
      models.VideoFrameEvidence("first-1.jpg", 2.0, "FIRST_5_SECS_VIDEO")
  ]


def test_detector_selects_full_transcript_and_frame_evidence_for_full_video():
  """Full-video feature groups use full-video evidence pack."""
  service = OpenAIServiceStub({"features": []})

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=_preprocess_result(),
      feature_configs=[_feature("supers", models.VideoSegment.FULL_VIDEO)],
  )

  assert service.calls[0]["transcript"] == "Shop now at Chewy."
  assert service.calls[0]["transcript_available"] is True
  assert service.calls[0]["frame_evidence"] == [
      models.VideoFrameEvidence("full-1.jpg", 0.0, "FULL_VIDEO"),
      models.VideoFrameEvidence("full-2.jpg", 12.0, "FULL_VIDEO"),
  ]
```

- [ ] **Step 2: Write failing API prompt test**

Update `tests/test_openai_api_service.py::test_build_input_includes_text_and_image_data_url` call:

```python
      transcript_available=True,
      frame_evidence=[
          models.VideoFrameEvidence(str(frame_path), 3.25, "FULL_VIDEO"),
      ],
```

Add assertions:

```python
  assert "Transcript available: True" in payload[1]["content"][0]["text"]
  assert "Frame evidence:" in payload[1]["content"][0]["text"]
  assert f"{frame_path.name} at 3.25s" in payload[1]["content"][0]["text"]
```

Update `OpenAIAPIService.evaluate_features(...)` test calls to pass:

```python
      transcript="hello",
      transcript_available=True,
      frame_evidence=[],
```

- [ ] **Step 3: Run tests and verify red**

Run:

```bash
pytest tests/test_openai_detector.py tests/test_openai_api_service.py -q
```

Expected: fails because `evaluate_features` signatures do not accept new evidence args.

- [ ] **Step 4: Implement detector evidence-pack selection**

In `llms_evaluation/openai_detector.py`, replace `evaluate_features` service call:

```python
    evidence_pack = self._select_evidence_pack(preprocess_result, feature_configs)
    response = self.openai_service.evaluate_features(
        prompt_config=prompt_config,
        preprocess_result=preprocess_result,
        model_name=config.openai_model,
        schema=models.OPENAI_VIDEO_RESPONSE_SCHEMA,
        frame_paths=evidence_pack["frame_paths"],
        transcript=evidence_pack["transcript"],
        transcript_available=evidence_pack["transcript_available"],
        frame_evidence=evidence_pack["frame_evidence"],
    )
```

Add helper and keep `_select_frames` only if existing tests still call it:

```python
  def _select_evidence_pack(
      self,
      preprocess_result: models.VideoPreprocessResult,
      feature_configs: list[models.VideoFeature],
  ) -> dict:
    if all(
        feature.video_segment == models.VideoSegment.FIRST_5_SECS_VIDEO
        for feature in feature_configs
    ):
      return {
          "frame_paths": preprocess_result.first_5_seconds_frames,
          "frame_evidence": preprocess_result.first_5_seconds_frame_evidence,
          "transcript": preprocess_result.first_5_seconds_transcript,
          "transcript_available": (
              preprocess_result.first_5_seconds_transcript_available
          ),
      }
    return {
        "frame_paths": preprocess_result.full_video_frames,
        "frame_evidence": preprocess_result.full_video_frame_evidence,
        "transcript": preprocess_result.transcript,
        "transcript_available": preprocess_result.transcript_available,
    }
```

- [ ] **Step 5: Implement API prompt timestamp labels**

Change `OpenAIAPIService.build_input` signature:

```python
      transcript_available: bool,
      frame_evidence: list[models.VideoFrameEvidence] | None = None,
```

Add helper:

```python
  def _format_frame_evidence(
      self,
      frame_paths: list[str],
      frame_evidence: list[models.VideoFrameEvidence] | None,
  ) -> str:
    if frame_evidence:
      lines = [
          f"- {Path(frame.path).name} at {frame.timestamp_seconds:.2f}s"
          for frame in frame_evidence
      ]
    else:
      lines = [f"- {Path(frame_path).name}" for frame_path in frame_paths]
    return "Frame evidence:\n" + "\n".join(lines)
```

Update text block:

```python
            f"Video duration seconds: {duration_seconds}\n"
            f"Transcript available: {transcript_available}\n"
            f"Transcript:\n{transcript or '[no transcript available]'}\n\n"
            f"{self._format_frame_evidence(frame_paths, frame_evidence)}\n\n"
            f"{prompt_config.prompt}"
```

Change `OpenAIAPIService.evaluate_features` signature:

```python
      transcript: str,
      transcript_available: bool,
      frame_evidence: list[models.VideoFrameEvidence] | None = None,
```

Pass through to `build_input`:

```python
            transcript=transcript,
            transcript_available=transcript_available,
            frame_evidence=frame_evidence,
```

- [ ] **Step 6: Run tests and verify green**

Run:

```bash
pytest tests/test_openai_detector.py tests/test_openai_api_service.py -q
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add llms_evaluation/openai_detector.py llms_evaluation/openai_api_service.py tests/test_openai_detector.py tests/test_openai_api_service.py
git commit -m "feat: send timestamped OpenAI evidence"
```

## Task 5: Notebook Visualization Dependency

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_openai_evidence_review_dependencies.py`

- [ ] **Step 1: Write failing dependency test**

Create `tests/test_openai_evidence_review_dependencies.py`:

```python
"""Tests for notebook visualization dependencies."""


def test_matplotlib_available_for_evidence_review():
  """Evidence review notebook can render 16:9 figures."""
  import matplotlib

  assert matplotlib.__version__
```

- [ ] **Step 2: Run test and verify red**

Run:

```bash
pytest tests/test_openai_evidence_review_dependencies.py -q
```

Expected in a clean environment before dependency install: fails with `ModuleNotFoundError: No module named 'matplotlib'`.

- [ ] **Step 3: Add dependency**

Append to `requirements.txt`:

```text
matplotlib==3.10.3
```

- [ ] **Step 4: Install dependency for local verification**

Run:

```bash
python3 -m pip install -r requirements.txt
```

Expected: command exits 0. If network is blocked by sandboxing, rerun with escalated approval.

- [ ] **Step 5: Run test and verify green**

Run:

```bash
pytest tests/test_openai_evidence_review_dependencies.py -q
```

Expected: test passes.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/test_openai_evidence_review_dependencies.py
git commit -m "chore: add evidence review plotting dependency"
```

## Task 6: Validation Sampling And Evidence Review Helpers

**Files:**
- Create: `notebooks/20260619_openai_evidence_review/README.md`
- Create: `notebooks/20260619_openai_evidence_review/evidence_review.py`
- Test: `tests/test_openai_evidence_review.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_openai_evidence_review.py`:

```python
"""Tests for OpenAI evidence review notebook helpers."""

import importlib.util
import json
from pathlib import Path


def _load_helpers():
  helper_path = (
      Path(__file__).resolve().parents[1]
      / "notebooks"
      / "20260619_openai_evidence_review"
      / "evidence_review.py"
  )
  spec = importlib.util.spec_from_file_location("evidence_review", helper_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_discover_sample_videos_chooses_two_per_platform_with_fixed_seed(tmp_path):
  """Sample discovery groups by platform and is deterministic."""
  helpers = _load_helpers()
  for platform in ("facebook", "google"):
    for index in range(3):
      video_path = tmp_path / platform / "videos" / f"ad_{index}.mp4"
      video_path.parent.mkdir(parents=True, exist_ok=True)
      video_path.write_bytes(b"video")

  first = helpers.discover_sample_videos(tmp_path, per_platform=2, seed=7)
  second = helpers.discover_sample_videos(tmp_path, per_platform=2, seed=7)

  assert first == second
  assert sorted(first) == ["facebook", "google"]
  assert all(len(paths) == 2 for paths in first.values())


def test_build_validation_command_uses_generic_metadata_when_missing(tmp_path):
  """Validation command remains runnable when no metadata file is provided."""
  helpers = _load_helpers()
  videos = {
      "google": [tmp_path / "google" / "videos" / "ad.mp4"],
  }

  command = helpers.build_validation_command(
      selected_videos=videos,
      output_json=tmp_path / "outputs" / "sample.json",
  )

  assert "--llm_provider" in command
  assert "OPENAI" in command
  assert "--creative_provider_type" in command
  assert "LOCAL" in command
  assert "-brand_name" in command
  assert "Unknown Brand" in command
  assert "-run_long_form_abcd" in command
  assert "-run_shorts" in command


def test_transcript_snippet_prefers_timecoded_segments(tmp_path):
  """Transcript snippets use segments near the frame timestamp when available."""
  helpers = _load_helpers()
  manifest = {
      "transcript_segments": [
          {"start": 0.0, "end": 1.0, "text": "early"},
          {"start": 3.0, "end": 4.0, "text": "near frame"},
          {"start": 8.0, "end": 9.0, "text": "late"},
      ],
      "transcript": "fallback transcript",
  }

  assert helpers.transcript_snippet(manifest, 3.5, window_seconds=1.0) == "near frame"


def test_load_assessment_feature_rows_flattens_detected_features(tmp_path):
  """Feature rows include long-form and shorts detected status."""
  helpers = _load_helpers()
  assessment_path = tmp_path / "assessment.json"
  assessment_path.write_text(
      json.dumps({
          "assessments": [{
              "video_uri": "ad.mp4",
              "long_form_abcd_evaluated_features": [{
                  "feature": {"id": "a_supers", "name": "Supers"},
                  "detected": True,
              }],
              "shorts_evaluated_features": [{
                  "feature": {"id": "shorts_human_voice", "name": "Voice"},
                  "detected": False,
              }],
          }]
      }),
      encoding="utf-8",
  )

  rows = helpers.load_feature_rows(assessment_path)

  assert rows["ad.mp4"] == [
      ("Long-form", "a_supers", "Supers", True),
      ("Shorts", "shorts_human_voice", "Voice", False),
  ]
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
pytest tests/test_openai_evidence_review.py -q
```

Expected: fails because helper module does not exist.

- [ ] **Step 3: Implement helper module**

Create `notebooks/20260619_openai_evidence_review/evidence_review.py`:

```python
"""Helpers for OpenAI evidence review notebook."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
DEFAULT_METADATA = {
    "brand_name": "Unknown Brand",
    "brand_variations": "Unknown Brand",
    "branded_products": "product",
    "branded_products_categories": "product,service",
    "branded_call_to_actions": "shop now,learn more,visit site",
}


def discover_sample_videos(
    sample_root: str | Path,
    per_platform: int = 2,
    seed: int = 20260619,
) -> dict[str, list[Path]]:
  """Return up to N videos per first-level platform directory."""
  sample_root = Path(sample_root)
  rng = random.Random(seed)
  selected = {}
  if not sample_root.exists():
    return selected
  for platform_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
    candidates = sorted(
        path
        for path in platform_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not candidates:
      continue
    chosen = candidates if len(candidates) <= per_platform else rng.sample(candidates, per_platform)
    selected[platform_dir.name] = sorted(chosen)
  return selected


def build_validation_command(
    selected_videos: dict[str, list[Path]],
    output_json: str | Path,
    metadata: dict | None = None,
) -> list[str]:
  """Build main.py command for validation sample."""
  metadata = metadata or DEFAULT_METADATA
  video_uris = ",".join(
      str(path)
      for platform_paths in selected_videos.values()
      for path in platform_paths
  )
  return [
      sys.executable,
      "main.py",
      "--llm_provider",
      "OPENAI",
      "--creative_provider_type",
      "LOCAL",
      "--video_uris",
      video_uris,
      "-brand_name",
      metadata["brand_name"],
      "-brand_variations",
      metadata["brand_variations"],
      "-branded_products",
      metadata["branded_products"],
      "-branded_products_categories",
      metadata["branded_products_categories"],
      "-branded_call_to_actions",
      metadata["branded_call_to_actions"],
      "-run_long_form_abcd",
      "-run_shorts",
      "-assessment_file",
      str(output_json),
  ]


def run_validation_sample(command: list[str]) -> subprocess.CompletedProcess | None:
  """Run validation command if OPENAI_API_KEY is present."""
  if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not loaded; skipping live validation run.")
    return None
  return subprocess.run(command, check=True, text=True)


def load_json(path: str | Path) -> dict:
  """Load JSON file."""
  return json.loads(Path(path).read_text(encoding="utf-8"))


def transcript_snippet(
    manifest: dict,
    timestamp_seconds: float,
    window_seconds: float = 2.5,
) -> str:
  """Return transcript text near a frame timestamp."""
  segments = manifest.get("transcript_segments") or []
  nearby = [
      segment["text"]
      for segment in segments
      if segment.get("start", 0) <= timestamp_seconds + window_seconds
      and segment.get("end", 0) >= timestamp_seconds - window_seconds
  ]
  if nearby:
    return " ".join(nearby)
  return manifest.get("transcript") or "[transcript is not timecoded]"


def load_feature_rows(assessment_path: str | Path) -> dict[str, list[tuple]]:
  """Flatten assessment features into checklist rows by video URI."""
  data = load_json(assessment_path)
  rows_by_video = {}
  for assessment in data.get("assessments", []):
    rows = []
    for category, key in (
        ("Long-form", "long_form_abcd_evaluated_features"),
        ("Shorts", "shorts_evaluated_features"),
    ):
      for feature_eval in assessment.get(key, []):
        feature = feature_eval.get("feature", {})
        rows.append((
            category,
            feature.get("id", ""),
            feature.get("name", feature.get("id", "")),
            bool(feature_eval.get("detected")),
        ))
    rows_by_video[assessment.get("video_uri", "")] = rows
  return rows_by_video


def render_evidence_figure(
    frame_path: str | Path,
    transcript_text: str,
    feature_rows: list[tuple],
    output_path: str | Path,
    title: str,
) -> Path:
  """Render a monochrome 16:9 evidence review figure."""
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
  grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.08)
  ax_left = fig.add_subplot(grid[0, 0])
  ax_right = fig.add_subplot(grid[0, 1])
  ax_left.axis("off")
  ax_right.axis("off")
  image = plt.imread(frame_path)
  ax_left.imshow(image)
  ax_left.set_title(title, fontsize=24, fontweight=340, loc="left")
  ax_left.text(
      0,
      -0.08,
      transcript_text,
      transform=ax_left.transAxes,
      fontsize=18,
      va="top",
      wrap=True,
  )
  y = 0.98
  ax_right.text(0, y, "FEATURE CHECKLIST", fontsize=14, family="monospace")
  y -= 0.08
  for category, feature_id, name, detected in feature_rows[:18]:
    mark = "[x]" if detected else "[ ]"
    ax_right.text(0, y, f"{mark} {category}: {name or feature_id}", fontsize=18)
    y -= 0.052
  fig.savefig(output_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  return output_path
```

- [ ] **Step 4: Create README**

Create `notebooks/20260619_openai_evidence_review/README.md`:

```markdown
# OpenAI Evidence Review

This folder contains helper code and a notebook for reviewing OpenAI evidence
extraction outputs after the detector implementation is complete.

Expected local inputs:

- `sample_videos/` symlink or directory with platform subdirectories.
- `OPENAI_API_KEY` loaded from `.zshrc` for the live validation run.
- `outputs/openai_validation_sample/abcd_results.json` after running the
  validation command.

The notebook renders 16:9 monochrome figures for Keynote: extracted frame and
nearby transcript on the left, detected feature checklist on the right.

Generated figures and validation outputs are local artifacts and should remain
untracked unless explicitly requested.
```

- [ ] **Step 5: Run tests and verify green**

Run:

```bash
pytest tests/test_openai_evidence_review.py -q
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add notebooks/20260619_openai_evidence_review/README.md notebooks/20260619_openai_evidence_review/evidence_review.py tests/test_openai_evidence_review.py
git commit -m "feat: add OpenAI evidence review helpers"
```

## Task 7: Evidence Review Notebook

**Files:**
- Create: `notebooks/20260619_openai_evidence_review/openai_evidence_review.ipynb`
- Modify: `notebooks/20260619_openai_evidence_review/README.md`
- Test: `tests/test_openai_evidence_review.py`

- [ ] **Step 1: Write failing notebook structure test**

Append to `tests/test_openai_evidence_review.py`:

```python
def test_evidence_review_notebook_exists_and_references_helpers():
  """Notebook is present and uses the helper module."""
  notebook_path = (
      Path(__file__).resolve().parents[1]
      / "notebooks"
      / "20260619_openai_evidence_review"
      / "openai_evidence_review.ipynb"
  )

  notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
  source = "\n".join(
      "".join(cell.get("source", []))
      for cell in notebook.get("cells", [])
  )

  assert "discover_sample_videos" in source
  assert "build_validation_command" in source
  assert "render_evidence_figure" in source
  assert "outputs/openai_validation_sample" in source
```

- [ ] **Step 2: Run test and verify red**

Run:

```bash
pytest tests/test_openai_evidence_review.py::test_evidence_review_notebook_exists_and_references_helpers -q
```

Expected: fails because notebook does not exist.

- [ ] **Step 3: Create notebook JSON**

Create `notebooks/20260619_openai_evidence_review/openai_evidence_review.ipynb` with this minimal notebook:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# OpenAI Evidence Review\n",
        "\n",
        "Renders 16:9 frame/transcript/checklist figures for selected validation videos."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "from evidence_review import discover_sample_videos, build_validation_command, run_validation_sample\n",
        "from evidence_review import load_json, load_feature_rows, transcript_snippet, render_evidence_figure\n",
        "\n",
        "ROOT = Path.cwd().parents[1] if Path.cwd().name == '20260619_openai_evidence_review' else Path.cwd()\n",
        "OUTPUT_DIR = ROOT / 'outputs' / 'openai_validation_sample'\n",
        "ASSESSMENT_JSON = OUTPUT_DIR / 'abcd_results.json'\n",
        "FIGURES_DIR = ROOT / 'notebooks' / '20260619_openai_evidence_review' / 'figures'\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "selected = discover_sample_videos(ROOT / 'sample_videos', per_platform=2, seed=20260619)\n",
        "selected\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "command = build_validation_command(selected, ASSESSMENT_JSON)\n",
        "print(' '.join(command))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Uncomment after implementation when OPENAI_API_KEY is loaded from .zshrc.\n",
        "# run_validation_sample(command)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "feature_rows = load_feature_rows(ASSESSMENT_JSON)\n",
        "feature_rows.keys()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Render figures by pairing each assessment with its cache manifest. Run this after validation output exists."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 4: Update README with live validation command**

Append:

```markdown
## Live Validation

After implementation and unit tests pass, run from the repo root:

```bash
zsh -lc 'source ~/.zshrc && python3 -c "from pathlib import Path; import sys; sys.path.insert(0, \"notebooks/20260619_openai_evidence_review\"); from evidence_review import discover_sample_videos, build_validation_command, run_validation_sample; selected = discover_sample_videos(Path(\"sample_videos\")); command = build_validation_command(selected, Path(\"outputs/openai_validation_sample/abcd_results.json\")); run_validation_sample(command)"'
```

This uses generic metadata when no local metadata file is supplied. The run is
intended to validate extraction, caching, and visualization plumbing, not brand
classification accuracy.
```

- [ ] **Step 5: Run tests and verify green**

Run:

```bash
pytest tests/test_openai_evidence_review.py -q
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add notebooks/20260619_openai_evidence_review/openai_evidence_review.ipynb notebooks/20260619_openai_evidence_review/README.md tests/test_openai_evidence_review.py
git commit -m "docs: add OpenAI evidence review notebook"
```

## Task 8: Integration Verification And Live Sample Run

**Files:**
- Generated local outputs only: `outputs/openai_validation_sample/*`
- Generated local figures only: `notebooks/20260619_openai_evidence_review/figures/*`

- [ ] **Step 1: Run targeted unit suite**

Run:

```bash
pytest \
  tests/test_openai_cli_config.py \
  tests/test_openai_video_preprocessor.py \
  tests/test_openai_detector.py \
  tests/test_openai_api_service.py \
  tests/test_openai_evidence_review.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Check OpenAI CLI help includes cache flag**

Run:

```bash
python3 main.py --help | rg -- '--refresh_cache|--max_frames|--frame_sample_rate'
```

Expected: output includes all three flags.

- [ ] **Step 4: Run live validation sample after implementation**

Run:

```bash
zsh -lc 'source ~/.zshrc && python3 -c "from pathlib import Path; import sys; sys.path.insert(0, \"notebooks/20260619_openai_evidence_review\"); from evidence_review import discover_sample_videos, build_validation_command, run_validation_sample; selected = discover_sample_videos(Path(\"sample_videos\")); print(selected); command = build_validation_command(selected, Path(\"outputs/openai_validation_sample/abcd_results.json\")); print(\" \".join(command)); run_validation_sample(command)"'
```

Expected when `OPENAI_API_KEY` is present: command exits 0 and writes:

```text
outputs/openai_validation_sample/abcd_results.json
outputs/openai_validation_sample/abcd_results.csv
```

Expected when `OPENAI_API_KEY` is missing: command prints `OPENAI_API_KEY not loaded; skipping live validation run.` and no success claim is made about live validation.

- [ ] **Step 5: Generate at least one evidence figure per sampled video**

Run a short Python command after validation output exists:

```bash
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "notebooks/20260619_openai_evidence_review")
from evidence_review import load_feature_rows, render_evidence_figure

rows_by_video = load_feature_rows("outputs/openai_validation_sample/abcd_results.json")
figures_dir = Path("notebooks/20260619_openai_evidence_review/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

for index, (video_uri, rows) in enumerate(rows_by_video.items(), start=1):
  cache_dirs = sorted(Path(".cache/abcds-detector").glob("*"))
  frame = next((p for p in cache_dirs for p in (p / "frames" / "full").glob("*.jpg")), None)
  if frame is None:
    raise SystemExit("No cached frame found for evidence figure generation")
  render_evidence_figure(
      frame_path=frame,
      transcript_text="Transcript excerpt unavailable in quick render.",
      feature_rows=rows,
      output_path=figures_dir / f"evidence_{index:02d}.png",
      title=Path(video_uri).name,
  )
print(f"Generated {len(list(figures_dir.glob('*.png')))} figures in {figures_dir}")
PY
```

Expected: prints generated figure count and creates PNG files under `notebooks/20260619_openai_evidence_review/figures/`.

- [ ] **Step 6: Confirm generated/private artifacts are untracked**

Run:

```bash
git status --short
```

Expected: no private videos, cache files, audio files, transcripts, validation outputs, or generated figures are staged. If generated outputs appear as untracked files, leave them untracked and mention them in the final report.

- [ ] **Step 7: Confirm no source commit is needed**

Task 8 should not require source edits. Do not commit generated validation outputs or figures unless the user explicitly requests it.

```bash
git status --short
```

Expected: only untracked local artifacts, such as `outputs/openai_validation_sample/*` or `notebooks/20260619_openai_evidence_review/figures/*`, may appear. Leave them untracked.

## Self-Review Checklist

- Spec coverage: Tasks 1-4 cover cache, extraction, first-5 routing, prompt timestamp labels. Task 5 covers plotting dependencies. Tasks 6-8 cover live validation and notebook review.
- Isolation: No task edits `docs/feature_inventory/*`. No task edits `features_repository/*`.
- TDD: Each behavior-changing task starts with failing tests and red/green commands.
- Private artifacts: Validation outputs, cache, frames, audio, transcripts, and figures remain untracked unless explicitly requested.

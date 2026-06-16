# OpenAI LLM CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenAI-backed, LLM-only batch CLI path that evaluates local videos and YouTube URLs and writes JSON results.

**Architecture:** Keep current feature configs, prompts, and `FeatureEvaluation` result shape. Add a separate OpenAI path for local/YouTube videos: resolve sources, preprocess media locally, transcribe audio, evaluate sampled frames plus transcript with OpenAI structured outputs, then write a JSON assessment file. Preserve current Gemini/GCS behavior.

**Tech Stack:** Python 3.11, argparse, dataclasses, OpenAI Python SDK, yt-dlp, ffmpeg CLI, pytest, pyink/ruff.

---

## File Structure

- Modify `requirements.txt`: add `openai` and `yt-dlp`.
- Modify `models.py`: add LLM provider enum, local source/result dataclasses, OpenAI schemas.
- Modify `configuration.py`: add OpenAI and local preprocessing config fields.
- Modify `utils.py`: parse new CLI args and populate config.
- Modify `creative_providers/creative_provider_registry.py`: register `LOCAL`.
- Create `creative_providers/local_creative_provider.py`: normalize local paths and YouTube URLs for OpenAI processing.
- Create `llms_evaluation/openai_video_preprocessor.py`: download YouTube videos, extract frames/audio, transcribe audio.
- Create `llms_evaluation/openai_api_service.py`: wrap OpenAI API calls and structured output parsing.
- Create `llms_evaluation/openai_detector.py`: OpenAI metadata and feature evaluation facade.
- Modify `evaluation_services/video_evaluation_service.py`: route feature evaluation to OpenAI when configured.
- Modify `main.py`: handle OpenAI-local source flow and collect assessments for JSON output.
- Modify `helpers/generic_helpers.py`: add JSON assessment serialization/writer.
- Modify `tests/test_abcd_parameters.py`: keep existing config test current.
- Create focused tests under `tests/` for config, local provider, preprocessing, OpenAI parsing, and JSON writer.
- Update `README.md`: add CLI batch quickstart and local sample video path.

## Task 1: Repository Remote And Branch Setup

**Files:**
- No source file changes.

- [ ] **Step 1: Rename current upstream remote**

Run:

```bash
git remote rename origin upstream
```

Expected:

```text
No output and exit code 0
```

- [ ] **Step 2: Add fork as origin**

Run:

```bash
git remote add origin git@github.com:gweng-chwy/abcds-detector.git
```

Expected:

```text
No output and exit code 0
```

- [ ] **Step 3: Create implementation branch**

Run:

```bash
git checkout -b openai-llm-cli
```

Expected:

```text
Switched to a new branch 'openai-llm-cli'
```

- [ ] **Step 4: Verify remotes**

Run:

```bash
git remote -v
```

Expected contains:

```text
origin	git@github.com:gweng-chwy/abcds-detector.git (fetch)
origin	git@github.com:gweng-chwy/abcds-detector.git (push)
upstream	git@github.com:google-marketing-solutions/abcds-detector.git (fetch)
upstream	git@github.com:google-marketing-solutions/abcds-detector.git (push)
```

## Task 2: Dependencies And Baseline Test Tooling

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add runtime dependencies**

Change `requirements.txt` to:

```text
google-cloud-aiplatform==1.97.0
google-cloud-videointelligence==2.16.1
google-cloud-storage==2.19.0
moviepy==1.0.3
google-api-python-client==2.172.0
pandas==2.3.0
pyarrow==20.0.0
openai==1.99.9
yt-dlp==2026.1.26
```

- [ ] **Step 2: Install dependencies**

Run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Expected:

```text
Both commands exit 0
```

- [ ] **Step 3: Run baseline tests**

Run:

```bash
pytest -q
```

Expected before later tasks:

```text
Tests may fail because current test mocks are stale. Record exact failures in the task notes and fix them in Task 3.
```

- [ ] **Step 4: Commit dependencies**

Run:

```bash
git add requirements.txt
git commit -m "chore: add OpenAI CLI dependencies"
```

Expected:

```text
Commit created
```

## Task 3: Config Models And CLI Parsing

**Files:**
- Modify: `models.py`
- Modify: `configuration.py`
- Modify: `utils.py`
- Modify: `tests/test_abcd_parameters.py`
- Create: `tests/test_openai_cli_config.py`

- [ ] **Step 1: Write failing CLI config tests**

Create `tests/test_openai_cli_config.py`:

```python
from utils import parse_args, build_abcd_params_config
import models


def test_parse_openai_local_batch_args():
  args = parse_args([
      "--llm_provider",
      "OPENAI",
      "--creative_provider_type",
      "LOCAL,YOUTUBE",
      "--video_uris",
      "sample_videos/google/videos/example.mp4,https://www.youtube.com/watch?v=abc",
      "--openai_model",
      "gpt-5.4-mini",
      "--openai_transcription_model",
      "gpt-4o-transcribe",
      "--cache_dir",
      ".cache/abcds-detector",
      "--max_frames",
      "24",
      "--frame_sample_rate",
      "1.0",
      "--assessment_file",
      "outputs/results.json",
      "--run_long_form_abcd",
  ])

  config = build_abcd_params_config(args)

  assert config.llm_provider_type == models.LLMProviderType.OPENAI
  assert config.creative_provider_types == [
      models.CreativeProviderType.LOCAL,
      models.CreativeProviderType.YOUTUBE,
  ]
  assert config.openai_model == "gpt-5.4-mini"
  assert config.openai_transcription_model == "gpt-4o-transcribe"
  assert config.cache_dir == ".cache/abcds-detector"
  assert config.max_frames == 24
  assert config.frame_sample_rate == 1.0
  assert config.assessment_file == "outputs/results.json"
  assert config.video_uris == [
      "sample_videos/google/videos/example.mp4",
      "https://www.youtube.com/watch?v=abc",
  ]


def test_parse_defaults_keep_gemini_and_gcs():
  args = parse_args([
      "-pi",
      "project",
      "-bn",
      "bucket",
      "-vu",
      "gs://bucket/brand/videos/",
  ])

  config = build_abcd_params_config(args)

  assert config.llm_provider_type == models.LLMProviderType.GEMINI
  assert config.creative_provider_type == models.CreativeProviderType.GCS
  assert config.creative_provider_types == [models.CreativeProviderType.GCS]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_openai_cli_config.py -q
```

Expected:

```text
FAIL because LLMProviderType, creative_provider_types, and OpenAI config fields do not exist
```

- [ ] **Step 3: Add models**

In `models.py`, add after `CreativeProviderType`:

```python
class LLMProviderType(Enum):
  """Enum that represents supported LLM providers."""

  GEMINI = "GEMINI"
  OPENAI = "OPENAI"


@dataclass
class VideoSource:
  """Resolved video source for local OpenAI evaluation."""

  original_uri: str
  local_path: str
  source_type: str


@dataclass
class VideoPreprocessResult:
  """Local media evidence prepared for OpenAI evaluation."""

  source: VideoSource
  duration_seconds: float
  full_video_frames: list[str]
  first_5_seconds_frames: list[str]
  audio_path: str | None
  transcript: str
  transcript_available: bool
```

Extend `CreativeProviderType` to:

```python
class CreativeProviderType(Enum):
  """Enum that represents creative source providers."""

  GCS = "GCS"
  YOUTUBE = "YOUTUBE"
  LOCAL = "LOCAL"
```

Add OpenAI response wrapper schema after `VIDEO_RESPONSE_SCHEMA`:

```python
OPENAI_VIDEO_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "features": VIDEO_RESPONSE_SCHEMA,
    },
    "required": ["features"],
    "additionalProperties": False,
}
```

- [ ] **Step 4: Add config defaults**

In `configuration.py`, import `LLMProviderType` and update `__init__` with:

```python
self.llm_provider_type = LLMProviderType.GEMINI
self.creative_provider_types = [CreativeProviderType.GCS]
self.openai_model: str = "gpt-5.4-mini"
self.openai_transcription_model: str = "gpt-4o-transcribe"
self.cache_dir: str = ".cache/abcds-detector"
self.max_frames: int = 24
self.frame_sample_rate: float = 1.0
```

Also make the existing Knowledge Graph key assignment safe when argparse returns
`None`:

```python
self.knowledge_graph_api_key = (knowledge_graph_api_key or "").strip()
```

Extend `set_parameters` signature with:

```python
llm_provider: str = "GEMINI",
openai_model: str = "gpt-5.4-mini",
openai_transcription_model: str = "gpt-4o-transcribe",
cache_dir: str = ".cache/abcds-detector",
max_frames: int = 24,
frame_sample_rate: float = 1.0,
```

Inside `set_parameters`, add:

```python
self.llm_provider_type = (
    LLMProviderType.OPENAI
    if (llm_provider or "GEMINI").upper() == LLMProviderType.OPENAI.value
    else LLMProviderType.GEMINI
)
self.openai_model = openai_model or "gpt-5.4-mini"
self.openai_transcription_model = (
    openai_transcription_model or "gpt-4o-transcribe"
)
self.cache_dir = cache_dir or ".cache/abcds-detector"
self.max_frames = int(max_frames or 24)
self.frame_sample_rate = float(frame_sample_rate or 1.0)
self.creative_provider_types = []
for provider in (creative_provider_type or CreativeProviderType.GCS.value).split(","):
  provider = provider.strip().upper()
  if provider:
    self.creative_provider_types.append(CreativeProviderType(provider))
self.creative_provider_type = self.creative_provider_types[0]
```

- [ ] **Step 5: Add argparse options**

In `utils.py`, pass new args into `set_parameters`:

```python
llm_provider=args.llm_provider,
openai_model=args.openai_model,
openai_transcription_model=args.openai_transcription_model,
cache_dir=args.cache_dir,
max_frames=args.max_frames,
frame_sample_rate=args.frame_sample_rate,
```

Change existing feature parsing from:

```python
features_to_evaluate=args.features_to_evaluate.split(","),
```

to:

```python
features_to_evaluate=(
    [feature.strip() for feature in args.features_to_evaluate.split(",")]
    if args.features_to_evaluate
    else []
),
```

Add parser arguments near existing LLM args:

```python
parser.add_argument(
    "--llm_provider",
    "-llmp",
    help="LLM provider. Supported values: GEMINI, OPENAI.",
    default="GEMINI",
)
parser.add_argument(
    "--openai_model",
    "-oaim",
    help="OpenAI model for video feature evaluation.",
    default="gpt-5.4-mini",
)
parser.add_argument(
    "--openai_transcription_model",
    "-oaitm",
    help="OpenAI transcription model.",
    default="gpt-4o-transcribe",
)
parser.add_argument(
    "--cache_dir",
    "-cdir",
    help="Local cache directory for downloads and preprocessing artifacts.",
    default=".cache/abcds-detector",
)
parser.add_argument(
    "--max_frames",
    "-mfr",
    help="Maximum frames sampled per video segment.",
    default=24,
)
parser.add_argument(
    "--frame_sample_rate",
    "-fsr",
    help="Frame sampling rate in frames per second.",
    default=1.0,
)
```

In `Configuration.set_llm_params`, keep current defaults when optional CLI
values are not supplied:

```python
self.llm_params.model_name = llm_name or self.llm_params.model_name
self.llm_params.location = location or self.llm_params.location
self.llm_params.generation_config = {
    "max_output_tokens": int(max_output_tokens or 65535),
    "temperature": float(temperature if temperature is not None else 1),
    "top_p": float(top_p if top_p is not None else 0.95),
    "response_schema": {"type": "string"},
}
```

- [ ] **Step 6: Fix existing config test**

Update `tests/test_abcd_parameters.py` `ArgsMock` and test instance to include:

```python
extract_brand_metadata: bool
run_long_form_abcd: bool
run_shorts: bool
features_to_evaluate: str
creative_provider_type: str
llm_provider: str
llm_location: str
openai_model: str
openai_transcription_model: str
cache_dir: str
max_frames: int
frame_sample_rate: float
```

Use these values in the `ArgsMock` instance:

```python
extract_brand_metadata=True,
run_long_form_abcd=True,
run_shorts=True,
features_to_evaluate="",
creative_provider_type="GCS",
llm_provider="GEMINI",
llm_location="us-central1",
openai_model="gpt-5.4-mini",
openai_transcription_model="gpt-4o-transcribe",
cache_dir=".cache/abcds-detector",
max_frames=24,
frame_sample_rate=1.0,
```

Remove assertions for nonexistent config fields such as `config.llm_name`,
`config.video_size_limit_mb`, `config.max_output_tokens`, `config.temperature`,
`config.top_p`, and `config.top_k`. Replace with:

```python
assert config.llm_params.model_name is not None
assert config.llm_params.generation_config["max_output_tokens"] is not None
assert config.llm_params.generation_config["temperature"] is not None
assert config.llm_params.generation_config["top_p"] is not None
assert config.llm_provider_type is not None
assert config.openai_model == "gpt-5.4-mini"
```

- [ ] **Step 7: Run config tests**

Run:

```bash
pytest tests/test_abcd_parameters.py tests/test_openai_cli_config.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 8: Commit config work**

Run:

```bash
git add models.py configuration.py utils.py tests/test_abcd_parameters.py tests/test_openai_cli_config.py
git commit -m "feat: add OpenAI CLI configuration"
```

Expected:

```text
Commit created
```

## Task 4: Local And YouTube Source Resolution

**Files:**
- Create: `creative_providers/local_creative_provider.py`
- Modify: `creative_providers/creative_provider_registry.py`
- Create: `tests/test_local_creative_provider.py`

- [ ] **Step 1: Write failing provider tests**

Create `tests/test_local_creative_provider.py`:

```python
from dataclasses import dataclass

import models
from creative_providers.local_creative_provider import (
    LocalCreativeProvider,
    is_youtube_url,
)


@dataclass
class ConfigMock:
  video_uris: list[str]


def test_is_youtube_url_matches_supported_hosts():
  assert is_youtube_url("https://www.youtube.com/watch?v=abc")
  assert is_youtube_url("https://youtu.be/abc")
  assert not is_youtube_url("sample_videos/google/videos/ad.mp4")


def test_local_provider_returns_video_sources(tmp_path):
  video = tmp_path / "ad.mp4"
  video.write_bytes(b"fake-video")
  config = ConfigMock([
      str(video),
      "https://www.youtube.com/watch?v=abc",
  ])

  sources = list(LocalCreativeProvider().get_creative_sources(config))

  assert sources[0] == models.VideoSource(
      original_uri=str(video),
      local_path=str(video),
      source_type="LOCAL",
  )
  assert sources[1] == models.VideoSource(
      original_uri="https://www.youtube.com/watch?v=abc",
      local_path="",
      source_type="YOUTUBE",
  )
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_local_creative_provider.py -q
```

Expected:

```text
FAIL because local_creative_provider.py does not exist
```

- [ ] **Step 3: Implement local provider**

Create `creative_providers/local_creative_provider.py`:

```python
"""Creative provider for local files and YouTube URLs."""

from urllib.parse import urlparse

import models


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def is_youtube_url(uri: str) -> bool:
  """Return True when uri points to a YouTube host."""
  parsed = urlparse(uri)
  return parsed.scheme in {"http", "https"} and parsed.netloc in YOUTUBE_HOSTS


class LocalCreativeProvider:
  """Resolves local files and YouTube URLs for OpenAI processing."""

  def get_creative_uris(self, config) -> list[str]:
    """Return original URIs for compatibility with provider factory callers."""
    return list(config.video_uris)

  def get_creative_sources(self, config) -> list[models.VideoSource]:
    """Return normalized video source objects."""
    sources = []
    for uri in config.video_uris:
      if is_youtube_url(uri):
        sources.append(
            models.VideoSource(
                original_uri=uri,
                local_path="",
                source_type=models.CreativeProviderType.YOUTUBE.value,
            )
        )
      else:
        sources.append(
            models.VideoSource(
                original_uri=uri,
                local_path=uri,
                source_type=models.CreativeProviderType.LOCAL.value,
            )
        )
    return sources
```

- [ ] **Step 4: Register local provider**

In `creative_providers/creative_provider_registry.py`, import:

```python
from creative_providers import local_creative_provider
```

Add registration:

```python
provider_factory.register_provider(
    models.CreativeProviderType.LOCAL.value,
    local_creative_provider.LocalCreativeProvider,
)
```

- [ ] **Step 5: Run provider tests**

Run:

```bash
pytest tests/test_local_creative_provider.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 6: Commit provider work**

Run:

```bash
git add creative_providers/local_creative_provider.py creative_providers/creative_provider_registry.py tests/test_local_creative_provider.py
git commit -m "feat: add local creative provider"
```

Expected:

```text
Commit created
```

## Task 5: Video Preprocessing

**Files:**
- Create: `llms_evaluation/openai_video_preprocessor.py`
- Create: `tests/test_openai_video_preprocessor.py`

- [ ] **Step 1: Write failing preprocessing tests**

Create `tests/test_openai_video_preprocessor.py`:

```python
from pathlib import Path
from unittest import mock

import models
from llms_evaluation.openai_video_preprocessor import (
    build_cache_key,
    VideoPreprocessor,
)


class OpenAIServiceMock:
  def transcribe_audio(self, audio_path):
    return "Shop Chewy for pet food."


def test_build_cache_key_is_stable():
  first = build_cache_key("https://www.youtube.com/watch?v=abc")
  second = build_cache_key("https://www.youtube.com/watch?v=abc")

  assert first == second
  assert len(first) == 16


def test_preprocess_local_video_invokes_ffmpeg(tmp_path):
  source_video = tmp_path / "ad.mp4"
  source_video.write_bytes(b"fake-video")
  source = models.VideoSource(
      original_uri=str(source_video),
      local_path=str(source_video),
      source_type="LOCAL",
  )
  preprocessor = VideoPreprocessor(
      cache_dir=str(tmp_path / "cache"),
      max_frames=2,
      frame_sample_rate=1.0,
      openai_service=OpenAIServiceMock(),
  )

  def fake_run(cmd):
    audio_path = tmp_path / "cache" / build_cache_key(str(source_video)) / "audio.mp4"
    if "ffprobe" in cmd:
      return mock.Mock(stdout="15.0")
    if str(audio_path) in cmd:
      audio_path.parent.mkdir(parents=True, exist_ok=True)
      audio_path.write_bytes(b"fake-audio")
    return mock.Mock(stdout="")

  with mock.patch.object(preprocessor, "_run", side_effect=fake_run) as run_mock:
    with mock.patch.object(preprocessor, "_probe_duration", return_value=15.0):
      result = preprocessor.preprocess(source)

  assert result.source == source
  assert result.duration_seconds == 15.0
  assert result.transcript == "Shop Chewy for pet food."
  assert result.transcript_available
  assert run_mock.call_count == 3
  assert Path(result.audio_path).name == "audio.mp4"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_openai_video_preprocessor.py -q
```

Expected:

```text
FAIL because openai_video_preprocessor.py does not exist
```

- [ ] **Step 3: Implement preprocessor**

Create `llms_evaluation/openai_video_preprocessor.py`:

```python
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
      openai_service,
  ):
    self.cache_dir = Path(cache_dir)
    self.max_frames = max_frames
    self.frame_sample_rate = frame_sample_rate
    self.openai_service = openai_service

  def preprocess(self, source: models.VideoSource) -> models.VideoPreprocessResult:
    """Download if needed, then extract frames, audio, transcript, and duration."""
    local_source = self._ensure_local(source)
    video_dir = self.cache_dir / build_cache_key(local_source.original_uri)
    full_frames_dir = video_dir / "frames" / "full"
    first_frames_dir = video_dir / "frames" / "first_5s"
    audio_path = video_dir / "audio.mp4"
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    first_frames_dir.mkdir(parents=True, exist_ok=True)

    duration_seconds = self._probe_duration(local_source.local_path)
    self._extract_frames(local_source.local_path, full_frames_dir)
    self._extract_first_5s_frames(local_source.local_path, first_frames_dir)
    self._extract_audio(local_source.local_path, audio_path)

    transcript = ""
    transcript_available = False
    if audio_path.exists():
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
        audio_path=str(audio_path),
        transcript=transcript,
        transcript_available=transcript_available,
    )

  def _ensure_local(self, source: models.VideoSource) -> models.VideoSource:
    if source.source_type != models.CreativeProviderType.YOUTUBE.value:
      if not os.path.exists(source.local_path):
        raise FileNotFoundError(f"Video file not found: {source.local_path}")
      return source

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
```

- [ ] **Step 4: Run preprocessing tests**

Run:

```bash
pytest tests/test_openai_video_preprocessor.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit preprocessing work**

Run:

```bash
git add llms_evaluation/openai_video_preprocessor.py tests/test_openai_video_preprocessor.py
git commit -m "feat: add OpenAI video preprocessing"
```

Expected:

```text
Commit created
```

## Task 6: OpenAI API Service And Detector

**Files:**
- Create: `llms_evaluation/openai_api_service.py`
- Create: `llms_evaluation/openai_detector.py`
- Create: `tests/test_openai_api_service.py`
- Create: `tests/test_openai_detector.py`

- [ ] **Step 1: Write failing OpenAI service tests**

Create `tests/test_openai_api_service.py`:

```python
import json
from unittest import mock

from llms_evaluation.openai_api_service import OpenAIAPIService
from prompts.prompt_generator import PromptConfig


def test_openai_service_requires_api_key(monkeypatch):
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)

  try:
    OpenAIAPIService()
  except ValueError as exc:
    assert "OPENAI_API_KEY" in str(exc)
  else:
    raise AssertionError("Expected missing API key error")


def test_parse_response_output_text(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")
  service = OpenAIAPIService(client=mock.Mock())
  response = mock.Mock(output_text=json.dumps({"features": []}))

  assert service.parse_json_response(response) == {"features": []}


def test_build_input_contains_text_and_images(monkeypatch, tmp_path):
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")
  frame = tmp_path / "frame.jpg"
  frame.write_bytes(b"image")
  service = OpenAIAPIService(client=mock.Mock())

  payload = service.build_input(
      PromptConfig(prompt="Question", system_instructions="System"),
      transcript="spoken words",
      frame_paths=[str(frame)],
      duration_seconds=15.0,
  )

  assert payload[0]["role"] == "system"
  assert payload[1]["role"] == "user"
  assert payload[1]["content"][0]["type"] == "input_text"
  assert payload[1]["content"][1]["type"] == "input_image"
```

- [ ] **Step 2: Run service tests to verify failure**

Run:

```bash
pytest tests/test_openai_api_service.py -q
```

Expected:

```text
FAIL because openai_api_service.py does not exist
```

- [ ] **Step 3: Implement OpenAI API service**

Create `llms_evaluation/openai_api_service.py`:

```python
"""OpenAI API service for video creative evaluation."""

import base64
import json
import os
from pathlib import Path

from openai import OpenAI

import models
from prompts.prompt_generator import PromptConfig


class OpenAIAPIService:
  """Wrapper around OpenAI APIs used by the OpenAI evaluation path."""

  def __init__(self, client=None):
    if not os.environ.get("OPENAI_API_KEY"):
      raise ValueError("OPENAI_API_KEY is required for llm_provider=OPENAI")
    self.client = client or OpenAI()

  def transcribe_audio(self, audio_path: str) -> str:
    """Transcribe audio file with OpenAI."""
    with open(audio_path, "rb") as audio_file:
      transcription = self.client.audio.transcriptions.create(
          model="gpt-4o-transcribe",
          file=audio_file,
      )
    return getattr(transcription, "text", "") or ""

  def evaluate_features(
      self,
      prompt_config: PromptConfig,
      preprocess_result: models.VideoPreprocessResult,
      model_name: str,
      schema: dict,
      frame_paths: list[str],
  ) -> dict:
    """Evaluate video features with sampled frames and transcript."""
    response = self.client.responses.create(
        model=model_name,
        input=self.build_input(
            prompt_config=prompt_config,
            transcript=preprocess_result.transcript,
            frame_paths=frame_paths,
            duration_seconds=preprocess_result.duration_seconds,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "abcd_feature_evaluation",
                "strict": True,
                "schema": schema,
            }
        },
    )
    return self.parse_json_response(response)

  def build_input(
      self,
      prompt_config: PromptConfig,
      transcript: str,
      frame_paths: list[str],
      duration_seconds: float,
  ) -> list[dict]:
    """Build Responses API input from prompt, transcript, and images."""
    user_content = [{
        "type": "input_text",
        "text": (
            f"Video duration seconds: {duration_seconds}\n"
            f"Transcript:\n{transcript or '[no transcript available]'}\n\n"
            f"{prompt_config.prompt}"
        ),
    }]
    for frame_path in frame_paths:
      user_content.append({
          "type": "input_image",
          "image_url": self._image_data_url(frame_path),
      })
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": prompt_config.system_instructions}],
        },
        {"role": "user", "content": user_content},
    ]

  def parse_json_response(self, response) -> dict:
    """Parse the JSON object returned by OpenAI structured outputs."""
    output_text = getattr(response, "output_text", "")
    return json.loads(output_text)

  def _image_data_url(self, frame_path: str) -> str:
    mime_type = "image/jpeg"
    encoded = base64.b64encode(Path(frame_path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
```

- [ ] **Step 4: Write detector tests**

Create `tests/test_openai_detector.py`:

```python
from dataclasses import dataclass

import models
from llms_evaluation.openai_detector import OpenAIDetector


@dataclass
class ConfigMock:
  openai_model: str = "gpt-5.4-mini"
  openai_transcription_model: str = "gpt-4o-transcribe"
  brand_name: str = "Chewy"
  brand_variations: list[str] = None
  branded_products: list[str] = None
  branded_products_categories: list[str] = None
  branded_call_to_actions: list[str] = None


class OpenAIServiceMock:
  def evaluate_features(self, prompt_config, preprocess_result, model_name, schema, frame_paths):
    return {
        "features": [{
            "id": "a_supers",
            "name": "Supers",
            "category": "LONG_FORM_ABCD",
            "sub_category": "ATTRACT",
            "video_segment": "FULL_VIDEO",
            "evaluation_criteria": "Any supers are present.",
            "detected": True,
            "confidence_score": 0.9,
            "rationale": "Text is visible.",
            "evidence": "Frame 1 shows text.",
            "strengths": "Clear text.",
            "weaknesses": "Sampled frames only.",
        }]
    }


def test_openai_detector_returns_features(tmp_path):
  config = ConfigMock(
      brand_variations=["Chewy"],
      branded_products=["Chewy Pharmacy"],
      branded_products_categories=["pet supplies"],
      branded_call_to_actions=["shop now"],
  )
  source = models.VideoSource("ad.mp4", "ad.mp4", "LOCAL")
  preprocess_result = models.VideoPreprocessResult(
      source=source,
      duration_seconds=15.0,
      full_video_frames=[],
      first_5_seconds_frames=[],
      audio_path=None,
      transcript="Shop now at Chewy.",
      transcript_available=True,
  )
  feature = models.VideoFeature(
      id="a_supers",
      name="Supers",
      category=models.VideoFeatureCategory.LONG_FORM_ABCD,
      sub_category=models.VideoFeatureSubCategory.ATTRACT,
      video_segment=models.VideoSegment.FULL_VIDEO,
      evaluation_criteria="Any supers are present.",
      prompt_template="Are there supers?",
      extra_instructions=[],
      evaluation_method=models.EvaluationMethod.LLMS,
      evaluation_function="",
      include_in_evaluation=True,
      group_by=models.VideoSegment.FULL_VIDEO,
  )

  result = OpenAIDetector(OpenAIServiceMock()).evaluate_features(
      config=config,
      preprocess_result=preprocess_result,
      feature_configs=[feature],
  )

  assert result[0]["id"] == "a_supers"
  assert result[0]["detected"] is True
```

- [ ] **Step 5: Implement OpenAI detector**

Create `llms_evaluation/openai_detector.py`:

```python
"""OpenAI detector facade for ABCD feature evaluation."""

import models
from prompts.prompt_generator import prompt_generator


class OpenAIDetector:
  """Evaluates ABCD features through OpenAI."""

  def __init__(self, openai_service):
    self.openai_service = openai_service

  def evaluate_features(
      self,
      config,
      preprocess_result: models.VideoPreprocessResult,
      feature_configs: list[models.VideoFeature],
  ) -> list[dict]:
    """Evaluate features and return OpenAI feature dictionaries."""
    prompt_config = prompt_generator.get_abcds_prompt_config(
        feature_configs,
        config,
    )
    frame_paths = self._select_frames(preprocess_result, feature_configs)
    response = self.openai_service.evaluate_features(
        prompt_config=prompt_config,
        preprocess_result=preprocess_result,
        model_name=config.openai_model,
        schema=models.OPENAI_VIDEO_RESPONSE_SCHEMA,
        frame_paths=frame_paths,
    )
    return response.get("features", [])

  def _select_frames(
      self,
      preprocess_result: models.VideoPreprocessResult,
      feature_configs: list[models.VideoFeature],
  ) -> list[str]:
    if all(
        feature.video_segment == models.VideoSegment.FIRST_5_SECS_VIDEO
        for feature in feature_configs
    ):
      return preprocess_result.first_5_seconds_frames
    return preprocess_result.full_video_frames
```

- [ ] **Step 6: Run OpenAI service tests**

Run:

```bash
pytest tests/test_openai_api_service.py tests/test_openai_detector.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 7: Commit OpenAI service work**

Run:

```bash
git add llms_evaluation/openai_api_service.py llms_evaluation/openai_detector.py tests/test_openai_api_service.py tests/test_openai_detector.py
git commit -m "feat: add OpenAI evaluation service"
```

Expected:

```text
Commit created
```

## Task 7: Route OpenAI Evaluation Through The Existing Pipeline

**Files:**
- Modify: `evaluation_services/video_evaluation_service.py`
- Modify: `main.py`
- Create: `tests/test_openai_pipeline_routing.py`

- [ ] **Step 1: Write failing routing test**

Create `tests/test_openai_pipeline_routing.py`:

```python
from unittest import mock

import models
from evaluation_services.video_evaluation_service import VideoEvaluationService


class ConfigMock:
  llm_provider_type = models.LLMProviderType.OPENAI
  extract_brand_metadata = False
  brand_name = "Chewy"
  brand_variations = ["Chewy"]
  branded_products = ["Chewy Pharmacy"]
  branded_products_categories = ["pet supplies"]
  branded_call_to_actions = ["shop now"]
  openai_model = "gpt-5.4-mini"


def test_openai_route_uses_preprocess_result():
  service = VideoEvaluationService()
  source = models.VideoSource("ad.mp4", "ad.mp4", "LOCAL")
  preprocess_result = models.VideoPreprocessResult(
      source=source,
      duration_seconds=15.0,
      full_video_frames=[],
      first_5_seconds_frames=[],
      audio_path=None,
      transcript="Shop Chewy.",
      transcript_available=True,
  )

  with mock.patch(
      "evaluation_services.video_evaluation_service.openai_detector.OpenAIDetector"
  ) as detector_cls:
    detector = detector_cls.return_value
    detector.evaluate_features.return_value = [{
        "id": "a_supers",
        "detected": True,
        "confidence_score": 0.8,
        "rationale": "Text visible.",
        "evidence": "Frame.",
        "strengths": "Clear.",
        "weaknesses": "Sampled.",
    }]

    result = service.evaluate_features(
        config=ConfigMock(),
        video_uri="ad.mp4",
        features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
        preprocess_result=preprocess_result,
    )

  assert len(result) == 1
  assert result[0].feature.id == "a_supers"
  assert result[0].detected is True
```

- [ ] **Step 2: Run routing test to verify failure**

Run:

```bash
pytest tests/test_openai_pipeline_routing.py -q
```

Expected:

```text
FAIL because evaluate_features does not accept preprocess_result and no OpenAI route exists
```

- [ ] **Step 3: Modify evaluation service**

In `evaluation_services/video_evaluation_service.py`, import:

```python
from llms_evaluation import openai_api_service
from llms_evaluation import openai_detector
```

Change signature:

```python
def evaluate_features(
    self,
    config: configuration.Configuration,
    video_uri: str,
    features_category: models.VideoFeatureCategory,
    preprocess_result: models.VideoPreprocessResult | None = None,
):
```

Add near top after metadata extraction block:

```python
if config.llm_provider_type == models.LLMProviderType.OPENAI:
  if preprocess_result is None:
    raise ValueError("preprocess_result is required for OpenAI evaluation")
  return self._evaluate_features_with_openai(
      config=config,
      preprocess_result=preprocess_result,
      features_category=features_category,
  )
```

Add helper method:

```python
def _evaluate_features_with_openai(
    self,
    config: configuration.Configuration,
    preprocess_result: models.VideoPreprocessResult,
    features_category: models.VideoFeatureCategory,
) -> list[models.FeatureEvaluation]:
  feature_groups = (
      feature_configs_handler.features_configs_handler
      .get_features_by_category_by_group_config(features_category)
  )
  detector = openai_detector.OpenAIDetector(
      openai_api_service.OpenAIAPIService()
  )
  feature_evaluations = []
  for feature_configs in feature_groups.values():
    evaluated_features = detector.evaluate_features(
        config=config,
        preprocess_result=preprocess_result,
        feature_configs=feature_configs,
    )
    for evaluated_feature in evaluated_features:
      feature = (
          feature_configs_handler.features_configs_handler.get_feature_by_id(
              evaluated_feature.get("id")
          )
      )
      if feature:
        feature_evaluations.append(
            models.FeatureEvaluation(
                feature=feature,
                detected=evaluated_feature.get("detected"),
                confidence_score=evaluated_feature.get("confidence_score"),
                rationale=evaluated_feature.get("rationale"),
                evidence=evaluated_feature.get("evidence"),
                strengths=evaluated_feature.get("strengths"),
                weaknesses=evaluated_feature.get("weaknesses"),
            )
        )
  return sorted(
      feature_evaluations,
      key=lambda feature_eval: (
          feature_eval.feature.category.value,
          feature_eval.feature.id,
      ),
      reverse=False,
  )
```

- [ ] **Step 4: Modify main OpenAI path**

In `main.py`, import:

```python
from creative_providers import local_creative_provider
from llms_evaluation import openai_api_service
from llms_evaluation import openai_video_preprocessor
```

Add helper before `execute_abcd_assessment_for_videos`:

```python
def get_video_sources(config: Configuration) -> list[models.VideoSource]:
  """Get video sources for OpenAI or legacy provider execution."""
  if config.llm_provider_type == models.LLMProviderType.OPENAI:
    return local_creative_provider.LocalCreativeProvider().get_creative_sources(
        config
    )
  creative_provider = creative_provider_registry.provider_factory.get_provider(
      config.creative_provider_type.value
  )
  return [
      models.VideoSource(
          original_uri=video_uri,
          local_path=video_uri,
          source_type=config.creative_provider_type.value,
      )
      for video_uri in creative_provider.get_creative_uris(config)
  ]
```

Inside `execute_abcd_assessment_for_videos`, replace current provider and loop setup with:

```python
video_sources = get_video_sources(config)
openai_service = None
preprocessor = None
if config.llm_provider_type == models.LLMProviderType.OPENAI:
  openai_service = openai_api_service.OpenAIAPIService()
  preprocessor = openai_video_preprocessor.VideoPreprocessor(
      cache_dir=config.cache_dir,
      max_frames=config.max_frames,
      frame_sample_rate=config.frame_sample_rate,
      openai_service=openai_service,
  )

video_assessments = []
for source in video_sources:
  video_uri = source.local_path or source.original_uri
  preprocess_result = (
      preprocessor.preprocess(source)
      if preprocessor
      else None
  )
```

Pass `preprocess_result=preprocess_result` into both `evaluate_features` calls.

Append each `video_assessment` to `video_assessments`, and at end of function return `video_assessments`.

- [ ] **Step 5: Run routing tests**

Run:

```bash
pytest tests/test_openai_pipeline_routing.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 6: Commit routing work**

Run:

```bash
git add evaluation_services/video_evaluation_service.py main.py tests/test_openai_pipeline_routing.py
git commit -m "feat: route OpenAI evaluations"
```

Expected:

```text
Commit created
```

## Task 8: JSON Assessment Output

**Files:**
- Modify: `helpers/generic_helpers.py`
- Modify: `main.py`
- Create: `tests/test_assessment_json_output.py`

- [ ] **Step 1: Write failing JSON writer tests**

Create `tests/test_assessment_json_output.py`:

```python
import json

import models
from helpers.generic_helpers import write_assessments_json


def test_write_assessments_json(tmp_path):
  feature = models.VideoFeature(
      id="a_supers",
      name="Supers",
      category=models.VideoFeatureCategory.LONG_FORM_ABCD,
      sub_category=models.VideoFeatureSubCategory.ATTRACT,
      video_segment=models.VideoSegment.FULL_VIDEO,
      evaluation_criteria="Text overlays are present.",
      prompt_template="Are there supers?",
      extra_instructions=[],
      evaluation_method=models.EvaluationMethod.LLMS,
      evaluation_function="",
      include_in_evaluation=True,
      group_by=models.VideoSegment.FULL_VIDEO,
  )
  assessment = models.VideoAssessment(
      brand_name="Chewy",
      video_uri="sample_videos/google/videos/ad.mp4",
      long_form_abcd_evaluated_features=[
          models.FeatureEvaluation(
              feature=feature,
              detected=True,
              confidence_score=0.9,
              rationale="Text visible.",
              evidence="Frame 1.",
              strengths="Clear.",
              weaknesses="Sampled.",
          )
      ],
      shorts_evaluated_features=[],
      config=object(),
  )
  output = tmp_path / "results.json"

  write_assessments_json([assessment], str(output))

  data = json.loads(output.read_text())
  assert data["assessments"][0]["brand_name"] == "Chewy"
  assert data["assessments"][0]["long_form_abcd_evaluated_features"][0]["id"] == "a_supers"
```

- [ ] **Step 2: Run JSON writer test to verify failure**

Run:

```bash
pytest tests/test_assessment_json_output.py -q
```

Expected:

```text
FAIL because write_assessments_json does not exist
```

- [ ] **Step 3: Implement JSON serialization**

In `helpers/generic_helpers.py`, add imports:

```python
from pathlib import Path
```

Add functions near `calculate_score`:

```python
def feature_evaluation_to_dict(
    eval_feature: models.FeatureEvaluation,
) -> dict:
  """Convert a FeatureEvaluation to JSON-safe dict."""
  return {
      "id": eval_feature.feature.id,
      "name": eval_feature.feature.name,
      "category": eval_feature.feature.category.value,
      "sub_category": eval_feature.feature.sub_category.value,
      "video_segment": eval_feature.feature.video_segment.value,
      "evaluation_criteria": eval_feature.feature.evaluation_criteria,
      "detected": eval_feature.detected,
      "confidence_score": eval_feature.confidence_score,
      "rationale": eval_feature.rationale,
      "evidence": eval_feature.evidence,
      "strengths": eval_feature.strengths,
      "weaknesses": eval_feature.weaknesses,
  }


def video_assessment_to_dict(
    video_assessment: models.VideoAssessment,
) -> dict:
  """Convert a VideoAssessment to JSON-safe dict."""
  return {
      "brand_name": video_assessment.brand_name,
      "video_uri": video_assessment.video_uri,
      "long_form_abcd_evaluated_features": [
          feature_evaluation_to_dict(eval_feature)
          for eval_feature in video_assessment.long_form_abcd_evaluated_features
      ],
      "shorts_evaluated_features": [
          feature_evaluation_to_dict(eval_feature)
          for eval_feature in video_assessment.shorts_evaluated_features
      ],
  }


def write_assessments_json(
    video_assessments: list[models.VideoAssessment],
    assessment_file: str,
) -> None:
  """Write video assessments to a JSON file."""
  output_path = Path(assessment_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "assessments": [
          video_assessment_to_dict(video_assessment)
          for video_assessment in video_assessments
      ]
  }
  output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Call JSON writer from main**

In `main.py`, after `execute_abcd_assessment_for_videos(config)`:

```python
video_assessments = execute_abcd_assessment_for_videos(config)
if config.assessment_file and video_assessments:
  generic_helpers.write_assessments_json(
      video_assessments,
      config.assessment_file,
  )
```

Ensure `execute_abcd_assessment_for_videos` returns a list in both OpenAI and
legacy paths.

- [ ] **Step 5: Run JSON tests**

Run:

```bash
pytest tests/test_assessment_json_output.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 6: Commit JSON output work**

Run:

```bash
git add helpers/generic_helpers.py main.py tests/test_assessment_json_output.py
git commit -m "feat: write assessment JSON output"
```

Expected:

```text
Commit created
```

## Task 9: README CLI Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add CLI quickstart section**

Add this section after the current "Where to start?" section:

```markdown
## Local OpenAI CLI Batch Execution

The CLI can evaluate local videos and YouTube URLs with OpenAI without Google
Cloud Storage, Video Intelligence, Vertex AI, BigQuery, or Knowledge Graph.

Set your OpenAI API key in the shell:

```bash
export OPENAI_API_KEY="sk-..."
```

Run a local sample:

```bash
python main.py \
  --llm_provider OPENAI \
  --creative_provider_type LOCAL,YOUTUBE \
  --video_uris "sample_videos/google/videos/Youtube_sWaCTG60wPA_Never_run_out_of_the_food_they_love_Chewy.mp4" \
  --brand_name "Chewy" \
  --brand_variations "Chewy,chewy.com" \
  --branded_products "Chewy Pharmacy,Autoship" \
  --branded_products_categories "pet food,pet medication,pet supplies" \
  --branded_call_to_actions "shop now,save now,order now" \
  --run_long_form_abcd \
  --run_shorts \
  --assessment_file outputs/abcd_results.json
```

Private sample videos should live outside git. This repository uses a local
`sample_videos` symlink for manual testing.
```

- [ ] **Step 2: Run markdown-free verification**

Run:

```bash
python main.py --help
```

Expected:

```text
Help text includes --llm_provider, --openai_model, --cache_dir, --max_frames, and --frame_sample_rate
```

- [ ] **Step 3: Commit docs**

Run:

```bash
git add README.md
git commit -m "docs: add OpenAI CLI quickstart"
```

Expected:

```text
Commit created
```

## Task 10: Verification And Sample Run

**Files:**
- No planned source file changes.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest \
  tests/test_abcd_parameters.py \
  tests/test_openai_cli_config.py \
  tests/test_local_creative_provider.py \
  tests/test_openai_video_preprocessor.py \
  tests/test_openai_api_service.py \
  tests/test_openai_detector.py \
  tests/test_openai_pipeline_routing.py \
  tests/test_assessment_json_output.py \
  -q
```

Expected:

```text
PASS
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected:

```text
PASS
```

- [ ] **Step 3: Verify API key presence without printing it**

Run:

```bash
zsh -lc 'test -n "$OPENAI_API_KEY" && echo OPENAI_API_KEY_PRESENT'
```

Expected:

```text
OPENAI_API_KEY_PRESENT
```

- [ ] **Step 4: Run one local sample video**

Run:

```bash
zsh -lc 'python main.py \
  --llm_provider OPENAI \
  --creative_provider_type LOCAL,YOUTUBE \
  --video_uris "sample_videos/google/videos/Youtube_sWaCTG60wPA_Never_run_out_of_the_food_they_love_Chewy.mp4" \
  --brand_name "Chewy" \
  --brand_variations "Chewy,chewy.com" \
  --branded_products "Chewy Pharmacy,Autoship" \
  --branded_products_categories "pet food,pet medication,pet supplies" \
  --branded_call_to_actions "shop now,save now,order now" \
  --run_long_form_abcd \
  --assessment_file outputs/sample_openai_results.json'
```

Expected:

```text
Command exits 0 and outputs/sample_openai_results.json contains one assessment
```

- [ ] **Step 5: Inspect JSON shape**

Run:

```bash
python -m json.tool outputs/sample_openai_results.json >/tmp/abcd_openai_results_pretty.json
```

Expected:

```text
Command exits 0
```

- [ ] **Step 6: Final commit if verification changes docs or tests**

Run only if files changed during verification:

```bash
git status --short
git add README.md tests docs
git commit -m "test: verify OpenAI CLI workflow"
```

Expected:

```text
Commit created only when tracked files changed
```

## Self-Review

Spec coverage:

- Fork remote setup is covered by Task 1.
- OpenAI key handling is covered by Tasks 3, 6, and 10.
- Local and YouTube inputs are covered by Tasks 4, 5, 7, and 10.
- CLI batch execution is covered by Tasks 3, 7, 8, 9, and 10.
- Default models are covered by Tasks 3 and 6.
- Sample videos are covered by Tasks 5, 9, and 10.
- JSON-only output is covered by Task 8.
- Keeping Google/notebook behavior is covered by Task 3 defaults and Task 7 routing.

Type consistency:

- Provider enum values use `models.CreativeProviderType`.
- LLM provider values use `models.LLMProviderType`.
- Source objects use `models.VideoSource`.
- Preprocessing result uses `models.VideoPreprocessResult`.
- JSON writer consumes `models.VideoAssessment`.

Execution notes:

- Unit tests mock OpenAI and do not require network.
- Manual sample run uses `zsh -lc` so `.zshrc` can expose `OPENAI_API_KEY`.
- `sample_videos`, `.cache/`, and `outputs/` are gitignored.

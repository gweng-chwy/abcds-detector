# OpenAI Evidence Extraction Upgrade Design

Date: 2026-06-19

## Goal

Improve the OpenAI local/YouTube evaluation path by making video and audio
evidence more representative, timestamped, and cacheable without changing the
feature inventory or feature definitions.

## Non-Goals

- Do not edit `docs/feature_inventory/*`.
- Do not edit `features_repository/*` unless a later implementation task proves
  it is required and the other active session has finished.
- Do not change feature meanings, feature IDs, or inventory docs.
- Do not require Google Cloud services for the OpenAI path.
- Do not add heavyweight external services for OCR, shot detection, or speech
  analysis in this pass.

## Current Problem

The current OpenAI preprocessor deletes cached frames and audio on every run.
Full-video frames are extracted with `ffmpeg -vf fps=<rate> -frames:v
<max_frames>`, which samples from the start of the video until the frame cap is
hit. For a default 24-frame run at 1 FPS, a longer video gets evidence from
roughly the first 24 seconds rather than the full runtime.

The evaluator receives transcript text and images, but it does not receive
per-frame timestamps. Speech features that ask about the first 5 seconds use
the full transcript unless the evaluation path provides a first-5 evidence pack.
Prompt templates that ask for exact timestamps, temporal density, text/audio
synchrony, or pacing therefore force the model to infer more than the evidence
can support.

## Design Summary

Add manifest-based artifact reuse and separate evidence packs for full-video
and first-5-second evaluation.

Each video gets a cache directory:

```text
.cache/abcds-detector/<cache_key>/
  preprocess_manifest.json
  source.mp4
  frames/full/frame_0001.jpg
  frames/first_5s/frame_0001.jpg
  audio.mp3
  audio_first_5s.mp3
  transcript.txt
  transcript_first_5s.txt
```

The manifest records source identity, extraction settings, artifact paths,
frame timestamps, audio settings, transcription model, transcript availability,
and an extraction strategy version. When the manifest matches the current input
and every referenced artifact exists, preprocessing returns cached evidence
without rerunning `ffmpeg`, `ffprobe`, `yt-dlp`, or OpenAI transcription.

## CLI And Configuration

Add one cache control flag:

```bash
--refresh_cache
```

Default behavior is cache reuse. `--refresh_cache` forces a rebuild for the
current video cache entry. It should not delete unrelated video cache
directories.

Existing extraction knobs remain:

- `--cache_dir`
- `--max_frames`
- `--frame_sample_rate`
- `--openai_transcription_model`

The implementation may add an internal first-5 frame rate derived from existing
settings instead of adding a new CLI flag. This keeps the public surface small.

## Source Fingerprint

Use a cheap fingerprint so private local videos are not read in full.

For local videos:

- original URI
- local path
- file size
- `mtime_ns`

For YouTube videos:

- original URL
- downloaded local path
- downloaded file size
- downloaded `mtime_ns`

The fingerprint is a staleness check, not a security hash. If users overwrite a
file while preserving size and mtime, they can use `--refresh_cache`.

## Cache Validation

A cache entry is valid only when all of these match:

- manifest schema version
- extraction strategy version
- source fingerprint
- `max_frames`
- `frame_sample_rate`
- audio extraction settings
- transcription model
- expected artifact paths exist
- frame metadata count matches existing frame files

If the manifest is missing, malformed, stale, incomplete, or references missing
files, rebuild the stale artifacts. If `--refresh_cache` is true, rebuild the
current video's cache entry.

## Frame Extraction

Full-video visual evidence should cover the full runtime.

The preprocessor should compute target timestamps across the duration and ask
ffmpeg for frames near those timestamps. The default `max_frames=24` should
produce at most 24 full-video frames spread from early to late video content.

First-5 evidence should be denser. Extract frames from `0` through
`min(duration, 5)` so first-5 logo, product, people, supers, and hook features
have better coverage. The implementation can cap first-5 frames with
`max_frames` to avoid surprising payload growth.

Every extracted frame gets metadata:

```json
{
  "path": ".cache/abcds-detector/<key>/frames/full/frame_0001.jpg",
  "timestamp_seconds": 3.25,
  "segment": "FULL_VIDEO"
}
```

The OpenAI prompt should include a compact frame index before the image payload:

```text
Frame evidence:
- full/frame_0001.jpg at 0.00s
- full/frame_0002.jpg at 3.25s
```

This does not make timestamps perfect, but it grounds model rationales in the
sampled evidence.

## Audio Extraction And Transcription

Extract two audio artifacts:

- `audio.mp3`: full video audio.
- `audio_first_5s.mp3`: first 5 seconds of audio.

Both use the current mono 16 kHz MP3 settings unless tests show a reason to
change them.

Transcribe each available audio artifact separately:

- full transcript -> `transcript.txt`
- first-5 transcript -> `transcript_first_5s.txt`

If transcription fails, record transcript unavailable in the manifest. Do not
retry failed transcription on every run unless the cache is refreshed or the
manifest becomes stale. Audio extraction failure should not fail preprocessing;
visual evidence can still be evaluated.

If the installed OpenAI SDK supports timecoded transcription options in the
current dependency version, the implementation can store segment timestamps in
the manifest. If not, plain transcript text is sufficient for this pass.

## Evidence Selection

Feature groups already run by `group_by`. Preserve that flow.

When every feature in the evaluated group targets `FIRST_5_SECS_VIDEO`, the
OpenAI detector should send:

- first-5 frames
- first-5 transcript
- first-5 frame metadata

Otherwise, send:

- full-video frames
- full transcript
- full-video frame metadata

This keeps first-5 speech checks from seeing unrelated later transcript text and
keeps full-video checks from losing late visual evidence.

## OpenAI Payload

Update the input builder to include:

- video duration
- transcript availability
- selected transcript
- frame evidence list with timestamps
- existing prompt text
- selected image payloads

The strict feature response schema can remain unchanged for this pass. The
feature inventory contains richer Shorts metric templates, but existing
downstream code only persists generic feature fields. This upgrade improves the
evidence the model sees without expanding the result schema.

## Error Handling

- Missing local video: fail before subprocess work.
- Missing or empty full-video frames: fail preprocessing.
- Missing first-5 frames: fail preprocessing, because first-5 feature groups
  depend on them.
- Missing audio stream: continue with transcript unavailable.
- Transcription failure: continue with transcript unavailable and cache that
  state.
- Malformed manifest: ignore and rebuild.
- Partial cache: rebuild missing or stale artifacts.
- YouTube download failure: fail that video with downloader error.

## Testing Strategy

Use TDD.

Focused unit tests should cover:

- cache hit skips `ffmpeg`, `ffprobe`, `yt-dlp`, and transcription;
- stale source fingerprint rebuilds;
- stale extraction settings rebuild;
- `refresh_cache=True` rebuilds;
- malformed manifest rebuilds;
- full-video frame extraction uses target timestamps across the duration;
- first-5 audio is extracted and transcribed separately;
- first-5 feature groups receive first-5 transcript and frames;
- full-video groups receive full transcript and uniformly sampled frames;
- OpenAI prompt text includes frame timestamp labels;
- transcription failure is cached as unavailable;
- no tests or implementation require touching `docs/feature_inventory/*`.

Manual verification can use private videos under `sample_videos/`, but those
assets and generated cache outputs must remain untracked.

## Post-Implementation Validation Run

After detector implementation and unit verification, run a real OpenAI pipeline
sample over local private videos.

The validation script or command sequence should:

- discover videos under `sample_videos/<platform>/**`;
- treat the first directory under `sample_videos` as the platform name;
- randomly choose up to 2 videos per platform with a fixed seed;
- use all available videos when a platform has fewer than 2 videos;
- run both long-form ABCD and Shorts features by default;
- write outputs under `outputs/openai_validation_sample/`;
- reuse cache by default and avoid `--refresh_cache` unless specifically
  validating rebuild behavior.

The user's `OPENAI_API_KEY` is available from `.zshrc`. The validation command
should run in a shell environment that loads that key. If the key is not present
at runtime, skip the live API run and report that validation is blocked by local
environment setup.

Generated outputs, private videos, extracted frames, audio, transcripts, and
cache artifacts must remain untracked.

## Evidence Review Notebook

Create a dedicated notebook folder after the validation run:

```text
notebooks/20260619_openai_evidence_review/
  README.md
  openai_evidence_review.ipynb
  figures/
```

The notebook should load:

- the validation assessment JSON;
- each selected video's preprocess manifest;
- extracted frame paths and frame timestamps;
- transcript text or timecoded transcript segments when available.

For each selected video, produce slide-like 16:9 figures suitable for the
default Keynote widescreen ratio. Each figure should place evidence on the left
and detected feature status on the right:

- left side: extracted frame and transcript snippet around that frame time;
- right side: feature checklist with checked state when `detected` is true;
- separate or clearly group long-form and Shorts features;
- use the largest practical font size that fits the 16:9 canvas;
- follow `DESIGN.md` typography: `figmaSans` when available, then Inter,
  Geist, system sans; use `figmaMono` or monospace fallback only for captions
  and metadata;
- use monochrome styling unless a later review explicitly asks for color.

If timecoded transcript segments are available, show transcript text within a
small window around the frame timestamp, such as plus or minus 2.5 seconds. If
only plain transcript text is available, show a compact transcript excerpt and
label it as not timecoded.

Notebook acceptance criteria:

- creates at least one generated 16:9 figure per sampled video;
- frame content is readable;
- checklist text is legible at Keynote ratio;
- notebook paths are relative to its folder or repository root;
- generated figure files live under the notebook folder;
- generated outputs remain local and untracked unless explicitly requested.

## Implementation Boundaries

Expected files to modify:

- `models.py`: extend preprocess result data to carry frame metadata and
  selected transcript information.
- `configuration.py`: add `refresh_cache`.
- `utils.py`: add CLI flag and config mapping.
- `llms_evaluation/openai_video_preprocessor.py`: manifest cache, uniform frame
  extraction, first-5 audio/transcript.
- `llms_evaluation/openai_detector.py`: select matching frame/transcript
  evidence by feature group.
- `llms_evaluation/openai_api_service.py`: include frame timestamp labels in
  prompt input.
- `notebooks/20260619_openai_evidence_review/`: evidence review notebook and
  README created after implementation validation.
- Tests under `tests/`.

Files intentionally not touched:

- `docs/feature_inventory/*`
- `features_repository/*` unless explicitly unblocked later.

## Acceptance Criteria

- Existing OpenAI CLI defaults remain compatible.
- Re-running preprocessing for an unchanged video reuses cached frames, audio,
  and transcripts.
- `--refresh_cache` forces rebuild for the current video.
- Full-video frames are spread across the runtime instead of stopping after the
  first `max_frames / frame_sample_rate` seconds.
- First-5 feature groups do not receive later transcript text.
- OpenAI prompt text includes frame timestamps.
- Unit tests prove cache hit, cache invalidation, evidence selection, and prompt
  timestamp behavior.
- Post-implementation validation samples up to 2 videos per platform from
  `sample_videos/` and writes JSON/CSV under
  `outputs/openai_validation_sample/`.
- Evidence review notebook generates 16:9 frame/transcript/checklist figures
  under `notebooks/20260619_openai_evidence_review/`.
- No changes are made to `docs/feature_inventory/*`.

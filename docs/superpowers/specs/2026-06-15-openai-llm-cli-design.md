# OpenAI LLM-Only CLI Batch Design

Date: 2026-06-15

## Goal

Add an OpenAI-backed, LLM-only batch CLI path to ABCDs Detector. The new path
must evaluate local videos and YouTube URLs without requiring Google Cloud
Storage, Google Video Intelligence, Vertex AI, BigQuery, or Knowledge Graph.

The existing notebook and Google-backed code should keep working.

## User Outcomes

- Run ABCD/Shorts evaluation from a terminal over many videos.
- Use an `OPENAI_API_KEY` environment variable instead of Google Cloud
  credentials.
- Provide local video files or YouTube URLs as inputs.
- Receive a structured JSON output file.
- Use sample videos from `sample_videos/`, a symlink to the user's local
  creative asset directory.

## Non-Goals

- No production UI.
- No BigQuery output in the MVP.
- No Google Knowledge Graph replacement service.
- No deterministic shot detector in the first pass.
- No committing private sample videos to git.
- No full rewrite of feature definitions or scoring models.

## Inputs

Required inputs:

- `--video_uris`: comma-delimited local paths or YouTube URLs.
- `OPENAI_API_KEY`: environment variable for OpenAI auth.

Recommended brand inputs:

- `--brand_name`
- `--brand_variations`
- `--branded_products`
- `--branded_products_categories`
- `--branded_call_to_actions`

If brand fields are missing, `--extract_brand_metadata` lets OpenAI infer them
from video evidence. Explicit user-provided values override inferred values.

Optional execution inputs:

- `--llm_provider OPENAI`
- `--openai_model gpt-5.4-mini`
- `--openai_transcription_model gpt-4o-transcribe`
- `--run_long_form_abcd`
- `--run_shorts`
- `--features_to_evaluate`
- `--assessment_file outputs/abcd_results.json`
- `--cache_dir .cache/abcds-detector`
- `--max_frames`
- `--frame_sample_rate`

Example:

```bash
OPENAI_API_KEY="..." python main.py \
  --llm_provider OPENAI \
  --creative_provider_type LOCAL,YOUTUBE \
  --video_uris "sample_videos/google/videos/Youtube_sWaCTG60wPA_Never_run_out_of_the_food_they_love_Chewy.mp4,https://www.youtube.com/watch?v=9A_aMZO0iSw" \
  --brand_name "Chewy" \
  --brand_variations "Chewy,chewy.com" \
  --branded_products "Chewy Pharmacy,Autoship" \
  --branded_products_categories "pet food,pet medication,pet supplies" \
  --branded_call_to_actions "shop now,save now,order now" \
  --run_long_form_abcd \
  --run_shorts \
  --assessment_file outputs/abcd_results.json
```

## Outputs

The MVP writes the existing JSON assessment output and a sibling CSV summary.
The JSON output contains one assessment per video and keeps the existing shape.

Top-level shape:

```json
{
  "assessments": [
    {
      "brand_name": "Chewy",
      "video_uri": "sample_videos/google/videos/example.mp4",
      "source_type": "LOCAL",
      "long_form_abcd_evaluated_features": [],
      "shorts_evaluated_features": [],
      "metadata": {
        "duration_seconds": 15.0,
        "sampled_frames": 24,
        "transcript_available": true
      }
    }
  ]
}
```

Each feature result keeps the existing project shape:

```json
{
  "id": "b_brand_visuals",
  "name": "Brand Visuals",
  "category": "LONG_FORM_ABCD",
  "sub_category": "BRAND",
  "video_segment": "FULL_VIDEO",
  "detected": true,
  "confidence_score": 0.86,
  "rationale": "Branding is visible in sampled frames.",
  "evidence": "Chewy logo appears on the end card around 0:13.",
  "strengths": "Logo and brand colors are clear.",
  "weaknesses": "Exact timing depends on frame sampling."
}
```

The example above is shape-only. Full output includes all enabled features.

When `--assessment_file outputs/abcd_results.json` is set, the CLI also writes
`outputs/abcd_results.csv`. The CSV contains one row per video and one column
per evaluated feature ID:

```csv
video_uri,brand_name,a_dynamic_start,a_quick_pacing,shorts_human_voice
sample_videos/google/videos/example.mp4,Chewy,true,false,true
```

If a feature was not evaluated for a video, its cell is blank. The CSV is a
detected-summary sidecar only; JSON remains the source of full rationale,
evidence, confidence, strengths, and weaknesses.

## Supported Features

The MVP attempts all existing configured features through LLM evaluation.

Long-form ABCD:

- `a_dynamic_start`
- `a_quick_pacing`
- `a_quick_pacing_1st_5_secs`
- `a_supers`
- `a_supers_with_audio`
- `b_brand_mention_speech`
- `b_brand_mention_speech_1st_5_secs`
- `b_brand_visuals`
- `b_brand_visuals_1st_5_secs`
- `b_product_mention_speech`
- `b_product_mention_speech_1st_5_secs`
- `b_product_mention_text`
- `b_product_mention_text_1st_5_secs`
- `b_product_visuals`
- `b_product_visuals_1st_5_secs`
- `c_overall_pacing`
- `c_presence_of_people`
- `c_presence_of_people_1st_5_secs`
- `c_visible_face`
- `c_visible_face_close_up`
- `d_audio_speech_early_1st_5_secs`
- `d_call_to_action_speech`
- `d_call_to_action_text`

Shorts:

- `tight_framing_index`
- `shorts_human_voice`
- `shorts_direct_to_camera`
- `shorts_has_supers`
- `shorts_product_closeup`
- `shorts_product_extreme_closeup`
- `shorts_product_context_index`
- `shorts_casual_language`
- `shorts_humor_index`
- `character_driven`
- `shorts_audio_cta`
- `special_offer_speech`
- `shorts_production_style_index`
- `shorts_sfv_adaptation_high`
- `shorts_emoji_usage`
- `shorts_personal_character_talk`
- `shorts_native_brand_context`
- `shorts_personal_character_type`
- `shorts_product_context`
- `shorts_video_format`

Known limitation: shot-count and pacing features are estimated from sampled
frames in the LLM-only MVP. A later hybrid pass can add deterministic local shot
detection.

## Architecture

### Repository Setup

Before implementation, point local development at the user's fork while keeping
the original Google repository as upstream:

```bash
git remote rename origin upstream
git remote add origin git@github.com:gweng-chwy/abcds-detector.git
git checkout -b openai-llm-cli
```

Expected remotes after setup:

```text
origin   git@github.com:gweng-chwy/abcds-detector.git
upstream git@github.com:google-marketing-solutions/abcds-detector.git
```

Implementation commits should land on `openai-llm-cli` and push to the fork.

### CLI Configuration

Extend existing argparse config rather than replacing it. New args should map
into `Configuration` fields.

New concepts:

- `llm_provider`: `GEMINI` or `OPENAI`
- `openai_model`
- `openai_transcription_model`
- `cache_dir`
- `max_frames`
- `frame_sample_rate`

Default model choices:

- Evaluation model: `gpt-5.4-mini`
- Transcription model: `gpt-4o-transcribe`

Rationale: OpenAI model docs recommend GPT-5.5 when capability matters most and
GPT-5.4 mini/nano for lower-cost, lower-latency workloads. Batch creative
evaluation should default to the cost/latency-oriented option while allowing
`--openai_model gpt-5.5` for higher-quality runs.

### Creative Providers

Add provider support without deleting existing providers.

- `LOCAL`: accepts local file paths and yields normalized local paths.
- `YOUTUBE`: downloads URLs into the cache using `yt-dlp`, then yields local
  downloaded paths.

Existing GCS provider remains available.

### Video Preprocessing

Add a local preprocessing layer:

- Validate video path exists.
- Capture duration metadata.
- Extract representative frames from the full video.
- Extract representative frames from the first 5 seconds.
- Extract audio.
- Transcribe audio with OpenAI.
- Store generated artifacts under `.cache/abcds-detector`.

Artifacts are local cache only and must not be committed.

### OpenAI Evaluation Service

Add an OpenAI service parallel to the Gemini service.

Responsibilities:

- Build OpenAI Responses API requests from existing prompts.
- Attach sampled frames as image inputs.
- Include transcript and timing metadata as text context.
- Request structured JSON matching the existing response schema.
- Validate and return parsed feature results.

The service should preserve existing `FeatureEvaluation` mapping so downstream
printing and storage logic needs minimal change.

### Prompt Strategy

Keep existing feature configurations and prompt templates. Add a short OpenAI
context preamble that explains:

- video evidence is sampled frames plus transcript;
- timestamps may be approximate;
- features must return false when evidence is insufficient;
- result IDs must exactly match feature IDs.

### Knowledge Graph Removal For MVP

Do not call Knowledge Graph on the OpenAI path. Brand/product matching uses:

- user-provided brand config;
- metadata inferred by OpenAI when enabled;
- sampled visual frames;
- transcript text.

## Sample Videos

The repo contains a local symlink:

```text
sample_videos -> /Users/gweng/Workspace/understand_creative/data/raw/creative_assets
```

Current sample content includes Google/YouTube and Facebook/Instagram MP4s plus
`video_asset_download_manifest.csv`.

Use this path in manual tests:

```text
sample_videos/google/videos/
sample_videos/facebook/videos/
```

`sample_videos` is gitignored.

## Error Handling

- Missing `OPENAI_API_KEY`: fail fast with clear message.
- Missing local video path: report the invalid input and continue other videos
  only if batch mode is enabled.
- YouTube download failure: include URL and downloader error in failed item.
- Audio extraction failure: continue with frame-only evaluation and mark
  `transcript_available: false`.
- OpenAI schema failure: retry once with stricter repair prompt, then record
  failed video.
- Empty feature result: warn and include failure details in JSON metadata.

## Testing

Unit tests:

- CLI parses OpenAI-specific args.
- local provider accepts valid local files.
- YouTube provider delegates to downloader and returns local cached path.
- preprocessor contract returns duration, frame paths, and transcript status.
- OpenAI response parser maps JSON to `FeatureEvaluation`.
- missing `OPENAI_API_KEY` fails before network calls.

Integration/manual tests:

- Run one short sample video from `sample_videos/google/videos/`.
- Run one sample video from `sample_videos/facebook/videos/`.
- Confirm JSON output contains long-form and/or Shorts feature arrays.

Networked OpenAI calls should not be required for unit tests. Mock OpenAI
responses for CI-safe coverage.

## Acceptance Criteria

- Existing tests still pass.
- `python main.py --help` shows OpenAI/local batch args.
- CLI can process at least one local MP4 from `sample_videos`.
- CLI can process a YouTube URL by downloading to cache first.
- JSON output includes one assessment per input video.
- No API keys, sample videos, extracted frames, audio, transcripts, cache, or
  outputs are tracked by git.

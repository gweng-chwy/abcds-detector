# OpenAI Evidence Review Helpers

This folder contains notebook-local helpers for reviewing OpenAI evidence
extraction outputs. The helper renders a 16:9 PNG with the extracted frame and
nearby transcript snippet on the left, and the evaluated ABCD/Shorts feature
checklist on the right.

## Expected Inputs

- Local sample videos grouped under first-level platform folders, for example
  `sample_videos/local/*.mp4` or `sample_videos/youtube/*.mov`.
- A JSON assessment file produced by `main.py -assessment_file ...`.
- Extracted frame images and preprocessing manifests from the local cache.

Generated videos, frames, transcripts, cache directories, and rendered figures
should stay untracked.

## Live Validation

Use `discover_sample_videos` and `build_validation_command` to prepare a small
OpenAI/local validation run. `run_validation_sample` skips the live run when
`OPENAI_API_KEY` is not set, so the notebook can be opened without credentials.

## Figure Output

`render_evidence_figure` saves a monochrome Keynote-ratio PNG at 150 dpi. Keep
rendered review artifacts in a gitignored output directory such as `.cache/` or
`sample_videos/outputs/`.

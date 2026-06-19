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

From the repository root, run the same workflow outside Jupyter with:

```bash
zsh -lc 'source ~/.zshrc && python3 -c "from pathlib import Path; import sys; root = Path.cwd(); sys.path.insert(0, str(root / \"notebooks\" / \"20260619_openai_evidence_review\")); from evidence_review import discover_sample_videos, build_validation_command, run_validation_sample; output_dir = root / \"outputs\" / \"openai_validation_sample\"; output_dir.mkdir(parents=True, exist_ok=True); selected = discover_sample_videos(root / \"sample_videos\", per_platform=2, seed=20260619); command = build_validation_command(selected, output_dir / \"assessment.json\"); print(\" \".join(str(part) for part in command)); run_validation_sample(command)"'
```

The generated command uses generic metadata such as `Unknown Brand` and
`Unknown Product`. This run validates the OpenAI extraction, local cache, and
visualization plumbing; it does not validate brand classification accuracy.

## Figure Output

`render_evidence_figure` saves a monochrome Keynote-ratio PNG at 150 dpi. The
notebook uses `notebooks/20260619_openai_evidence_review/figures/` for Keynote
handoff figures, and those rendered artifacts should stay untracked. For
gitignored scratch output, use `outputs/openai_validation_sample/`.

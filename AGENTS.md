# AGENTS.md

Repo guidance for AI coding agents working on ABCDs Detector.

This file adapts the engineering posture from
`multica-ai/andrej-karpathy-skills` for this repository: think before coding,
prefer simple changes, edit surgically, and verify outcomes.

## Project Context

ABCDs Detector evaluates video ads against long-form ABCD and Shorts creative
features. The current codebase is Python-first and historically centered on a
Colab notebook plus Google Cloud services:

- `main.py` orchestrates batch execution.
- `configuration.py`, `utils.py`, and `models.py` define runtime config and data
  structures.
- `features_repository/` defines supported feature prompts and metadata.
- `llms_evaluation/` contains LLM evaluation logic.
- `annotations_evaluation/` contains Google Video Intelligence annotation logic.
- `creative_providers/` resolves video sources.
- `gcp_api_services/` wraps Google Cloud APIs.
- `tests/` contains Python tests.

Near-term direction: add OpenAI-backed, LLM-only CLI batch execution for local
videos and YouTube URLs while preserving the existing feature definitions and
result shape.

## Think Before Coding

Before non-trivial edits, state assumptions and success criteria. If the request
has more than one reasonable interpretation, ask or present the tradeoff before
choosing.

Push back when a simpler or safer approach fits the goal better. Do not hide
confusion behind speculative implementation.

## Simplicity First

Use the smallest change that solves the requested problem.

- Do not add features beyond the request.
- Do not add abstractions for one-off logic.
- Do not add configurability unless the user asked for it or existing patterns
  require it.
- Prefer existing modules and dataclasses over new frameworks.
- If a change becomes large, pause and re-check the design.

## Surgical Changes

Touch only files needed for the task.

- Match existing style: Python 3.11, 2-space indentation, Google-style
  docstrings where useful.
- Do not reformat unrelated files.
- Do not remove unrelated comments, imports, or dead code.
- Clean up only unused code introduced by your own changes.
- Keep notebook changes separate from library/CLI changes unless explicitly
  requested.

Every changed line should trace back to the user's request.

## Secrets And Local Data

Never commit API keys, service-account files, downloaded videos, extracted
frames, audio files, transcripts, or local cache output.

Use environment variables for secrets. For OpenAI work, prefer:

```bash
export OPENAI_API_KEY="..."
```

If local sample videos are needed, place private/manual test assets under a
gitignored directory such as `sample_videos/`. Only tiny public fixtures should
live under `tests/fixtures/`.

## Development Commands

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies when needed:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run formatter/linter if available in the environment:

```bash
pyink .
ruff check .
```

Run the current CLI help:

```bash
python main.py --help
```

## Verification

For behavior changes, prefer tests before implementation when practical. At
minimum, run the narrowest relevant verification command and report exactly what
passed or failed.

For OpenAI/local-video work, verify without committing private media:

- unit tests for argument parsing, provider routing, preprocessing contracts,
  and result mapping;
- mocked OpenAI responses for schema handling;
- optional manual run against local files supplied by the user.

## Git Hygiene

The fork for development is:

```text
git@github.com:gweng-chwy/abcds-detector.git
```

Keep upstream available separately if configured. Do not rewrite user work or
discard local changes without explicit permission.

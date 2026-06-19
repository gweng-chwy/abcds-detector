#!/usr/bin/env python3

"""Helpers for reviewing OpenAI video evidence extraction outputs."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}

DEFAULT_METADATA = {
    "brand_name": "Unknown Brand",
    "brand_variations": "Unknown Brand",
    "branded_products": "Unknown Product",
    "branded_products_categories": "Unknown Category",
    "branded_call_to_actions": "Unknown CTA",
}

SANS_FONT_FALLBACKS = [
    "figmaSans",
    "figmaSans Fallback",
    "Inter",
    "Geist",
    "Arial",
    "DejaVu Sans",
]

MONO_FONT_FALLBACKS = [
    "figmaMono",
    "figmaMono Fallback",
    "JetBrains Mono",
    "Geist Mono",
    "Menlo",
    "DejaVu Sans Mono",
]


def discover_sample_videos(
    sample_root: str | Path,
    per_platform: int = 2,
    seed: int = 20260619,
) -> dict[str, list[Path]]:
  """Return deterministic video samples grouped by first-level directory."""
  sample_root = Path(sample_root)
  if not sample_root.is_dir():
    return {}

  rng = random.Random(seed)
  grouped_videos = {}

  for platform_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
    videos = sorted(
        path
        for path in platform_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) > per_platform:
      videos = sorted(rng.sample(videos, per_platform))
    grouped_videos[platform_dir.name] = videos

  return grouped_videos


def build_validation_command(
    selected_videos: dict[str, list[Path]] | list[Path],
    output_json: str | Path,
    metadata: dict[str, str] | None = None,
) -> list[str]:
  """Build a CLI command for validating a local OpenAI sample batch."""
  video_paths = _flatten_selected_videos(selected_videos)
  metadata_values = {**DEFAULT_METADATA, **(metadata or {})}

  return [
      sys.executable,
      "main.py",
      "--llm_provider",
      "OPENAI",
      "--creative_provider_type",
      "LOCAL",
      "--video_uris",
      ",".join(str(path) for path in video_paths),
      "-brand_name",
      metadata_values["brand_name"],
      "-brand_variations",
      metadata_values["brand_variations"],
      "-branded_products",
      metadata_values["branded_products"],
      "-branded_products_categories",
      metadata_values["branded_products_categories"],
      "-branded_call_to_actions",
      metadata_values["branded_call_to_actions"],
      "-use_llms",
      "-run_long_form_abcd",
      "-run_shorts",
      "-assessment_file",
      str(output_json),
  ]


def run_validation_sample(command: list[str]):
  """Run validation when OpenAI credentials are available."""
  if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set; skipping live validation.")
    return None

  return subprocess.run(command, check=True, text=True)


def load_json(path: str | Path) -> Any:
  """Load JSON from a local path."""
  with Path(path).open(encoding="utf-8") as input_file:
    return json.load(input_file)


def transcript_snippet(
    manifest: dict[str, Any],
    timestamp_seconds: float,
    window_seconds: float = 2.5,
) -> str:
  """Return transcript text near the requested timestamp."""
  lower_bound = timestamp_seconds - window_seconds
  upper_bound = timestamp_seconds + window_seconds
  segments = manifest.get("transcript_segments") or []
  nearby_segments = []

  for segment in segments:
    start = _first_present(
        segment,
        ("start", "start_time", "start_seconds", "timestamp_seconds"),
    )
    end = _first_present(segment, ("end", "end_time", "end_seconds"))
    if start is None:
      continue
    if end is None:
      end = start
    if float(start) <= upper_bound and float(end) >= lower_bound:
      text = str(segment.get("text", "")).strip()
      if text:
        nearby_segments.append(text)

  if nearby_segments:
    return " ".join(nearby_segments)

  if timestamp_seconds <= 5:
    first_5_text = _read_text_path(manifest.get("first_5_seconds_transcript_path"))
    if first_5_text:
      return first_5_text

  transcript_text = _read_text_path(manifest.get("transcript_path"))
  if transcript_text:
    return transcript_text

  transcript = str(manifest.get("transcript", "")).strip()
  return transcript or "No transcript available."


def load_feature_rows(assessment_path: str | Path) -> dict[str, list[tuple]]:
  """Flatten evaluated features into rows keyed by video URI."""
  data = load_json(assessment_path)
  if isinstance(data, dict):
    assessments = data.get("assessments", [])
  elif isinstance(data, list):
    assessments = data
  else:
    assessments = []
  feature_rows = {}

  for assessment in assessments:
    video_uri = assessment.get("video_uri", "")
    rows = []
    for key in (
        "long_form_abcd_evaluated_features",
        "shorts_evaluated_features",
    ):
      for feature_result in assessment.get(key, []) or []:
        feature = feature_result.get("feature", feature_result)
        rows.append((
            feature.get("category", feature_result.get("category", "")),
            feature.get("id", feature_result.get("id", "")),
            feature.get("name", feature_result.get("name", "")),
            bool(feature_result.get("detected", False)),
        ))
    feature_rows[video_uri] = rows

  return feature_rows


def render_evidence_figure(
    frame_path: str | Path,
    transcript_text: str,
    feature_rows: list[tuple],
    output_path: str | Path,
    title: str,
) -> Path:
  """Render a 16:9 evidence review figure."""
  os.environ.setdefault(
      "MPLCONFIGDIR",
      str(Path(os.environ.get("TMPDIR", "/tmp")) / "abcds-matplotlib"),
  )
  import matplotlib

  matplotlib.use("Agg", force=True)
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure
  import matplotlib.image as mpimg

  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  matplotlib.rcParams.update({
      "font.family": "sans-serif",
      "font.sans-serif": SANS_FONT_FALLBACKS,
      "font.monospace": MONO_FONT_FALLBACKS,
      "text.color": "black",
      "axes.edgecolor": "black",
  })

  fig = Figure(figsize=(13.333333333333334, 7.5), dpi=150, facecolor="white")
  FigureCanvasAgg(fig)
  grid = fig.add_gridspec(
      2,
      2,
      width_ratios=[1.15, 0.85],
      height_ratios=[0.74, 0.26],
      wspace=0.08,
      hspace=0.08,
      top=0.88,
  )

  image_axis = fig.add_subplot(grid[0, 0])
  transcript_axis = fig.add_subplot(grid[1, 0])
  checklist_axis = fig.add_subplot(grid[:, 1])

  image_axis.imshow(mpimg.imread(frame_path))
  image_axis.set_axis_off()
  fig.text(
      0.06,
      0.94,
      title,
      ha="left",
      va="top",
      fontsize=20,
      color="black",
  )

  _configure_text_axis(transcript_axis)
  transcript_axis.text(
      0.0,
      0.96,
      "TRANSCRIPT",
      transform=transcript_axis.transAxes,
      ha="left",
      va="top",
      fontsize=9,
      fontfamily="monospace",
  )
  transcript_axis.text(
      0.0,
      0.72,
      "\n".join(textwrap.wrap(transcript_text, width=54)),
      transform=transcript_axis.transAxes,
      ha="left",
      va="top",
      fontsize=17,
      linespacing=1.25,
  )

  _configure_text_axis(checklist_axis)
  checklist_axis.text(
      0.0,
      0.98,
      "FEATURE CHECKLIST",
      transform=checklist_axis.transAxes,
      ha="left",
      va="top",
      fontsize=9,
      fontfamily="monospace",
  )
  checklist_text = _format_feature_checklist(feature_rows)
  checklist_axis.text(
      0.0,
      0.92,
      checklist_text,
      transform=checklist_axis.transAxes,
      ha="left",
      va="top",
      fontsize=_checklist_font_size(feature_rows),
      linespacing=1.18,
  )

  fig.savefig(output_path, dpi=150, facecolor="white")
  fig.clear()
  return output_path


def _flatten_selected_videos(
    selected_videos: dict[str, list[Path]] | list[Path],
) -> list[Path]:
  if isinstance(selected_videos, dict):
    return [
        Path(path)
        for videos in selected_videos.values()
        for path in videos
    ]
  return [Path(path) for path in selected_videos]


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]):
  for key in keys:
    if key in mapping:
      return mapping[key]
  return None


def _read_text_path(path_value: str | Path | None) -> str:
  if not path_value:
    return ""

  path = Path(path_value)
  if not path.exists() or not path.is_file():
    return ""

  return path.read_text(encoding="utf-8").strip()


def _configure_text_axis(axis) -> None:
  axis.set_facecolor("white")
  axis.set_axis_off()


def _format_feature_checklist(feature_rows: list[tuple]) -> str:
  lines = []
  for category, feature_id, name, detected in feature_rows:
    mark = "[x]" if detected else "[ ]"
    label = name or feature_id
    category_text = str(category).replace("_", " ")
    line = f"{mark} {label}"
    if category_text:
      line = f"{line}\n    {category_text}"
    lines.append(line)
  return "\n\n".join(lines) if lines else "No evaluated features found."


def _checklist_font_size(feature_rows: list[tuple]) -> int:
  if len(feature_rows) <= 8:
    return 15
  if len(feature_rows) <= 14:
    return 12
  return 10

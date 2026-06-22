#!/usr/bin/env python3

"""Helpers for reviewing OpenAI video evidence extraction outputs."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
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

FIGURE_SIZE = (13.333333333333334, 7.5)
FIGURE_DPI = 180
ATTRIBUTION_HEADING = "ATTRIBUTION CHECKLIST"
FIGURE_WIDTH_RATIOS = (1.05, 0.95)
THUMBNAIL_GRID_SPACING = 0.0

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

_TIMESTAMP_PATTERN = re.compile(
    r"(?<![\d:])(?P<minutes>\d{1,2}):(?P<seconds>\d{2}(?:\.\d+)?)"
    r"|(?<![\d:])~?(?P<plain_seconds>\d+(?:\.\d+)?)\s*s\b"
)


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
    candidate_dirs = [platform_dir]
    nested_video_dir = platform_dir / "videos"
    if nested_video_dir.is_dir():
      candidate_dirs.append(nested_video_dir)

    videos = sorted({
        path
        for candidate_dir in candidate_dirs
        for path in candidate_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    })
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


def build_preprocess_cache_key(video_uri: str) -> str:
  """Return the OpenAI preprocessor cache key for a video URI."""
  return hashlib.sha256(video_uri.encode("utf-8")).hexdigest()[:16]


def load_preprocess_manifest(
    cache_dir: str | Path,
    video_uri: str,
) -> dict[str, Any]:
  """Load a preprocessor manifest from the OpenAI cache layout."""
  manifest_path = (
      Path(cache_dir)
      / build_preprocess_cache_key(video_uri)
      / "preprocess_manifest.json"
  )
  return load_json(manifest_path)


def select_review_frame(
    manifest: dict[str, Any],
    prefer_first_5: bool = False,
) -> tuple[Path, float]:
  """Select a frame path and timestamp from preprocessor evidence lists."""
  evidence_keys = (
      (
          "first_5_seconds_frame_evidence",
          "full_video_frame_evidence",
      )
      if prefer_first_5
      else (
          "full_video_frame_evidence",
          "first_5_seconds_frame_evidence",
      )
  )

  for evidence_key in evidence_keys:
    for frame in manifest.get(evidence_key, []) or []:
      frame_path = str(frame.get("path", "")).strip()
      if frame_path:
        return (
            Path(frame_path),
            float(frame.get("timestamp_seconds", 0.0)),
        )

  raise ValueError("No frame evidence found in preprocess manifest.")


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


def frame_entries_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
  """Return full-video frame entries labeled by sampled timestamp."""
  evidence = manifest.get("full_video_frame_evidence") or []
  if not evidence:
    evidence = manifest.get("first_5_seconds_frame_evidence") or []

  frame_entries = []
  for index, frame in enumerate(evidence, start=1):
    frame_path = str(frame.get("path", "")).strip()
    if not frame_path:
      continue
    timestamp_seconds = float(frame.get("timestamp_seconds", 0.0))
    frame_index = f"F{index:02d}"
    frame_entries.append({
        "index": frame_index,
        "label": f"{frame_index} {timestamp_seconds:.1f}s",
        "path": Path(frame_path),
        "timestamp_seconds": timestamp_seconds,
    })
  return frame_entries


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
        evidence_text = " ".join(
            str(feature_result.get(field_name, "")).strip()
            for field_name in ("evidence", "rationale")
            if str(feature_result.get(field_name, "")).strip()
        )
        rows.append((
            feature.get("category", feature_result.get("category", "")),
            feature.get("id", feature_result.get("id", "")),
            feature.get("name", feature_result.get("name", "")),
            bool(feature_result.get("detected", False)),
            evidence_text,
        ))
    feature_rows[video_uri] = rows

  return feature_rows


def render_evidence_figure(
    frame_entries: list[dict[str, Any]],
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

  fig = Figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor="white")
  FigureCanvasAgg(fig)
  grid = fig.add_gridspec(
      2,
      2,
      width_ratios=FIGURE_WIDTH_RATIOS,
      height_ratios=[0.84, 0.16],
      left=0.04,
      right=0.985,
      bottom=0.055,
      top=0.91,
      wspace=0.045,
      hspace=0.075,
  )

  transcript_axis = fig.add_subplot(grid[1, 0])
  checklist_axis = fig.add_subplot(grid[:, 1])
  thumbnail_grid = grid[0, 0].subgridspec(
      *_thumbnail_grid_shape(len(frame_entries)),
      wspace=THUMBNAIL_GRID_SPACING,
      hspace=THUMBNAIL_GRID_SPACING,
  )

  for axis_index, frame_entry in enumerate(frame_entries):
    row_count, column_count = _thumbnail_grid_shape(len(frame_entries))
    thumbnail_axis = fig.add_subplot(
        thumbnail_grid[axis_index // column_count, axis_index % column_count]
    )
    thumbnail_axis.imshow(mpimg.imread(frame_entry["path"]))
    thumbnail_axis.text(
        0.03,
        0.93,
        frame_entry["label"],
        transform=thumbnail_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontfamily="monospace",
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )
    thumbnail_axis.set_axis_off()
  for empty_index in range(len(frame_entries), math.prod(_thumbnail_grid_shape(len(frame_entries)))):
    row_count, column_count = _thumbnail_grid_shape(len(frame_entries))
    empty_axis = fig.add_subplot(
        thumbnail_grid[empty_index // column_count, empty_index % column_count]
    )
    empty_axis.set_axis_off()

  fig.text(
      0.06,
      0.94,
      _compact_title(title),
      ha="left",
      va="top",
      fontsize=12,
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
      "\n".join(textwrap.wrap(transcript_text, width=78))[:1200],
      transform=transcript_axis.transAxes,
      ha="left",
      va="top",
      fontsize=10,
      linespacing=1.12,
  )

  _configure_text_axis(checklist_axis)
  checklist_axis.text(
      0.0,
      0.98,
      ATTRIBUTION_HEADING,
      transform=checklist_axis.transAxes,
      ha="left",
      va="top",
      fontsize=10,
      fontfamily="monospace",
  )
  checklist_columns = _feature_checklist_columns(
      feature_rows,
      frame_entries,
      columns=_checklist_column_count(feature_rows),
  )
  checklist_font_size = _checklist_font_size(feature_rows)
  for column_index, lines in enumerate(checklist_columns):
    checklist_axis.text(
        column_index / len(checklist_columns),
        0.92,
        "\n".join(lines),
        transform=checklist_axis.transAxes,
        ha="left",
        va="top",
        fontsize=checklist_font_size,
        fontfamily="monospace",
        linespacing=1.02,
      )

  fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
  fig.clear()
  return output_path


def format_feature_checklist(
    feature_rows: list[tuple],
    frame_entries: list[dict[str, Any]],
    columns: int = 2,
) -> str:
  """Return a compact feature checklist with nearest frame indexes."""
  checklist_columns = _feature_checklist_columns(
      feature_rows,
      frame_entries,
      columns=columns,
  )
  if not checklist_columns:
    return "No evaluated features found."
  row_count = max(len(column) for column in checklist_columns)
  output_lines = []
  for row_index in range(row_count):
    output_lines.append(
        "   ".join(
            column[row_index] if row_index < len(column) else ""
            for column in checklist_columns
        ).rstrip()
    )
  return "\n".join(output_lines)


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


def _thumbnail_grid_shape(frame_count: int) -> tuple[int, int]:
  if frame_count <= 0:
    return (1, 1)
  if frame_count <= 6:
    columns = 3
  elif frame_count <= 12:
    columns = 4
  elif frame_count <= 25:
    columns = 5
  else:
    columns = 6
  return (math.ceil(frame_count / columns), columns)


def _feature_checklist_columns(
    feature_rows: list[tuple],
    frame_entries: list[dict[str, Any]],
    columns: int,
) -> list[list[str]]:
  if not feature_rows:
    return [["No evaluated features found."]]

  checklist_lines = []
  for row in feature_rows:
    category, feature_id, name, detected = row[:4]
    evidence_text = row[4] if len(row) > 4 else ""
    mark = "[x]" if detected else "[ ]"
    refs = _frame_refs_for_feature(bool(detected), str(evidence_text), frame_entries)
    label = _shorten_feature_label(str(name or feature_id))
    checklist_lines.append(f"{mark} {label}\n    {refs}")

  columns = max(1, min(columns, len(checklist_lines)))
  rows_per_column = math.ceil(len(checklist_lines) / columns)
  return [
      checklist_lines[index : index + rows_per_column]
      for index in range(0, len(checklist_lines), rows_per_column)
  ]


def _frame_refs_for_feature(
    detected: bool,
    evidence_text: str,
    frame_entries: list[dict[str, Any]],
    max_refs: int = 3,
) -> str:
  if not detected or not evidence_text or not frame_entries:
    return "--"

  refs = []
  for timestamp in _timestamps_from_text(evidence_text):
    nearest_frame = min(
        frame_entries,
        key=lambda frame_entry: abs(
            float(frame_entry["timestamp_seconds"]) - timestamp
        ),
    )
    frame_index = str(nearest_frame["index"])
    if frame_index not in refs:
      refs.append(frame_index)
    if len(refs) >= max_refs:
      break
  return ",".join(refs) if refs else "--"


def _timestamps_from_text(text: str) -> list[float]:
  timestamps = []
  for match in _TIMESTAMP_PATTERN.finditer(text):
    if match.group("plain_seconds") is not None:
      timestamps.append(float(match.group("plain_seconds")))
      continue
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    timestamps.append(minutes * 60 + seconds)
  return timestamps


def _shorten_feature_label(label: str, max_chars: int = 22) -> str:
  label = " ".join(label.split())
  if len(label) <= max_chars:
    return label
  return label[: max_chars - 1].rstrip() + "."


def _compact_title(title: str) -> str:
  parts = Path(title).parts
  if "sample_videos" in parts:
    start = parts.index("sample_videos")
    return "/".join(parts[start:])
  return title


def _checklist_column_count(feature_rows: list[tuple]) -> int:
  if len(feature_rows) <= 24:
    return 1
  if len(feature_rows) <= 54:
    return 2
  return 3


def _checklist_font_size(feature_rows: list[tuple]) -> int:
  if len(feature_rows) <= 24:
    return 8.6
  if len(feature_rows) <= 54:
    return 7.8
  return 6.2

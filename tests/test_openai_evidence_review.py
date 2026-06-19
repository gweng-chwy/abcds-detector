#!/usr/bin/env python3

"""Tests for OpenAI evidence review notebook helpers."""

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


def _load_evidence_review():
  helper_path = (
      Path(__file__).resolve().parents[1]
      / "notebooks"
      / "20260619_openai_evidence_review"
      / "evidence_review.py"
  )
  spec = importlib.util.spec_from_file_location(
      "openai_evidence_review", helper_path
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_helper_imports_by_file_path():
  """Notebook helper can be loaded directly from its file path."""
  helper = _load_evidence_review()

  assert helper.VIDEO_EXTENSIONS == {".mp4", ".mov", ".m4v"}


def test_render_font_fallbacks_follow_design_order():
  """Figure rendering keeps DESIGN.md Figma fonts before generic fallbacks."""
  helper = _load_evidence_review()

  assert helper.SANS_FONT_FALLBACKS[:4] == [
      "figmaSans",
      "figmaSans Fallback",
      "Inter",
      "Geist",
  ]
  assert helper.MONO_FONT_FALLBACKS[:4] == [
      "figmaMono",
      "figmaMono Fallback",
      "JetBrains Mono",
      "Geist Mono",
  ]


def test_discover_sample_videos_groups_platform_dirs_deterministically(tmp_path):
  """Video discovery samples first-level platform directories deterministically."""
  youtube = tmp_path / "youtube"
  local = tmp_path / "local"
  nested = youtube / "nested"
  youtube.mkdir()
  local.mkdir()
  nested.mkdir()
  for filename in ["b.mp4", "a.mov", "c.m4v", "ignore.txt"]:
    (youtube / filename).write_text("video", encoding="utf-8")
  for filename in ["z.mp4", "x.mov", "y.m4v"]:
    (local / filename).write_text("video", encoding="utf-8")
  (nested / "nested.mp4").write_text("video", encoding="utf-8")
  (tmp_path / "root.mp4").write_text("video", encoding="utf-8")

  helper = _load_evidence_review()

  first = helper.discover_sample_videos(tmp_path, per_platform=2, seed=7)
  second = helper.discover_sample_videos(tmp_path, per_platform=2, seed=7)

  assert set(first) == {"local", "youtube"}
  assert first == second
  assert all(len(videos) == 2 for videos in first.values())
  assert [path.parent.name for path in first["youtube"]] == ["youtube", "youtube"]
  assert {path.name for path in first["youtube"]}.issubset(
      {"a.mov", "b.mp4", "c.m4v"}
  )


def test_build_validation_command_uses_openai_local_flags_and_unknown_metadata(
    tmp_path,
):
  """Validation command is ready for local OpenAI batch assessment."""
  selected_videos = {
      "local": [tmp_path / "local" / "a.mp4"],
      "youtube": [tmp_path / "youtube" / "b.mov"],
  }
  output_json = tmp_path / "results.json"
  helper = _load_evidence_review()

  command = helper.build_validation_command(selected_videos, output_json)

  assert command[:2] == [sys.executable, "main.py"]
  assert command[command.index("--llm_provider") + 1] == "OPENAI"
  assert command[command.index("--creative_provider_type") + 1] == "LOCAL"
  assert command[command.index("--video_uris") + 1] == ",".join(
      str(path)
      for videos in selected_videos.values()
      for path in videos
  )
  assert command[command.index("-brand_name") + 1] == "Unknown Brand"
  assert command[command.index("-brand_variations") + 1] == "Unknown Brand"
  assert command[command.index("-branded_products") + 1] == "Unknown Product"
  assert (
      command[command.index("-branded_products_categories") + 1]
      == "Unknown Category"
  )
  assert "-run_long_form_abcd" in command
  assert "-run_shorts" in command
  assert command[command.index("-assessment_file") + 1] == str(output_json)


def test_transcript_snippet_prefers_timecoded_segments_near_timestamp():
  """Transcript snippets use nearby segments before broad transcript text."""
  helper = _load_evidence_review()
  manifest = {
      "transcript": "Full transcript fallback.",
      "transcript_segments": [
          {"start": 0.0, "end": 1.0, "text": "Opening line."},
          {"start_time": 4.0, "end_time": 6.0, "text": "Product shown."},
          {"start_seconds": 9.0, "end_seconds": 11.0, "text": "CTA appears."},
      ],
  }

  snippet = helper.transcript_snippet(manifest, 5.0, window_seconds=1.0)

  assert snippet == "Product shown."


def test_transcript_snippet_falls_back_to_text_or_placeholder():
  """Transcript snippets remain usable when segment data is unavailable."""
  helper = _load_evidence_review()

  assert helper.transcript_snippet({"transcript": " Plain transcript. "}, 3.0) == (
      "Plain transcript."
  )
  assert helper.transcript_snippet({}, 3.0) == "No transcript available."


def test_load_feature_rows_flattens_assessment_feature_lists(tmp_path):
  """Feature rows combine long-form and Shorts feature outputs by video URI."""
  assessment_path = tmp_path / "assessment.json"
  assessment_path.write_text(
      json.dumps({
          "assessments": [
              {
                  "video_uri": "sample_videos/ad-1.mp4",
                  "long_form_abcd_evaluated_features": [
                      {
                          "category": "LONG_FORM_ABCD",
                          "id": "a_supers",
                          "name": "Supers",
                          "detected": True,
                      }
                  ],
                  "shorts_evaluated_features": [
                      {
                          "category": "SHORTS",
                          "id": "shorts_pacing",
                          "name": "Fast pacing",
                          "detected": False,
                      }
                  ],
              },
              {
                  "video_uri": "sample_videos/ad-2.mp4",
                  "long_form_abcd_evaluated_features": [],
                  "shorts_evaluated_features": [
                      {
                          "category": "SHORTS",
                          "id": "shorts_hook",
                          "name": "Opening hook",
                          "detected": True,
                      }
                  ],
              },
          ]
      }),
      encoding="utf-8",
  )
  helper = _load_evidence_review()

  rows = helper.load_feature_rows(assessment_path)

  assert rows == {
      "sample_videos/ad-1.mp4": [
          ("LONG_FORM_ABCD", "a_supers", "Supers", True),
          ("SHORTS", "shorts_pacing", "Fast pacing", False),
      ],
      "sample_videos/ad-2.mp4": [
          ("SHORTS", "shorts_hook", "Opening hook", True),
      ],
  }


def test_render_evidence_figure_writes_non_empty_png(tmp_path):
  """Evidence figure rendering produces a PNG artifact for notebook review."""
  frame_path = tmp_path / "frame.png"
  Image.new("RGB", (32, 18), "white").save(frame_path)
  output_path = tmp_path / "evidence.png"
  helper = _load_evidence_review()

  rendered_path = helper.render_evidence_figure(
      frame_path=frame_path,
      transcript_text="Product appears in the opening shot.",
      feature_rows=[
          ("LONG_FORM_ABCD", "a_supers", "Supers", True),
          ("SHORTS", "shorts_hook", "Opening hook", False),
      ],
      output_path=output_path,
      title="sample_videos/ad.mp4",
  )

  assert rendered_path == output_path
  assert output_path.exists()
  assert output_path.stat().st_size > 0

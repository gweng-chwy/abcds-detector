#!/usr/bin/env python3

"""Tests for OpenAI evidence review notebook helpers."""

import importlib.util
import hashlib
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


def test_openai_evidence_review_notebook_references_helper_workflow():
  """Notebook documents the approved evidence review helper workflow."""
  notebook_path = (
      Path(__file__).resolve().parents[1]
      / "notebooks"
      / "20260619_openai_evidence_review"
      / "openai_evidence_review.ipynb"
  )

  notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
  source = "\n".join(
      "".join(cell.get("source", ""))
      for cell in notebook.get("cells", [])
  )

  assert notebook["nbformat"] == 4
  assert notebook["nbformat_minor"] == 5
  assert "discover_sample_videos" in source
  assert "build_validation_command" in source
  assert "render_evidence_figure" in source
  assert "outputs/openai_validation_sample" in source
  assert "load_feature_rows" in source
  assert "full_transcript_text" in source
  assert "transcript_snippet(manifest" not in source
  assert ".cache/abcds-detector" in source
  assert "load_preprocess_manifest" in source
  assert "frame_entries_from_manifest" in source


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


def test_discover_sample_videos_finds_nested_platform_video_dirs(tmp_path):
  """Video discovery supports repo sample layout under platform/videos."""
  google_videos = tmp_path / "google" / "videos"
  tiktok_videos = tmp_path / "tiktok" / "videos"
  google_videos.mkdir(parents=True)
  tiktok_videos.mkdir(parents=True)
  for filename in ["g1.mp4", "g2.mov", "g3.m4v"]:
    (google_videos / filename).write_text("video", encoding="utf-8")
  for filename in ["t1.mp4", "t2.mp4"]:
    (tiktok_videos / filename).write_text("video", encoding="utf-8")

  helper = _load_evidence_review()

  selected = helper.discover_sample_videos(tmp_path, per_platform=2, seed=13)

  assert set(selected) == {"google", "tiktok"}
  assert all(len(videos) == 2 for videos in selected.values())
  assert all(path.parent.name == "videos" for path in selected["google"])
  assert all(path.parent.name == "videos" for path in selected["tiktok"])


def test_discover_sample_videos_returns_empty_for_missing_or_file_path(tmp_path):
  """Video discovery tolerates absent optional local sample inputs."""
  helper = _load_evidence_review()
  file_path = tmp_path / "not-a-directory.mp4"
  file_path.write_text("video", encoding="utf-8")

  assert helper.discover_sample_videos(tmp_path / "missing") == {}
  assert helper.discover_sample_videos(file_path) == {}


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


def test_transcript_snippet_reads_preprocessor_transcript_path(tmp_path):
  """Transcript snippets support Task 3 preprocessor manifest file paths."""
  transcript_path = tmp_path / "transcript.txt"
  transcript_path.write_text(" Full transcript from preprocessor. ", encoding="utf-8")
  helper = _load_evidence_review()

  assert helper.transcript_snippet({"transcript_path": str(transcript_path)}, 8.0) == (
      "Full transcript from preprocessor."
  )


def test_full_transcript_text_prefers_complete_preprocessor_transcript(tmp_path):
  """Evidence figures use full-video transcript text rather than first-5s text."""
  transcript_path = tmp_path / "transcript.txt"
  first_5_path = tmp_path / "transcript_first_5s.txt"
  transcript_path.write_text(
      "Full transcript line one.\nFull transcript line two.",
      encoding="utf-8",
  )
  first_5_path.write_text("Opening-only transcript.", encoding="utf-8")
  helper = _load_evidence_review()

  transcript = helper.full_transcript_text({
      "transcript_path": str(transcript_path),
      "first_5_seconds_transcript_path": str(first_5_path),
      "transcript_segments": [
          {"start": 0.0, "end": 1.0, "text": "Segment text."},
      ],
  })

  assert transcript == "Full transcript line one.\nFull transcript line two."


def test_full_transcript_text_falls_back_to_segments_before_first_5(tmp_path):
  """Full transcript fallback can reconstruct text from available segments."""
  first_5_path = tmp_path / "transcript_first_5s.txt"
  first_5_path.write_text("Opening-only transcript.", encoding="utf-8")
  helper = _load_evidence_review()

  transcript = helper.full_transcript_text({
      "first_5_seconds_transcript_path": str(first_5_path),
      "transcript_segments": [
          {"start": 0.0, "end": 1.0, "text": "Opening line."},
          {"start": 5.0, "end": 7.0, "text": "Closing line."},
      ],
  })

  assert transcript == "Opening line. Closing line."


def test_load_preprocess_manifest_uses_preprocessor_cache_key(tmp_path):
  """Manifest loading follows the OpenAI preprocessor cache layout."""
  video_uri = "sample_videos/youtube/ad.mp4"
  cache_key = hashlib.sha256(video_uri.encode("utf-8")).hexdigest()[:16]
  manifest_path = tmp_path / cache_key / "preprocess_manifest.json"
  manifest_path.parent.mkdir()
  manifest_path.write_text(
      json.dumps({"source": {"original_uri": video_uri}}),
      encoding="utf-8",
  )
  helper = _load_evidence_review()

  manifest = helper.load_preprocess_manifest(tmp_path, video_uri)

  assert manifest == {"source": {"original_uri": video_uri}}


def test_select_review_frame_prefers_first_5_seconds_when_requested():
  """Frame selection can focus on first-5-second evidence for review."""
  helper = _load_evidence_review()
  manifest = {
      "full_video_frame_evidence": [
          {"path": ".cache/full/frame_0001.jpg", "timestamp_seconds": 8.5}
      ],
      "first_5_seconds_frame_evidence": [
          {"path": ".cache/first/frame_0001.jpg", "timestamp_seconds": 1.25}
      ],
  }

  frame_path, timestamp_seconds = helper.select_review_frame(
      manifest,
      prefer_first_5=True,
  )

  assert frame_path == Path(".cache/first/frame_0001.jpg")
  assert timestamp_seconds == 1.25


def test_select_review_frame_falls_back_to_full_video_evidence():
  """Frame selection uses full-video evidence when first-5s evidence is absent."""
  helper = _load_evidence_review()
  manifest = {
      "first_5_seconds_frame_evidence": [],
      "full_video_frame_evidence": [
          {"path": ".cache/full/frame_0001.jpg", "timestamp_seconds": 6.0}
      ],
  }

  frame_path, timestamp_seconds = helper.select_review_frame(
      manifest,
      prefer_first_5=True,
  )

  assert frame_path == Path(".cache/full/frame_0001.jpg")
  assert timestamp_seconds == 6.0


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
                          "sub_category": "ATTRACT",
                          "id": "a_supers",
                          "name": "Supers",
                          "detected": True,
                          "evidence": "Visible at 0.0s and 4.5s.",
                      }
                  ],
                  "shorts_evaluated_features": [
                      {
                          "category": "SHORTS",
                          "sub_category": "ATTRACT",
                          "id": "shorts_pacing",
                          "name": "Fast pacing",
                          "detected": False,
                          "evidence": "",
                      }
                  ],
              },
              {
                  "video_uri": "sample_videos/ad-2.mp4",
                  "long_form_abcd_evaluated_features": [],
                  "shorts_evaluated_features": [
                      {
                          "category": "SHORTS",
                          "sub_category": "CONNECT",
                          "id": "shorts_hook",
                          "name": "Opening hook",
                          "detected": True,
                          "evidence": "Hook appears around 00:02.",
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
          (
              "LONG_FORM_ABCD",
              "a_supers",
              "Supers",
              True,
              "Visible at 0.0s and 4.5s.",
              "ATTRACT",
          ),
          ("SHORTS", "shorts_pacing", "Fast pacing", False, "", "ATTRACT"),
      ],
      "sample_videos/ad-2.mp4": [
          (
              "SHORTS",
              "shorts_hook",
              "Opening hook",
              True,
              "Hook appears around 00:02.",
              "CONNECT",
          ),
      ],
  }


def test_frame_entries_from_manifest_uses_full_video_timeline():
  """Figure thumbnails use full-video frame evidence with timestamp labels."""
  helper = _load_evidence_review()
  manifest = {
      "full_video_frame_evidence": [
          {"path": ".cache/full/frame_0001.jpg", "timestamp_seconds": 0.0},
          {"path": ".cache/full/frame_0002.jpg", "timestamp_seconds": 2.25},
      ],
      "first_5_seconds_frame_evidence": [
          {"path": ".cache/first/frame_0001.jpg", "timestamp_seconds": 0.0},
      ],
  }

  entries = helper.frame_entries_from_manifest(manifest)

  assert entries == [
      {
          "index": "F01",
          "label": "F01 0.0s",
          "path": Path(".cache/full/frame_0001.jpg"),
          "timestamp_seconds": 0.0,
      },
      {
          "index": "F02",
          "label": "F02 2.2s",
          "path": Path(".cache/full/frame_0002.jpg"),
          "timestamp_seconds": 2.25,
      },
  ]


def test_feature_frame_references_map_timestamps_to_nearest_frame():
  """Feature checklist can cite nearest thumbnail indexes from evidence text."""
  helper = _load_evidence_review()
  frame_entries = [
      {"index": "F01", "timestamp_seconds": 0.0},
      {"index": "F02", "timestamp_seconds": 2.0},
      {"index": "F03", "timestamp_seconds": 5.0},
  ]
  rows = [
      ("LONG_FORM_ABCD", "a_supers", "Supers", True, "Seen at 0.0s and 4.7s."),
      ("SHORTS", "shorts_hook", "Hook", False, "Not present at 00:02."),
  ]

  checklist = helper.format_feature_checklist(rows, frame_entries, columns=1)

  assert "[x] Supers\n    F01,F03" in checklist
  assert "[ ] Hook\n    --" in checklist


def test_feature_checklist_groups_rows_by_abcde_subcategory():
  """Attribution checklist groups features into A/B/C/D/E strategy sections."""
  helper = _load_evidence_review()
  frame_entries = [
      {"index": "F01", "timestamp_seconds": 0.0},
      {"index": "F02", "timestamp_seconds": 2.0},
  ]
  rows = [
      ("SHORTS", "shorts_extra", "Production Style", True, "Seen at 2.0s.", "NONE"),
      ("LONG_FORM_ABCD", "b_brand", "Brand Visuals", True, "Seen at 0.0s.", "BRAND"),
      ("LONG_FORM_ABCD", "a_start", "Dynamic Start", False, "", "ATTRACT"),
      ("LONG_FORM_ABCD", "d_cta", "Call To Action", True, "Seen at 2.0s.", "DIRECT"),
      ("SHORTS", "c_context", "Product Context", False, "", "CONNECT"),
  ]

  checklist_columns = helper._feature_checklist_columns(
      rows,
      frame_entries,
      columns=2,
  )
  long_form_column, shorts_column = checklist_columns

  assert long_form_column.index("A / ATTRACT") < long_form_column.index("B / BRAND")
  assert long_form_column.index("B / BRAND") < long_form_column.index("D / DIRECT")
  assert shorts_column.index("C / CONNECT") < shorts_column.index("E / EXTRA")
  assert "[x] Production Style\n    F02" in shorts_column


def test_feature_checklist_splits_long_form_and_shorts_sections():
  """Attribution checklist separates long-form and Shorts before ABCDE groups."""
  helper = _load_evidence_review()
  frame_entries = [{"index": "F01", "timestamp_seconds": 0.0}]
  rows = [
      ("SHORTS", "shorts_extra", "Production Style", True, "Seen at 0.0s.", "NONE"),
      ("LONG_FORM_ABCD", "b_brand", "Brand Visuals", True, "Seen at 0.0s.", "BRAND"),
      ("SHORTS", "shorts_hook", "Human Voice Presence", False, "", "ATTRACT"),
      ("LONG_FORM_ABCD", "a_start", "Dynamic Start", False, "", "ATTRACT"),
  ]

  checklist_columns = helper._feature_checklist_columns(
      rows,
      frame_entries,
      columns=2,
  )

  assert checklist_columns[0][:3] == [
      "LONG FORM ABCD",
      "A / ATTRACT",
      "[ ] Dynamic Start\n    --",
  ]
  assert "B / BRAND" in checklist_columns[0]
  assert "[x] Brand Visuals\n    F01" in checklist_columns[0]
  assert checklist_columns[1][:3] == [
      "SHORTS",
      "A / ATTRACT",
      "[ ] Human Voice Presence\n    --",
  ]
  assert "E / EXTRA" in checklist_columns[1]
  assert "[x] Production Style\n    F01" in checklist_columns[1]


def test_feature_checklist_infers_group_from_legacy_feature_ids():
  """Checklist grouping remains compatible with older tuple rows."""
  helper = _load_evidence_review()

  checklist = helper.format_feature_checklist(
      [
          ("LONG_FORM_ABCD", "b_brand_visuals", "Brand Visuals", True, ""),
          ("LONG_FORM_ABCD", "a_supers", "Supers", False, ""),
          ("SHORTS", "shorts_native_style", "Production Style", True, ""),
      ],
      [],
      columns=1,
  )

  assert checklist.index("A / ATTRACT") < checklist.index("B / BRAND")
  assert "E / EXTRA" in checklist


def test_attribution_group_headers_are_identifiable_for_bold_rendering():
  """Figure renderer can bold group headers without bolding feature rows."""
  helper = _load_evidence_review()

  assert helper._is_attribution_group_header("A / ATTRACT")
  assert helper._is_attribution_group_header("E / EXTRA")
  assert not helper._is_attribution_group_header("[x] Brand Visuals")


def test_feature_labels_compact_parenthetical_suffixes_without_breaking_them():
  """Feature labels keep speech/text and first-5s variants distinguishable."""
  helper = _load_evidence_review()

  assert helper._shorten_feature_label("Brand Mention (Speech)") == (
      "Brand Mention [Sp]"
  )
  assert helper._shorten_feature_label(
      "Brand Mention (Speech) (First 5 seconds)"
  ) == "Brand Mention [Sp 5s]"
  assert helper._shorten_feature_label("Product Mention (Speech)") == (
      "Product Mention [Sp]"
  )
  assert helper._shorten_feature_label(
      "Product Mention (Speech) (First 5 seconds)"
  ) == "Product Mention [Sp 5s]"
  assert helper._shorten_feature_label("Product Mention (Text)") == (
      "Product Mention [Txt]"
  )
  assert helper._shorten_feature_label(
      "Product Mention (Text) (First 5 seconds)"
  ) == "Product Mention [Txt 5s]"
  assert helper._shorten_feature_label("Brand Visuals (First 5 seconds)") == (
      "Brand Visuals [5s]"
  )


def test_feature_labels_do_not_leave_unclosed_parentheses():
  """Truncated checklist labels avoid dangling parenthetical fragments."""
  helper = _load_evidence_review()

  for label in [
      "Brand Mention (Speech) (First 5 seconds)",
      "Product Mention (Speech) (First 5 seconds)",
      "Product Mention (Text) (First 5 seconds)",
  ]:
    shortened = helper._shorten_feature_label(label)
    assert shortened.count("(") == shortened.count(")")


def test_dense_layout_uses_larger_thumbnail_and_checklist_grid():
  """Dense figures reserve more space for readable thumbnails and features."""
  helper = _load_evidence_review()

  assert helper.ATTRIBUTION_HEADING == "ATTRIBUTION CHECKLIST"
  assert helper.THUMBNAIL_GRID_SPACING == 0.0
  assert helper.FIGURE_WIDTH_RATIOS == (1.05, 0.95)
  assert helper.FIGURE_HEIGHT_RATIOS == (0.78, 0.22)
  assert helper._thumbnail_grid_shape(24) == (5, 5)
  assert helper._checklist_column_count([None] * 46) == 2
  assert helper._checklist_font_size([None] * 46) >= 7.5


def test_transcript_block_uses_available_space_without_overflowing():
  """Transcript layout keeps long full text inside a bounded figure block."""
  helper = _load_evidence_review()
  transcript = " ".join(f"word{index:03d}" for index in range(180))

  block = helper.format_transcript_block(transcript)
  lines = block.splitlines()

  assert len(lines) <= helper.TRANSCRIPT_MAX_LINES
  assert lines[-1].endswith("...")
  assert all(len(line) <= helper.TRANSCRIPT_WRAP_WIDTH for line in lines)


def test_transcript_block_preserves_short_full_transcript():
  """Short transcripts render without unnecessary truncation."""
  helper = _load_evidence_review()
  transcript = "A concise full transcript that fits on one line."

  assert helper.format_transcript_block(transcript) == transcript


def test_compact_title_uses_sample_video_relative_path():
  """Figure titles stay compact for absolute sample video paths."""
  helper = _load_evidence_review()

  assert helper._compact_title(
      "/Users/gweng/Workspace/abcds-detector/sample_videos/google/videos/ad.mp4"
  ) == "sample_videos/google/videos/ad.mp4"


def test_render_evidence_figure_writes_non_empty_png(tmp_path):
  """Evidence figure rendering produces a PNG artifact for notebook review."""
  frame_entries = []
  for index in range(1, 25):
    frame_path = tmp_path / f"frame_{index:04d}.png"
    Image.new("RGB", (64, 36), "white").save(frame_path)
    frame_entries.append({
        "index": f"F{index:02d}",
        "label": f"F{index:02d} {index - 1}.0s",
        "path": frame_path,
        "timestamp_seconds": float(index - 1),
    })
  output_path = tmp_path / "evidence.png"
  helper = _load_evidence_review()

  rendered_path = helper.render_evidence_figure(
      frame_entries=frame_entries,
      transcript_text="Product appears in the opening shot.",
      feature_rows=[
          (
              "LONG_FORM_ABCD",
              f"feature_{index:02d}",
              f"Feature {index:02d}",
              index % 3 == 0,
              f"Evidence at {index % 6}.0s.",
          )
          for index in range(46)
      ],
      output_path=output_path,
      title="sample_videos/ad.mp4",
  )

  assert rendered_path == output_path
  assert output_path.exists()
  assert output_path.stat().st_size > 0
  with Image.open(output_path) as output_image:
    assert output_image.size == (2400, 1350)

  import matplotlib.pyplot as plt

  assert plt.get_fignums() == []

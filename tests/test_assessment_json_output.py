#!/usr/bin/env python3

"""Tests for JSON assessment output."""

import importlib
import json
import sys

import models


_FORBIDDEN_IMPORTS = (
    "google.cloud.bigquery",
    "moviepy.editor",
    "gcp_api_services.bigquery_api_service",
    "gcp_api_services.gcs_api_service",
)


class _BlockImports:
  """Import hook that fails when forbidden modules are imported."""

  def find_spec(self, fullname, path=None, target=None):
    if fullname in _FORBIDDEN_IMPORTS:
      raise AssertionError(f"{fullname} imported at module load")
    return None


def _import_generic_helpers_with_google_import_guard(monkeypatch):
  """Import generic_helpers and fail if it imports Google-heavy modules."""
  sys.modules.pop("helpers.generic_helpers", None)
  for module_name in _FORBIDDEN_IMPORTS:
    monkeypatch.delitem(sys.modules, module_name, raising=False)
  import_guard = _BlockImports()
  sys.meta_path.insert(0, import_guard)
  try:
    return importlib.import_module("helpers.generic_helpers")
  finally:
    sys.meta_path.remove(import_guard)


def _feature(feature_id="a_supers"):
  return models.VideoFeature(
      id=feature_id,
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


def _feature_eval(feature_id="a_supers"):
  return models.FeatureEvaluation(
      feature=_feature(feature_id),
      detected=True,
      confidence_score=0.9,
      rationale="Text visible.",
      evidence="Frame 1.",
      strengths="Clear.",
      weaknesses="Sampled.",
  )


def test_write_assessments_json(tmp_path, monkeypatch):
  """JSON writer serializes assessments and creates parent directories."""
  generic_helpers = _import_generic_helpers_with_google_import_guard(monkeypatch)
  assessment = models.VideoAssessment(
      brand_name="Chewy",
      video_uri="sample_videos/ad.mp4",
      long_form_abcd_evaluated_features=[_feature_eval()],
      shorts_evaluated_features=[],
      config=object(),
  )
  output = tmp_path / "nested" / "results.json"

  generic_helpers.write_assessments_json([assessment], str(output))

  data = json.loads(output.read_text(encoding="utf-8"))
  assert data == {
      "assessments": [{
          "brand_name": "Chewy",
          "video_uri": "sample_videos/ad.mp4",
          "long_form_abcd_evaluated_features": [{
              "id": "a_supers",
              "name": "Supers",
              "category": "LONG_FORM_ABCD",
              "sub_category": "ATTRACT",
              "video_segment": "FULL_VIDEO",
              "evaluation_criteria": "Text overlays are present.",
              "detected": True,
              "confidence_score": 0.9,
              "rationale": "Text visible.",
              "evidence": "Frame 1.",
              "strengths": "Clear.",
              "weaknesses": "Sampled.",
          }],
          "shorts_evaluated_features": [],
      }]
  }


def test_main_writes_assessment_json_when_assessment_file_set(
    tmp_path, monkeypatch
):
  """main writes returned assessments to the configured JSON path."""
  import main

  assessment = models.VideoAssessment(
      brand_name="Chewy",
      video_uri="sample_videos/ad.mp4",
      long_form_abcd_evaluated_features=[_feature_eval()],
      shorts_evaluated_features=[],
      config=object(),
  )
  output = tmp_path / "results.json"
  write_calls = []

  monkeypatch.setattr(
      main,
      "execute_abcd_assessment_for_videos",
      lambda config: [assessment],
  )

  from helpers import generic_helpers

  monkeypatch.setattr(
      generic_helpers,
      "write_assessments_json",
      lambda video_assessments, assessment_file: write_calls.append(
          (video_assessments, assessment_file)
      ),
  )

  main.main([
      "--llm_provider",
      "OPENAI",
      "--video_uris",
      "sample_videos/ad.mp4",
      "-brand_name",
      "Chewy",
      "-brand_variations",
      "Chewy",
      "-branded_products",
      "Chewy Pharmacy",
      "-branded_products_categories",
      "pet supplies",
      "-assessment_file",
      str(output),
  ])

  assert write_calls == [([assessment], str(output))]

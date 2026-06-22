"""Tests for OpenAI detector."""

from dataclasses import dataclass, field

import pytest

import models
from llms_evaluation.openai_detector import OpenAIDetector


@dataclass
class ConfigStub:
  """Minimal configuration object required by prompt generation."""

  openai_model: str = "gpt-test"
  brand_name: str = "Chewy"
  brand_variations: list[str] = field(default_factory=lambda: ["Chewy"])
  branded_products: list[str] = field(default_factory=lambda: ["Chewy Pharmacy"])
  branded_products_categories: list[str] = field(
      default_factory=lambda: ["pet supplies"]
  )
  branded_call_to_actions: list[str] = field(default_factory=lambda: ["shop now"])


class OpenAIServiceStub:
  """Captures detector calls while returning configured API output."""

  def __init__(self, response):
    self.response = response
    self.calls = []

  def evaluate_features(
      self,
      prompt_config,
      preprocess_result,
      model_name,
      schema,
      frame_paths,
      transcript,
      transcript_available,
      frame_evidence,
  ):
    self.calls.append({
        "prompt_config": prompt_config,
        "preprocess_result": preprocess_result,
        "model_name": model_name,
        "schema": schema,
        "frame_paths": frame_paths,
        "transcript": transcript,
        "transcript_available": transcript_available,
        "frame_evidence": frame_evidence,
    })
    return self.response


class OpenAISequenceServiceStub(OpenAIServiceStub):
  """Returns configured API outputs in sequence."""

  def __init__(self, responses):
    super().__init__(responses[0])
    self.responses = list(responses)

  def evaluate_features(self, *args, **kwargs):
    response = self.responses[min(len(self.calls), len(self.responses) - 1)]
    super().evaluate_features(*args, **kwargs)
    return response


def _preprocess_result():
  return models.VideoPreprocessResult(
      source=models.VideoSource("ad.mp4", "ad.mp4", "LOCAL"),
      duration_seconds=15.0,
      full_video_frames=["full-1.jpg", "full-2.jpg"],
      first_5_seconds_frames=["first-1.jpg"],
      audio_path=None,
      transcript="Shop now at Chewy.",
      transcript_available=True,
      full_video_frame_evidence=[
          models.VideoFrameEvidence("full-1.jpg", 0.0),
          models.VideoFrameEvidence("full-2.jpg", 9.5),
      ],
      first_5_seconds_frame_evidence=[
          models.VideoFrameEvidence("first-1.jpg", 2.0),
      ],
      full_video_transcript="Full transcript from dedicated field.",
      first_5_seconds_transcript="First five transcript.",
      first_5_seconds_transcript_available=True,
  )


def _feature(feature_id, video_segment):
  return models.VideoFeature(
      id=feature_id,
      name=f"Feature {feature_id}",
      category=models.VideoFeatureCategory.LONG_FORM_ABCD,
      sub_category=models.VideoFeatureSubCategory.ATTRACT,
      video_segment=video_segment,
      evaluation_criteria="Evaluate the feature.",
      prompt_template="Is this feature present?",
      extra_instructions=[],
      evaluation_method=models.EvaluationMethod.LLMS,
      evaluation_function="",
      include_in_evaluation=True,
      group_by=video_segment,
  )


def _response_for_feature_ids(*feature_ids):
  return {
      "features": [
          {"id": feature_id, "detected": False} for feature_id in feature_ids
      ]
  }


def test_detector_returns_features_from_openai_response():
  """Detector unwraps the features list returned by the OpenAI service."""
  expected_features = [{
      "id": "a_supers",
      "detected": True,
  }]
  service = OpenAIServiceStub({"features": expected_features})

  result = OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=_preprocess_result(),
      feature_configs=[_feature("a_supers", models.VideoSegment.FULL_VIDEO)],
  )

  assert result == expected_features
  assert service.calls[0]["model_name"] == "gpt-test"
  assert service.calls[0]["schema"] == models.OPENAI_VIDEO_RESPONSE_SCHEMA
  assert "Feature ID: a_supers" in service.calls[0]["prompt_config"].prompt


def test_detector_retries_missing_feature_response_once():
  """Incomplete OpenAI feature lists are retried before returning results."""
  expected_features = [
      {"id": "first", "detected": True},
      {"id": "second", "detected": False},
  ]
  service = OpenAISequenceServiceStub([
      {"features": [{"id": "first", "detected": True}]},
      {"features": list(reversed(expected_features))},
  ])

  result = OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=_preprocess_result(),
      feature_configs=[
          _feature("first", models.VideoSegment.FULL_VIDEO),
          _feature("second", models.VideoSegment.FULL_VIDEO),
      ],
  )

  assert result == expected_features
  assert len(service.calls) == 2


def test_detector_raises_when_response_has_no_features_after_retry():
  """Missing feature results fail loudly instead of writing partial output."""
  service = OpenAIServiceStub({})

  with pytest.raises(ValueError, match="missing feature ids: a_supers"):
    OpenAIDetector(service).evaluate_features(
        config=ConfigStub(),
        preprocess_result=_preprocess_result(),
        feature_configs=[_feature("a_supers", models.VideoSegment.FULL_VIDEO)],
    )
  assert len(service.calls) == 2


def test_detector_prompt_preserves_legacy_evaluation_directives():
  """OpenAI prompt keeps the core legacy ABCD evaluation instructions."""
  service = OpenAIServiceStub(_response_for_feature_ids("a_supers"))

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=_preprocess_result(),
      feature_configs=[_feature("a_supers", models.VideoSegment.FULL_VIDEO)],
  )

  system_instructions = service.calls[0]["prompt_config"].system_instructions
  assert "No Hallucination" in system_instructions
  assert "ambiguous or unverifiable" in system_instructions
  assert "must answer false" in system_instructions
  assert "confidence score from 0.0" in system_instructions
  assert "Use timestamps" in system_instructions
  assert "Strict Adherence to Format" in system_instructions
  assert "Feature ID Handling" in system_instructions
  assert "exact, case-sensitive copy" in system_instructions


def test_detector_selects_first_5s_frames_when_all_features_are_first_5s():
  """First-five-second feature groups evaluate only first-five-second frames."""
  service = OpenAIServiceStub(
      _response_for_feature_ids("brand-early", "product-early")
  )
  preprocess_result = _preprocess_result()

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=preprocess_result,
      feature_configs=[
          _feature("brand-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
          _feature("product-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
      ],
  )

  assert service.calls[0]["frame_paths"] == ["first-1.jpg"]


def test_detector_selects_first_5s_evidence_when_all_features_are_first_5s():
  """First-five-second feature groups use first-five-second evidence pack."""
  service = OpenAIServiceStub(
      _response_for_feature_ids("brand-early", "product-early")
  )
  preprocess_result = _preprocess_result()

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=preprocess_result,
      feature_configs=[
          _feature("brand-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
          _feature("product-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
      ],
  )

  assert service.calls[0]["transcript"] == "First five transcript."
  assert service.calls[0]["transcript_available"] is True
  assert service.calls[0]["frame_evidence"] == [
      models.VideoFrameEvidence("first-1.jpg", 2.0),
  ]


def test_detector_selects_explicit_unavailable_first_5s_transcript():
  """First-five-second availability remains false when transcript is absent."""
  service = OpenAIServiceStub(_response_for_feature_ids("brand-early"))
  preprocess_result = _preprocess_result()
  preprocess_result.first_5_seconds_transcript = ""
  preprocess_result.first_5_seconds_transcript_available = False

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=preprocess_result,
      feature_configs=[
          _feature("brand-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
      ],
  )

  assert service.calls[0]["transcript"] == ""
  assert service.calls[0]["transcript_available"] is False


def test_detector_selects_full_frames_when_any_feature_needs_full_video():
  """Mixed feature groups keep full-video frame evidence."""
  service = OpenAIServiceStub(_response_for_feature_ids("brand-early", "supers"))
  preprocess_result = _preprocess_result()

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=preprocess_result,
      feature_configs=[
          _feature("brand-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
          _feature("supers", models.VideoSegment.FULL_VIDEO),
      ],
  )

  assert service.calls[0]["frame_paths"] == ["full-1.jpg", "full-2.jpg"]


def test_detector_selects_full_evidence_when_any_feature_needs_full_video():
  """Mixed feature groups use full-video transcript and frame evidence."""
  service = OpenAIServiceStub(_response_for_feature_ids("brand-early", "supers"))
  preprocess_result = _preprocess_result()

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=preprocess_result,
      feature_configs=[
          _feature("brand-early", models.VideoSegment.FIRST_5_SECS_VIDEO),
          _feature("supers", models.VideoSegment.FULL_VIDEO),
      ],
  )

  assert service.calls[0]["transcript"] == "Full transcript from dedicated field."
  assert service.calls[0]["transcript_available"] is True
  assert service.calls[0]["frame_evidence"] == [
      models.VideoFrameEvidence("full-1.jpg", 0.0),
      models.VideoFrameEvidence("full-2.jpg", 9.5),
  ]


def test_detector_falls_back_to_legacy_transcript_for_full_video_evidence():
  """Full-video evidence falls back to the legacy transcript field."""
  service = OpenAIServiceStub(_response_for_feature_ids("supers"))
  preprocess_result = _preprocess_result()
  preprocess_result.full_video_transcript = ""

  OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=preprocess_result,
      feature_configs=[_feature("supers", models.VideoSegment.FULL_VIDEO)],
  )

  assert service.calls[0]["transcript"] == "Shop now at Chewy."

"""Tests for OpenAI detector."""

from dataclasses import dataclass, field

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
  ):
    self.calls.append({
        "prompt_config": prompt_config,
        "preprocess_result": preprocess_result,
        "model_name": model_name,
        "schema": schema,
        "frame_paths": frame_paths,
    })
    return self.response


def _preprocess_result():
  return models.VideoPreprocessResult(
      source=models.VideoSource("ad.mp4", "ad.mp4", "LOCAL"),
      duration_seconds=15.0,
      full_video_frames=["full-1.jpg", "full-2.jpg"],
      first_5_seconds_frames=["first-1.jpg"],
      audio_path=None,
      transcript="Shop now at Chewy.",
      transcript_available=True,
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


def test_detector_returns_empty_list_when_response_has_no_features():
  """Missing feature results are treated as an empty evaluation result."""
  service = OpenAIServiceStub({})

  result = OpenAIDetector(service).evaluate_features(
      config=ConfigStub(),
      preprocess_result=_preprocess_result(),
      feature_configs=[_feature("a_supers", models.VideoSegment.FULL_VIDEO)],
  )

  assert result == []


def test_detector_selects_first_5s_frames_when_all_features_are_first_5s():
  """First-five-second feature groups evaluate only first-five-second frames."""
  service = OpenAIServiceStub({"features": []})
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


def test_detector_selects_full_frames_when_any_feature_needs_full_video():
  """Mixed feature groups keep full-video frame evidence."""
  service = OpenAIServiceStub({"features": []})
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

#!/usr/bin/env python3

"""Tests for routing OpenAI CLI execution through local preprocessing."""

from pathlib import Path
import subprocess
import sys

import pytest

from configuration import Configuration
import models


def _openai_config():
  config = Configuration()
  config.llm_provider_type = models.LLMProviderType.OPENAI
  config.creative_provider_type = models.CreativeProviderType.LOCAL
  config.extract_brand_metadata = False
  config.use_llms = True
  config.use_annotations = False
  config.run_long_form_abcd = True
  config.run_shorts = False
  config.bq_table_name = ""
  config.brand_name = "Chewy"
  config.brand_variations = ["Chewy"]
  config.branded_products = ["Chewy Pharmacy"]
  config.branded_products_categories = ["pet supplies"]
  config.branded_call_to_actions = ["shop now"]
  config.video_uris = ["sample_videos/ad.mp4"]
  return config


def _preprocess_result(source=None):
  source = source or models.VideoSource(
      original_uri="sample_videos/ad.mp4",
      local_path="sample_videos/ad.mp4",
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  return models.VideoPreprocessResult(
      source=source,
      duration_seconds=12.0,
      full_video_frames=["full-1.jpg"],
      first_5_seconds_frames=["first-1.jpg"],
      audio_path=None,
      transcript="Shop now at Chewy.",
      transcript_available=True,
  )


def test_openai_video_evaluation_maps_detector_dicts(monkeypatch):
  """OpenAI detector results are mapped into FeatureEvaluation objects."""
  from evaluation_services import video_evaluation_service
  from llms_evaluation import openai_api_service
  from llms_evaluation import openai_detector

  detector_instances = []

  class DetectorStub:
    def __init__(self, openai_service):
      self.openai_service = openai_service
      self.calls = []
      detector_instances.append(self)

    def evaluate_features(self, config, preprocess_result, feature_configs):
      self.calls.append((config, preprocess_result, feature_configs))
      if any(feature.id == "a_supers" for feature in feature_configs):
        return [{
            "id": "a_supers",
            "detected": True,
            "confidence_score": 0.91,
            "rationale": "Supers are visible.",
            "evidence": "Frame 1 shows overlaid text.",
            "strengths": "Readable text.",
            "weaknesses": "Brief appearance.",
        }]
      return []

  monkeypatch.setattr(
      openai_api_service, "OpenAIAPIService", lambda **kwargs: object()
  )
  monkeypatch.setattr(openai_detector, "OpenAIDetector", DetectorStub)

  preprocess_result = _preprocess_result()
  evaluations = video_evaluation_service.VideoEvaluationService().evaluate_features(
      config=_openai_config(),
      video_uri="sample_videos/ad.mp4",
      features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
      preprocess_result=preprocess_result,
  )

  assert len(evaluations) == 1
  assert evaluations[0] == models.FeatureEvaluation(
      feature=evaluations[0].feature,
      detected=True,
      confidence_score=0.91,
      rationale="Supers are visible.",
      evidence="Frame 1 shows overlaid text.",
      strengths="Readable text.",
      weaknesses="Brief appearance.",
  )
  assert evaluations[0].feature.id == "a_supers"
  assert detector_instances[0].calls[0][1] is preprocess_result


def test_openai_video_evaluation_uses_configured_frame_limit(monkeypatch):
  """OpenAI evaluation service honors the CLI max_frames setting."""
  from evaluation_services import video_evaluation_service
  from llms_evaluation import openai_api_service
  from llms_evaluation import openai_detector

  service_kwargs = []

  class DetectorStub:
    def __init__(self, openai_service):
      self.openai_service = openai_service

    def evaluate_features(self, config, preprocess_result, feature_configs):
      return []

  def openai_service_factory(**kwargs):
    service_kwargs.append(kwargs)
    return object()

  monkeypatch.setattr(
      openai_api_service, "OpenAIAPIService", openai_service_factory
  )
  monkeypatch.setattr(openai_detector, "OpenAIDetector", DetectorStub)

  config = _openai_config()
  config.max_frames = 100
  video_evaluation_service.VideoEvaluationService().evaluate_features(
      config=config,
      video_uri="sample_videos/ad.mp4",
      features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
      preprocess_result=_preprocess_result(),
  )

  assert service_kwargs == [{"max_frame_count": 100}]


def test_openai_video_evaluation_filters_configured_features(monkeypatch):
  """Feature filters limit OpenAI evaluation to requested feature IDs."""
  from evaluation_services import video_evaluation_service
  from llms_evaluation import openai_api_service
  from llms_evaluation import openai_detector

  evaluated_feature_groups = []

  class DetectorStub:
    def __init__(self, openai_service):
      pass

    def evaluate_features(self, config, preprocess_result, feature_configs):
      evaluated_feature_groups.append([feature.id for feature in feature_configs])
      return [{
          "id": feature_configs[0].id,
          "detected": True,
          "confidence_score": 0.8,
          "rationale": "Filtered feature evaluated.",
          "evidence": "Frame 1.",
          "strengths": "Clear.",
          "weaknesses": "Sampled.",
      }]

  monkeypatch.setattr(
      openai_api_service, "OpenAIAPIService", lambda **kwargs: object()
  )
  monkeypatch.setattr(openai_detector, "OpenAIDetector", DetectorStub)

  config = _openai_config()
  config.features_to_evaluate = ["a_supers"]
  evaluations = video_evaluation_service.VideoEvaluationService().evaluate_features(
      config=config,
      video_uri="sample_videos/ad.mp4",
      features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
      preprocess_result=_preprocess_result(),
  )

  assert evaluated_feature_groups == [["a_supers"]]
  assert [evaluation.feature.id for evaluation in evaluations] == ["a_supers"]


def test_openai_extract_brand_metadata_requires_explicit_metadata():
  """OpenAI path rejects metadata inference until it is implemented."""
  import main

  config = _openai_config()
  config.extract_brand_metadata = True
  config.brand_name = None
  config.brand_variations = []
  config.branded_products = []
  config.branded_products_categories = []

  with pytest.raises(ValueError, match="OpenAI metadata extraction"):
    main.validate_config(config)


def test_openai_video_evaluation_requires_preprocess_result():
  """OpenAI evaluation cannot run without preprocessed local evidence."""
  from evaluation_services import video_evaluation_service

  with pytest.raises(ValueError, match="preprocess_result"):
    video_evaluation_service.VideoEvaluationService().evaluate_features(
        config=_openai_config(),
        video_uri="sample_videos/ad.mp4",
        features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
    )


def test_get_video_sources_openai_uses_local_provider():
  """OpenAI source routing resolves local and YouTube inputs without GCS."""
  import main

  config = _openai_config()
  config.video_uris = [
      "sample_videos/ad.mp4",
      "https://youtu.be/abc123",
  ]

  assert main.get_video_sources(config) == [
      models.VideoSource(
          original_uri="sample_videos/ad.mp4",
          local_path="sample_videos/ad.mp4",
          source_type=models.CreativeProviderType.LOCAL.value,
      ),
      models.VideoSource(
          original_uri="https://youtu.be/abc123",
          local_path="",
          source_type=models.CreativeProviderType.YOUTUBE.value,
      ),
  ]


def test_execute_openai_assessment_returns_video_assessments(monkeypatch):
  """OpenAI CLI execution preprocesses sources and returns assessments."""
  import main
  from evaluation_services import video_evaluation_service
  from llms_evaluation import openai_api_service
  from llms_evaluation import openai_video_preprocessor

  source = models.VideoSource(
      original_uri="sample_videos/ad.mp4",
      local_path="sample_videos/ad.mp4",
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  preprocess_result = _preprocess_result(source)
  feature_eval = models.FeatureEvaluation(
      feature=models.VideoFeature(
          id="a_supers",
          name="Supers",
          category=models.VideoFeatureCategory.LONG_FORM_ABCD,
          sub_category=models.VideoFeatureSubCategory.ATTRACT,
          video_segment=models.VideoSegment.FULL_VIDEO,
          evaluation_criteria="Supers visible",
          prompt_template="Are supers visible?",
          extra_instructions=[],
          evaluation_method=models.EvaluationMethod.LLMS,
          evaluation_function="",
          include_in_evaluation=True,
          group_by=models.VideoSegment.FULL_VIDEO,
      ),
      detected=True,
      confidence_score=0.9,
      rationale="Visible.",
      evidence="Frame 1.",
      strengths="Readable.",
      weaknesses="None.",
  )

  class PreprocessorStub:
    def __init__(
        self,
        cache_dir,
        max_frames,
        frame_sample_rate,
        openai_service,
        transcription_model,
    ):
      self.calls = []

    def preprocess(self, received_source):
      self.calls.append(received_source)
      return preprocess_result

  def evaluate_features(config, video_uri, features_category, preprocess_result):
    assert video_uri == source.original_uri
    assert preprocess_result is not None
    assert features_category == models.VideoFeatureCategory.LONG_FORM_ABCD
    return [feature_eval]

  monkeypatch.setattr(main, "get_video_sources", lambda config: [source])
  monkeypatch.setattr(openai_api_service, "OpenAIAPIService", lambda: object())
  monkeypatch.setattr(
      openai_video_preprocessor, "VideoPreprocessor", PreprocessorStub
  )
  monkeypatch.setattr(
      video_evaluation_service.video_evaluation_service,
      "evaluate_features",
      evaluate_features,
  )

  assessments = main.execute_abcd_assessment_for_videos(_openai_config())

  assert assessments == [
      models.VideoAssessment(
          brand_name="Chewy",
          video_uri="sample_videos/ad.mp4",
          long_form_abcd_evaluated_features=[feature_eval],
          shorts_evaluated_features=[],
          config=assessments[0].config,
      )
  ]


def test_execute_openai_assessment_routes_shorts_preprocess_result(monkeypatch):
  """OpenAI Shorts execution uses the same preprocessed evidence."""
  import main
  from evaluation_services import video_evaluation_service
  from llms_evaluation import openai_api_service
  from llms_evaluation import openai_video_preprocessor

  source = models.VideoSource(
      original_uri="sample_videos/ad.mp4",
      local_path="sample_videos/ad.mp4",
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  preprocess_result = _preprocess_result(source)
  routed_categories = []

  class PreprocessorStub:
    def __init__(
        self,
        cache_dir,
        max_frames,
        frame_sample_rate,
        openai_service,
        transcription_model,
    ):
      pass

    def preprocess(self, received_source):
      assert received_source == source
      return preprocess_result

  def evaluate_features(config, video_uri, features_category, preprocess_result):
    assert video_uri == source.original_uri
    assert preprocess_result is not None
    routed_categories.append(features_category)
    return []

  config = _openai_config()
  config.run_long_form_abcd = False
  config.run_shorts = True

  monkeypatch.setattr(main, "get_video_sources", lambda config: [source])
  monkeypatch.setattr(openai_api_service, "OpenAIAPIService", lambda: object())
  monkeypatch.setattr(
      openai_video_preprocessor, "VideoPreprocessor", PreprocessorStub
  )
  monkeypatch.setattr(
      video_evaluation_service.video_evaluation_service,
      "evaluate_features",
      evaluate_features,
  )

  assessments = main.execute_abcd_assessment_for_videos(config)

  assert routed_categories == [models.VideoFeatureCategory.SHORTS]
  assert assessments[0].shorts_evaluated_features == []


def test_main_help_avoids_google_client_initialization():
  """CLI help should not initialize Google clients or require ADC."""
  repo_root = Path(__file__).resolve().parents[1]

  completed = subprocess.run(
      [sys.executable, "main.py", "--help"],
      cwd=repo_root,
      check=False,
      capture_output=True,
      text=True,
  )

  assert completed.returncode == 0, completed.stderr
  assert "--llm_provider" in completed.stdout


def test_main_cli_failures_exit_nonzero_without_api_key(monkeypatch):
  """Batch CLI failures are visible to automation through a nonzero exit."""
  repo_root = Path(__file__).resolve().parents[1]
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)

  completed = subprocess.run(
      [
          sys.executable,
          "main.py",
          "--llm_provider",
          "OPENAI",
          "--creative_provider_type",
          "LOCAL",
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
          "-run_long_form_abcd",
      ],
      cwd=repo_root,
      check=False,
      capture_output=True,
      text=True,
      env={},
  )

  assert completed.returncode != 0
  assert "OPENAI_API_KEY" in completed.stderr

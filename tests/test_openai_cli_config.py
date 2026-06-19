#!/usr/bin/env python3

"""Tests for OpenAI CLI configuration parsing."""

import sys
import types

from configuration import Configuration
import main
import models
import utils


def test_openai_local_youtube_cli_config():
  """OpenAI CLI options support local and YouTube providers together."""
  args = utils.parse_args([
      "--llm_provider",
      "OPENAI",
      "--video_uris",
      "sample_videos/ad.mp4,, https://www.youtube.com/watch?v=abc123",
      "--creative_provider_type",
      "LOCAL,YOUTUBE",
      "--openai_model",
      "gpt-5.4-mini",
      "--openai_transcription_model",
      "gpt-4o-transcribe",
      "--cache_dir",
      ".cache/custom",
      "--max_frames",
      "12",
      "--frame_sample_rate",
      "0.5",
      "--refresh_cache",
  ])

  config = utils.build_abcd_params_config(args)

  assert config.llm_provider_type == models.LLMProviderType.OPENAI
  assert config.creative_provider_type == models.CreativeProviderType.LOCAL
  assert config.creative_provider_types == [
      models.CreativeProviderType.LOCAL,
      models.CreativeProviderType.YOUTUBE,
  ]
  assert config.openai_model == "gpt-5.4-mini"
  assert config.openai_transcription_model == "gpt-4o-transcribe"
  assert config.cache_dir == ".cache/custom"
  assert config.max_frames == 12
  assert config.frame_sample_rate == 0.5
  assert config.refresh_cache is True
  assert config.video_uris == [
      "sample_videos/ad.mp4",
      "https://www.youtube.com/watch?v=abc123",
  ]


def test_default_cli_config_uses_gemini_gcs_and_safe_optional_values():
  """Legacy defaults remain Gemini and GCS when OpenAI flags are omitted."""
  args = utils.parse_args([])

  config = utils.build_abcd_params_config(args)

  assert config.llm_provider_type == models.LLMProviderType.GEMINI
  assert config.creative_provider_type == models.CreativeProviderType.GCS
  assert config.creative_provider_types == [models.CreativeProviderType.GCS]
  assert config.knowledge_graph_api_key == ""
  assert config.features_to_evaluate == []
  assert config.video_uris == []
  assert config.openai_model == "gpt-5.4-mini"
  assert config.openai_transcription_model == "gpt-4o-transcribe"
  assert config.cache_dir == ".cache/abcds-detector"
  assert config.max_frames == 24
  assert config.frame_sample_rate == 1.0
  assert config.refresh_cache is False
  assert config.llm_params.model_name == "gemini-2.5-pro"
  assert config.llm_params.location == "us-central1"
  assert config.llm_params.generation_config["max_output_tokens"] == 65535
  assert config.llm_params.generation_config["temperature"] == 1
  assert config.llm_params.generation_config["top_p"] == 0.95


def test_openai_cli_refresh_cache_flag_defaults_false():
  """Refresh cache is opt-in so repeated OpenAI runs can reuse artifacts."""
  args = utils.parse_args([])

  config = utils.build_abcd_params_config(args)

  assert config.refresh_cache is False


def test_openai_cli_refresh_cache_flag_sets_config():
  """Refresh cache can be requested from the CLI."""
  args = utils.parse_args(["--refresh_cache"])

  config = utils.build_abcd_params_config(args)

  assert config.refresh_cache is True


def test_refresh_cache_set_parameters_preserves_existing_when_omitted():
  """Refresh cache follows optional config update semantics."""
  config = Configuration()
  config.refresh_cache = True

  config.set_parameters(
      project_id="",
      project_zone="",
      bucket_name="",
      knowledge_graph_api_key="",
      bigquery_dataset="abcd_detector_ds",
      bigquery_table="abcd_assessments",
      assessment_file="",
      use_annotations=False,
      use_llms=True,
      extract_brand_metadata=True,
      run_long_form_abcd=True,
      run_shorts=True,
      features_to_evaluate=None,
      creative_provider_type=None,
      verbose=False,
  )

  assert config.refresh_cache is True


def test_blank_video_uris_are_ignored():
  """Blank video URI values do not create truthy work lists."""
  args = utils.parse_args(["--video_uris", " , ,, "])

  config = utils.build_abcd_params_config(args)

  assert config.video_uris == []


def test_features_to_evaluate_long_alias_parses():
  """Documented double-dash feature filter alias is supported."""
  args = utils.parse_args(["--features_to_evaluate", "a_supers,b_brand_visuals"])

  config = utils.build_abcd_params_config(args)

  assert config.features_to_evaluate == ["a_supers", "b_brand_visuals"]


def test_legacy_video_and_provider_aliases_still_parse():
  """Legacy single-dash and short provider aliases remain supported."""
  legacy_long_args = utils.parse_args([
      "-video_uris",
      "gs://bucket/video.mp4",
      "-creative_provider_type",
      "GCS",
  ])
  legacy_short_args = utils.parse_args([
      "-vu",
      "https://www.youtube.com/watch?v=abc123",
      "-crpt",
      "YOUTUBE",
  ])

  legacy_long_config = utils.build_abcd_params_config(legacy_long_args)
  legacy_short_config = utils.build_abcd_params_config(legacy_short_args)

  assert legacy_long_config.video_uris == ["gs://bucket/video.mp4"]
  assert legacy_long_config.creative_provider_type == (
      models.CreativeProviderType.GCS
  )
  assert legacy_short_config.video_uris == [
      "https://www.youtube.com/watch?v=abc123"
  ]
  assert legacy_short_config.creative_provider_type == (
      models.CreativeProviderType.YOUTUBE
  )


def test_openai_video_response_schema_wraps_feature_schema():
  """OpenAI schema wraps feature responses under a required features key."""
  assert models.OPENAI_VIDEO_RESPONSE_SCHEMA == {
      "type": "object",
      "properties": {
          "features": models.VIDEO_RESPONSE_SCHEMA,
      },
      "required": ["features"],
      "additionalProperties": False,
  }


def test_video_preprocess_result_tracks_source_and_outputs():
  """Video preprocessing results keep source, frame, audio, and transcript data."""
  source = models.VideoSource(
      original_uri="https://www.youtube.com/watch?v=abc123",
      local_path=".cache/video.mp4",
      source_type=models.CreativeProviderType.YOUTUBE.value,
  )

  result = models.VideoPreprocessResult(
      source=source,
      duration_seconds=12.5,
      full_video_frames=["frame-1.jpg"],
      first_5_seconds_frames=["frame-0.jpg"],
      audio_path=None,
      transcript="",
      transcript_available=False,
  )

  assert result.source == source
  assert result.duration_seconds == 12.5
  assert result.audio_path is None
  assert not result.transcript_available


def test_video_preprocess_result_tracks_timestamped_evidence():
  """Video preprocessing results can carry frame metadata and first-5 transcript."""
  source = models.VideoSource(
      original_uri="ad.mp4",
      local_path="ad.mp4",
      source_type=models.CreativeProviderType.LOCAL.value,
  )
  full_frame = models.VideoFrameEvidence(
      path="frames/full/frame_0001.jpg",
      timestamp_seconds=0.0,
  )
  first_frame = models.VideoFrameEvidence(
      path="frames/first_5s/frame_0001.jpg",
      timestamp_seconds=0.0,
  )

  result = models.VideoPreprocessResult(
      source=source,
      duration_seconds=12.5,
      full_video_frames=[full_frame.path],
      first_5_seconds_frames=[first_frame.path],
      audio_path="audio.mp3",
      transcript="full transcript",
      transcript_available=True,
      full_video_frame_evidence=[full_frame],
      first_5_seconds_frame_evidence=[first_frame],
      first_5_seconds_transcript="first five transcript",
  )

  assert result.full_video_frame_evidence == [full_frame]
  assert result.first_5_seconds_frame_evidence == [first_frame]
  assert result.full_video_transcript == ""
  assert result.first_5_seconds_transcript == "first five transcript"


def test_openai_preprocessor_builder_passes_refresh_cache(monkeypatch):
  """OpenAI preprocessor builder wires refresh cache without extra setup."""
  fake_openai_api_service = types.SimpleNamespace(
      OpenAIAPIService=lambda: object()
  )
  monkeypatch.setitem(
      sys.modules,
      "llms_evaluation.openai_api_service",
      fake_openai_api_service,
  )
  config = Configuration()
  config.refresh_cache = True

  preprocessor = main._build_openai_preprocessor(config)

  assert preprocessor.refresh_cache is True

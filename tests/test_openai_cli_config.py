#!/usr/bin/env python3

"""Tests for OpenAI CLI configuration parsing."""

import models
import utils


def test_openai_local_youtube_cli_config():
  """OpenAI CLI options support local and YouTube providers together."""
  args = utils.parse_args([
      "--llm_provider",
      "OPENAI",
      "-video_uris",
      "sample_videos/ad.mp4,https://www.youtube.com/watch?v=abc123",
      "-creative_provider_type",
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
  assert config.openai_model == "gpt-5.4-mini"
  assert config.openai_transcription_model == "gpt-4o-transcribe"
  assert config.cache_dir == ".cache/abcds-detector"
  assert config.max_frames == 24
  assert config.frame_sample_rate == 1.0
  assert config.llm_params.model_name == "gemini-2.5-pro"
  assert config.llm_params.location == "us-central1"
  assert config.llm_params.generation_config["max_output_tokens"] == 65535
  assert config.llm_params.generation_config["temperature"] == 1
  assert config.llm_params.generation_config["top_p"] == 0.95


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

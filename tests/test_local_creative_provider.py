#!/usr/bin/env python3

"""Tests for local and YouTube creative source resolution."""

import importlib
import sys
import types
from types import SimpleNamespace

import models
from creative_providers import local_creative_provider


def test_is_youtube_url_matches_supported_youtube_hosts():
  """YouTube watch and short URLs are detected."""
  assert local_creative_provider.is_youtube_url(
      "https://www.youtube.com/watch?v=abc"
  )
  assert local_creative_provider.is_youtube_url(
      "https://youtube.com/watch?v=abc"
  )
  assert local_creative_provider.is_youtube_url(
      "https://m.youtube.com/watch?v=abc"
  )
  assert local_creative_provider.is_youtube_url("https://youtu.be/abc")


def test_is_youtube_url_rejects_local_paths_and_other_hosts():
  """Local paths and non-YouTube URLs are not treated as YouTube sources."""
  assert not local_creative_provider.is_youtube_url("sample_videos/ad.mp4")
  assert not local_creative_provider.is_youtube_url("/tmp/ad.mp4")
  assert not local_creative_provider.is_youtube_url(
      "https://example.com/watch?v=abc"
  )
  assert not local_creative_provider.is_youtube_url(
      "https://notyoutube.com/watch?v=abc"
  )


def test_local_creative_provider_returns_config_video_uris():
  """URI resolution remains compatible with existing provider callers."""
  config = SimpleNamespace(video_uris=[
      "sample_videos/ad.mp4",
      "https://www.youtube.com/watch?v=abc",
  ])
  provider = local_creative_provider.LocalCreativeProvider()

  assert provider.get_creative_uris(config) is config.video_uris


def test_local_creative_provider_builds_typed_video_sources():
  """Local paths and YouTube URLs produce distinct VideoSource entries."""
  config = SimpleNamespace(video_uris=[
      "sample_videos/ad.mp4",
      "https://youtu.be/abc",
  ])
  provider = local_creative_provider.LocalCreativeProvider()

  assert provider.get_creative_sources(config) == [
      models.VideoSource(
          original_uri="sample_videos/ad.mp4",
          local_path="sample_videos/ad.mp4",
          source_type=models.CreativeProviderType.LOCAL.value,
      ),
      models.VideoSource(
          original_uri="https://youtu.be/abc",
          local_path="",
          source_type=models.CreativeProviderType.YOUTUBE.value,
      ),
  ]


def test_creative_provider_registry_returns_local_provider(monkeypatch):
  """The global creative provider registry includes LOCAL."""
  gcs_service_module = types.SimpleNamespace(gcs_api_service=object())
  monkeypatch.setitem(
      sys.modules, "gcp_api_services.gcs_api_service", gcs_service_module
  )
  sys.modules.pop("creative_providers.creative_provider_registry", None)

  creative_provider_registry = importlib.import_module(
      "creative_providers.creative_provider_registry"
  )
  provider = creative_provider_registry.provider_factory.get_provider(
      models.CreativeProviderType.LOCAL.value
  )

  assert isinstance(provider, local_creative_provider.LocalCreativeProvider)

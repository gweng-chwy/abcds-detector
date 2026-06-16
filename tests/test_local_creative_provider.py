#!/usr/bin/env python3

"""Tests for local and YouTube creative source resolution."""

from contextlib import contextmanager
import importlib
import sys
import types
from types import SimpleNamespace

import models


_MISSING = object()


@contextmanager
def _preserve_module_cache(*module_names):
  """Restore selected module cache entries after import-isolation tests."""
  saved_modules = {
      module_name: sys.modules.get(module_name, _MISSING)
      for module_name in module_names
  }
  saved_parent_attrs = {}
  for module_name in module_names:
    parent_name, _, attribute_name = module_name.rpartition(".")
    parent_module = sys.modules.get(parent_name)
    if parent_module is not None and attribute_name:
      saved_parent_attrs[(parent_name, attribute_name)] = getattr(
          parent_module, attribute_name, _MISSING
      )

  try:
    yield
  finally:
    for module_name, module in saved_modules.items():
      if module is _MISSING:
        sys.modules.pop(module_name, None)
      else:
        sys.modules[module_name] = module

    for (parent_name, attribute_name), value in saved_parent_attrs.items():
      parent_module = sys.modules.get(parent_name)
      if parent_module is None:
        continue

      if value is _MISSING:
        if hasattr(parent_module, attribute_name):
          delattr(parent_module, attribute_name)
      else:
        setattr(parent_module, attribute_name, value)


def _import_local_creative_provider():
  return importlib.import_module("creative_providers.local_creative_provider")


def test_local_creative_provider_import_does_not_import_configuration():
  """Importing the local provider does not trigger configuration side effects."""
  with _preserve_module_cache(
      "configuration", "creative_providers.local_creative_provider"
  ):
    sys.modules.pop("configuration", None)
    sys.modules.pop("creative_providers.local_creative_provider", None)

    _import_local_creative_provider()

    assert "configuration" not in sys.modules


def test_is_youtube_url_matches_supported_youtube_hosts():
  """YouTube watch and short URLs are detected."""
  local_creative_provider = _import_local_creative_provider()

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
  local_creative_provider = _import_local_creative_provider()

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
  local_creative_provider = _import_local_creative_provider()
  config = SimpleNamespace(video_uris=[
      "sample_videos/ad.mp4",
      "https://www.youtube.com/watch?v=abc",
  ])
  provider = local_creative_provider.LocalCreativeProvider()

  assert provider.get_creative_uris(config) is config.video_uris


def test_local_creative_provider_builds_typed_video_sources():
  """Local paths and YouTube URLs produce distinct VideoSource entries."""
  local_creative_provider = _import_local_creative_provider()
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
  local_creative_provider = _import_local_creative_provider()
  with _preserve_module_cache(
      "gcp_api_services",
      "gcp_api_services.gcs_api_service",
      "creative_providers.gcs_creative_provider",
      "creative_providers.creative_provider_registry",
  ):
    gcs_service_module = types.ModuleType("gcp_api_services.gcs_api_service")
    gcs_service_module.gcs_api_service = object()
    monkeypatch.setitem(
        sys.modules, "gcp_api_services.gcs_api_service", gcs_service_module
    )
    sys.modules.pop("creative_providers.gcs_creative_provider", None)
    sys.modules.pop("creative_providers.creative_provider_registry", None)

    creative_provider_registry = importlib.import_module(
        "creative_providers.creative_provider_registry"
    )
    provider = creative_provider_registry.provider_factory.get_provider(
        models.CreativeProviderType.LOCAL.value
    )

  assert isinstance(provider, local_creative_provider.LocalCreativeProvider)

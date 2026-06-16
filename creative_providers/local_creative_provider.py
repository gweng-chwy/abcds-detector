"""Creative provider for local files and direct YouTube URLs."""

from urllib import parse

import configuration
import models


_YOUTUBE_HOSTS = frozenset({
    "www.youtube.com",
    "youtube.com",
    "m.youtube.com",
    "youtu.be",
})


def is_youtube_url(uri: str) -> bool:
  """Return whether the URI points to a supported YouTube host."""
  parsed_uri = parse.urlparse(uri.strip())
  if parsed_uri.scheme not in ("http", "https"):
    return False

  return (parsed_uri.hostname or "").lower() in _YOUTUBE_HOSTS


class LocalCreativeProvider:
  """Creative provider for local paths and YouTube URLs."""

  def __init__(self):
    pass

  def get_creative_uris(self, config: configuration.Configuration) -> list[str]:
    """Return configured video URIs for compatibility with legacy callers."""
    return config.video_uris

  def get_creative_sources(
      self, config: configuration.Configuration
  ) -> list[models.VideoSource]:
    """Resolve configured URIs into typed video source objects."""
    sources = []
    for uri in config.video_uris:
      if is_youtube_url(uri):
        sources.append(
            models.VideoSource(
                original_uri=uri,
                local_path="",
                source_type=models.CreativeProviderType.YOUTUBE.value,
            )
        )
      else:
        sources.append(
            models.VideoSource(
                original_uri=uri,
                local_path=uri,
                source_type=models.CreativeProviderType.LOCAL.value,
            )
        )

    return sources

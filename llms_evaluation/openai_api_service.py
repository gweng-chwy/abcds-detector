"""OpenAI API service for video creative evaluation."""

import base64
import copy
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

import models


DEFAULT_MAX_FRAME_COUNT = 24
DEFAULT_MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
})


def to_openai_strict_schema(schema: dict) -> dict:
  """Return an OpenAI strict-structured-output compatible schema copy."""
  strict_schema = copy.deepcopy(schema)
  _make_schema_objects_strict(strict_schema)
  return strict_schema


def _make_schema_objects_strict(schema_fragment) -> None:
  if isinstance(schema_fragment, dict):
    properties = schema_fragment.get("properties")
    if isinstance(properties, dict):
      schema_fragment["additionalProperties"] = False
      schema_fragment["required"] = list(properties.keys())
      for property_schema in properties.values():
        _make_schema_objects_strict(property_schema)
    elif schema_fragment.get("type") == "object":
      schema_fragment["additionalProperties"] = False
      schema_fragment.setdefault("required", [])

    if "items" in schema_fragment:
      _make_schema_objects_strict(schema_fragment["items"])

    for key in ("anyOf", "allOf", "oneOf"):
      if key in schema_fragment:
        _make_schema_objects_strict(schema_fragment[key])
    if "not" in schema_fragment:
      _make_schema_objects_strict(schema_fragment["not"])
    return

  if isinstance(schema_fragment, (list, tuple)):
    for item in schema_fragment:
      _make_schema_objects_strict(item)


class OpenAIAPIService:
  """Wrapper around OpenAI APIs used by the OpenAI evaluation path."""

  def __init__(
      self,
      client=None,
      max_frame_count: int = DEFAULT_MAX_FRAME_COUNT,
      max_total_image_bytes: int = DEFAULT_MAX_TOTAL_IMAGE_BYTES,
      allowed_image_mime_types: frozenset[str] = ALLOWED_IMAGE_MIME_TYPES,
  ):
    """Initialize the service with an injected or real OpenAI client."""
    if client is None and not os.environ.get("OPENAI_API_KEY"):
      raise ValueError("OPENAI_API_KEY is required for llm_provider=OPENAI")
    self.client = client or OpenAI()
    self.max_frame_count = max_frame_count
    self.max_total_image_bytes = max_total_image_bytes
    self.allowed_image_mime_types = allowed_image_mime_types

  def transcribe_audio(
      self,
      audio_path: str,
      model_name: str = "gpt-4o-transcribe",
  ) -> str:
    """Transcribe audio file with OpenAI."""
    with open(audio_path, "rb") as audio_file:
      transcription = self.client.audio.transcriptions.create(
          model=model_name,
          file=audio_file,
      )
    return getattr(transcription, "text", "") or ""

  def build_input(
      self,
      prompt_config: models.PromptConfig,
      transcript: str,
      frame_paths: list[str],
      duration_seconds: float,
  ) -> list[dict]:
    """Build Responses API input from prompt, transcript, and images."""
    image_inputs = self._build_image_inputs(frame_paths)
    user_content = [{
        "type": "input_text",
        "text": (
            f"Video duration seconds: {duration_seconds}\n"
            f"Transcript:\n{transcript or '[no transcript available]'}\n\n"
            f"{prompt_config.prompt}"
        ),
    }]
    for image_input in image_inputs:
      user_content.append({
          "type": "input_image",
          "image_url": image_input,
      })

    return [
        {
            "role": "system",
            "content": [{
                "type": "input_text",
                "text": prompt_config.system_instructions,
            }],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

  def evaluate_features(
      self,
      prompt_config: models.PromptConfig,
      preprocess_result: models.VideoPreprocessResult,
      model_name: str,
      schema: dict,
      frame_paths: list[str],
  ) -> dict:
    """Evaluate video features with sampled frames and transcript."""
    response = self.client.responses.create(
        model=model_name,
        input=self.build_input(
            prompt_config=prompt_config,
            transcript=preprocess_result.transcript,
            frame_paths=frame_paths,
            duration_seconds=preprocess_result.duration_seconds,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "abcd_feature_evaluation",
                "strict": True,
                "schema": to_openai_strict_schema(schema),
            }
        },
    )
    return self.parse_json_response(response)

  def parse_json_response(self, response) -> dict:
    """Parse the JSON object returned by OpenAI structured outputs."""
    output_text = getattr(response, "output_text", None)
    if output_text:
      return self._loads_json(output_text)

    output_text = self._find_text_in_output(getattr(response, "output", None))
    if output_text:
      return self._loads_json(output_text)

    raise ValueError("No JSON text found in OpenAI response")

  def _build_image_inputs(self, frame_paths: list[str]) -> list[str]:
    if len(frame_paths) > self.max_frame_count:
      raise ValueError(
          f"Frame count {len(frame_paths)} exceeds maximum "
          f"{self.max_frame_count}"
      )

    image_inputs = []
    total_image_bytes = 0
    for frame_path in frame_paths:
      path = Path(frame_path)
      mime_type = mimetypes.guess_type(frame_path)[0]
      if mime_type not in self.allowed_image_mime_types:
        raise ValueError(
            f"Unsupported image MIME type {mime_type or 'unknown'} for "
            f"{frame_path}"
        )

      image_size = path.stat().st_size
      total_image_bytes += image_size
      if total_image_bytes > self.max_total_image_bytes:
        raise ValueError(
            f"Total image bytes {total_image_bytes} exceeds maximum "
            f"{self.max_total_image_bytes}"
        )
      image_bytes = path.read_bytes()
      image_inputs.append(self._image_data_url(mime_type, image_bytes))

    return image_inputs

  def _image_data_url(self, mime_type: str, image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

  def _loads_json(self, output_text: str) -> dict:
    try:
      parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
      raise ValueError("OpenAI response did not contain valid JSON") from exc
    if not isinstance(parsed, dict):
      raise ValueError("OpenAI JSON response must be an object")
    return parsed

  def _find_text_in_output(self, output: Any) -> str:
    if output is None:
      return ""

    if isinstance(output, str):
      return output

    if isinstance(output, dict):
      direct_text = output.get("text")
      if isinstance(direct_text, str) and direct_text:
        return direct_text
      content_text = self._find_text_in_output(output.get("content"))
      if content_text:
        return content_text
      for value in output.values():
        nested_text = self._find_text_in_output(value)
        if nested_text:
          return nested_text
      return ""

    if isinstance(output, (list, tuple)):
      for item in output:
        nested_text = self._find_text_in_output(item)
        if nested_text:
          return nested_text
      return ""

    direct_text = getattr(output, "text", None)
    if isinstance(direct_text, str) and direct_text:
      return direct_text

    return self._find_text_in_output(getattr(output, "content", None))

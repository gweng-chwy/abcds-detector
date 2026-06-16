"""Tests for OpenAI API service."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import models
from llms_evaluation.openai_api_service import OpenAIAPIService


def test_missing_api_key_raises_without_injected_client(monkeypatch):
  """The real OpenAI client path requires OPENAI_API_KEY."""
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)

  with pytest.raises(ValueError, match="OPENAI_API_KEY"):
    OpenAIAPIService()


def test_injected_client_bypasses_api_key_requirement(monkeypatch):
  """Tests can inject a client without configuring environment secrets."""
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  client = mock.Mock()

  service = OpenAIAPIService(client=client)

  assert service.client is client


def test_transcribe_audio_uses_configured_model_and_returns_text(tmp_path):
  """Audio transcription sends the requested model and returns SDK text."""
  audio_path = tmp_path / "audio.mp3"
  audio_path.write_bytes(b"audio")
  client = mock.Mock()
  captured_file_bytes = {}

  def fake_create(model, file):
    captured_file_bytes["model"] = model
    captured_file_bytes["content"] = file.read()
    return SimpleNamespace(text="spoken words")

  client.audio.transcriptions.create.side_effect = fake_create
  service = OpenAIAPIService(client=client)

  result = service.transcribe_audio(
      str(audio_path),
      model_name="custom-transcribe",
  )

  assert result == "spoken words"
  assert captured_file_bytes == {
      "model": "custom-transcribe",
      "content": b"audio",
  }


def test_build_input_includes_text_and_image_data_url(tmp_path):
  """Responses input contains text context and encoded frame evidence."""
  frame_path = tmp_path / "frame.jpg"
  frame_path.write_bytes(b"image")
  service = OpenAIAPIService(client=mock.Mock())

  payload = service.build_input(
      prompt_config=models.PromptConfig(
          prompt="Question?",
          system_instructions="System instructions",
      ),
      transcript="spoken words",
      frame_paths=[str(frame_path)],
      duration_seconds=15.0,
  )

  expected_data_url = (
      "data:image/jpeg;base64,"
      f"{base64.b64encode(b'image').decode('utf-8')}"
  )
  assert payload[0] == {
      "role": "system",
      "content": [{"type": "input_text", "text": "System instructions"}],
  }
  assert payload[1]["role"] == "user"
  assert payload[1]["content"][0]["type"] == "input_text"
  assert "Video duration seconds: 15.0" in payload[1]["content"][0]["text"]
  assert "Transcript:\nspoken words" in payload[1]["content"][0]["text"]
  assert "Question?" in payload[1]["content"][0]["text"]
  assert payload[1]["content"][1] == {
      "type": "input_image",
      "image_url": expected_data_url,
      "detail": "auto",
  }


def test_evaluate_features_passes_model_schema_and_parses_json(tmp_path):
  """Feature evaluation uses structured outputs and returns parsed JSON."""
  frame_path = tmp_path / "frame.jpg"
  frame_path.write_bytes(b"image")
  client = mock.Mock()
  client.responses.create.return_value = SimpleNamespace(
      output_text=json.dumps({"features": [{"id": "feature-1"}]})
  )
  service = OpenAIAPIService(client=client)
  preprocess_result = models.VideoPreprocessResult(
      source=models.VideoSource("ad.mp4", "ad.mp4", "LOCAL"),
      duration_seconds=9.5,
      full_video_frames=[],
      first_5_seconds_frames=[],
      audio_path=None,
      transcript="hello",
      transcript_available=True,
  )
  schema = {"type": "object", "properties": {"features": {"type": "array"}}}

  result = service.evaluate_features(
      prompt_config=models.PromptConfig(
          prompt="Question",
          system_instructions="System",
      ),
      preprocess_result=preprocess_result,
      model_name="gpt-test",
      schema=schema,
      frame_paths=[str(frame_path)],
  )

  assert result == {"features": [{"id": "feature-1"}]}
  create_call = client.responses.create.call_args
  assert create_call.kwargs["model"] == "gpt-test"
  assert create_call.kwargs["text"]["format"] == {
      "type": "json_schema",
      "name": "abcd_feature_evaluation",
      "strict": True,
      "schema": {
          "type": "object",
          "properties": {"features": {"type": "array"}},
          "required": ["features"],
          "additionalProperties": False,
      },
  }
  assert create_call.kwargs["input"][0]["role"] == "system"


def test_evaluate_features_adapts_nested_schema_for_openai_strict_outputs():
  """Strict OpenAI schemas require closed objects and all properties required."""
  client = mock.Mock()
  client.responses.create.return_value = SimpleNamespace(
      output_text=json.dumps({"features": []})
  )
  service = OpenAIAPIService(client=client)
  schema = {
      "type": "object",
      "properties": {
          "features": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "id": {"type": "string"},
                      "detected": {"type": "boolean"},
                  },
                  "required": ["id"],
              },
          },
      },
      "required": [],
  }
  preprocess_result = models.VideoPreprocessResult(
      source=models.VideoSource("ad.mp4", "ad.mp4", "LOCAL"),
      duration_seconds=9.5,
      full_video_frames=[],
      first_5_seconds_frames=[],
      audio_path=None,
      transcript="hello",
      transcript_available=True,
  )

  service.evaluate_features(
      prompt_config=models.PromptConfig(
          prompt="Question",
          system_instructions="System",
      ),
      preprocess_result=preprocess_result,
      model_name="gpt-test",
      schema=schema,
      frame_paths=[],
  )

  strict_schema = client.responses.create.call_args.kwargs["text"]["format"][
      "schema"
  ]
  feature_item_schema = strict_schema["properties"]["features"]["items"]
  assert strict_schema["additionalProperties"] is False
  assert strict_schema["required"] == ["features"]
  assert feature_item_schema["additionalProperties"] is False
  assert feature_item_schema["required"] == ["id", "detected"]
  assert "additionalProperties" not in schema
  assert schema["required"] == []
  assert schema["properties"]["features"]["items"]["required"] == ["id"]


def test_openai_video_response_schema_is_adapted_for_strict_outputs(tmp_path):
  """The existing model schema is closed deeply before sending to OpenAI."""
  frame_path = tmp_path / "frame.jpg"
  frame_path.write_bytes(b"image")
  client = mock.Mock()
  client.responses.create.return_value = SimpleNamespace(
      output_text=json.dumps({"features": []})
  )
  service = OpenAIAPIService(client=client)
  preprocess_result = models.VideoPreprocessResult(
      source=models.VideoSource("ad.mp4", "ad.mp4", "LOCAL"),
      duration_seconds=9.5,
      full_video_frames=[],
      first_5_seconds_frames=[],
      audio_path=None,
      transcript="hello",
      transcript_available=True,
  )

  service.evaluate_features(
      prompt_config=models.PromptConfig(
          prompt="Question",
          system_instructions="System",
      ),
      preprocess_result=preprocess_result,
      model_name="gpt-test",
      schema=models.OPENAI_VIDEO_RESPONSE_SCHEMA,
      frame_paths=[str(frame_path)],
  )

  strict_schema = client.responses.create.call_args.kwargs["text"]["format"][
      "schema"
  ]
  feature_item_schema = strict_schema["properties"]["features"]["items"]
  feature_property_keys = list(feature_item_schema["properties"].keys())
  assert feature_item_schema["additionalProperties"] is False
  assert feature_item_schema["required"] == feature_property_keys
  assert models.OPENAI_VIDEO_RESPONSE_SCHEMA["properties"]["features"]["items"].get(
      "additionalProperties"
  ) is None


def test_build_input_rejects_too_many_frames(tmp_path):
  """Frame count is capped before base64 encoding."""
  frame_paths = []
  for index in range(2):
    frame_path = tmp_path / f"frame-{index}.jpg"
    frame_path.write_bytes(b"image")
    frame_paths.append(str(frame_path))
  service = OpenAIAPIService(client=mock.Mock(), max_frame_count=1)

  with pytest.raises(ValueError, match="Frame count .* exceeds maximum 1"):
    service.build_input(
        prompt_config=models.PromptConfig(
            prompt="Question?",
            system_instructions="System instructions",
        ),
        transcript="spoken words",
        frame_paths=frame_paths,
        duration_seconds=15.0,
    )


def test_build_input_rejects_total_image_bytes_over_cap(tmp_path, monkeypatch):
  """Total image evidence size is capped before request construction."""
  frame_path = tmp_path / "frame.jpg"
  frame_path.write_bytes(b"image")
  service = OpenAIAPIService(client=mock.Mock(), max_total_image_bytes=4)

  def fail_read_bytes(self):
    raise AssertionError("oversized image should not be read")

  monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

  with pytest.raises(ValueError, match="Total image bytes .* exceeds maximum 4"):
    service.build_input(
        prompt_config=models.PromptConfig(
            prompt="Question?",
            system_instructions="System instructions",
        ),
        transcript="spoken words",
        frame_paths=[str(frame_path)],
        duration_seconds=15.0,
    )


def test_build_input_rejects_unsupported_image_mime_type(tmp_path):
  """Only image MIME types supported by the OpenAI payload are allowed."""
  frame_path = tmp_path / "frame.gif"
  frame_path.write_bytes(b"image")
  service = OpenAIAPIService(client=mock.Mock())

  with pytest.raises(ValueError, match="Unsupported image MIME type image/gif"):
    service.build_input(
        prompt_config=models.PromptConfig(
            prompt="Question?",
            system_instructions="System instructions",
        ),
        transcript="spoken words",
        frame_paths=[str(frame_path)],
        duration_seconds=15.0,
    )


def test_parse_json_response_handles_common_output_content_shape():
  """Parser accepts SDK responses that expose JSON through output content."""
  service = OpenAIAPIService(client=mock.Mock())
  response = SimpleNamespace(
      output=[
          SimpleNamespace(
              content=[
                  SimpleNamespace(
                      type="output_text",
                      text=json.dumps({"features": [{"id": "feature-1"}]}),
                  )
              ]
          )
      ]
  )

  assert service.parse_json_response(response) == {
      "features": [{"id": "feature-1"}]
  }


def test_parse_json_response_raises_clear_error_when_text_missing():
  """Missing text in an SDK response produces a clear local error."""
  service = OpenAIAPIService(client=mock.Mock())

  with pytest.raises(ValueError, match="No JSON text found"):
    service.parse_json_response(SimpleNamespace(output=[]))

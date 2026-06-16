"""Tests for OpenAI API service."""

import base64
import json
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
      "schema": schema,
  }
  assert create_call.kwargs["input"][0]["role"] == "system"


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

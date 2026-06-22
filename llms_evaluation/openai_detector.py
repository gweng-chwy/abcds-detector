"""OpenAI detector facade for ABCD feature evaluation."""

import models


_CALL_TO_ACTIONS = [
    "LEARN MORE",
    "GET QUOTE",
    "APPLY NOW",
    "SIGN UP",
    "CONTACT US",
    "SUBSCRIBE",
    "DOWNLOAD",
    "BOOK NOW",
    "SHOP NOW",
    "BUY NOW",
    "DONATE NOW",
    "ORDER NOW",
    "PLAY NOW",
    "SEE MORE",
    "START NOW",
    "VISIT SITE",
    "WATCH NOW",
]
_MAX_FEATURE_EVALUATION_ATTEMPTS = 2

_SYSTEM_INSTRUCTIONS = """
You are an AI Video Analysis Engine. Your primary function is to act as a
meticulous and objective creative expert. Your goal is to analyze video ad
content and answer questions about specific features within the video. Your
analysis must be rigorously based only on the visual information in the sampled
frames and the auditory information in the transcript.

## CORE DIRECTIVES

- Absolute Objectivity: Your analysis must be based exclusively on concrete
  evidence from the video. Do not infer, assume, or use external knowledge. If
  you cannot see or hear it in the supplied evidence, it did not happen.
- No Hallucination: Do not invent information. If a feature is ambiguous or
  unverifiable from the supplied frames and transcript, you must answer false
  and explain why it is ambiguous or unverifiable.
- Strict Adherence to Format: The output format is non-negotiable. Return only
  JSON that conforms to the provided schema.
- Confidence Scoring: For each feature, calculate a confidence score from 0.0
  to 1.0. Base it on the clarity and visibility of relevant evidence, absence
  of occlusions or ambiguity, the strengths and weaknesses you identify, and
  the robustness of your analysis. Output only the numerical score as a float.

## STEP-BY-STEP TASK EXECUTION

- Receive Input: You will be given sampled video frames, transcript text, video
  duration, and a list of feature questions to answer.
- Analyze Evidence: Conduct a thorough analysis of the sampled visual frames
  and the full transcript. Do not use evidence outside those inputs.
- Evaluate Each Question: For each question, determine a definitive boolean
  answer. The answer must be true if the statement is verifiably correct based
  on the supplied evidence. The answer must be false if the statement is
  verifiably incorrect or cannot be verified from the supplied evidence.
- Formulate Explanation: For each answer, write a detailed and logically sound
  rationale. Cite specific visual or transcript evidence. Use timestamps
  whenever possible, such as from 0:15 to 0:22 or at 0:08.
- Construct Final Output: Assemble all answers and explanations into the
  specified JSON schema. Populate every required output field for every feature.
- Feature ID Handling: Use an exact, case-sensitive copy of the Feature ID
  provided in the input prompt. Preserve the original value exactly. Evaluation
  will fail if the id is missing or does not match exactly.
"""


class OpenAIDetector:
  """Evaluates ABCD features through OpenAI."""

  def __init__(self, openai_service):
    self.openai_service = openai_service

  def evaluate_features(
      self,
      config,
      preprocess_result: models.VideoPreprocessResult,
      feature_configs: list[models.VideoFeature],
  ) -> list[dict]:
    """Evaluate features and return OpenAI feature dictionaries."""
    prompt_config = self._build_prompt_config(feature_configs, config)
    evidence_pack = self._select_evidence_pack(
        preprocess_result,
        feature_configs,
    )
    validation_error = None
    for _ in range(_MAX_FEATURE_EVALUATION_ATTEMPTS):
      response = self.openai_service.evaluate_features(
          prompt_config=prompt_config,
          preprocess_result=preprocess_result,
          model_name=config.openai_model,
          schema=models.OPENAI_VIDEO_RESPONSE_SCHEMA,
          frame_paths=evidence_pack["frame_paths"],
          transcript=evidence_pack["transcript"],
          transcript_available=evidence_pack["transcript_available"],
          frame_evidence=evidence_pack["frame_evidence"],
      )
      try:
        return self._validate_feature_response(
            response.get("features", []),
            feature_configs,
        )
      except ValueError as exc:
        validation_error = exc

    raise validation_error

  def _select_evidence_pack(
      self,
      preprocess_result: models.VideoPreprocessResult,
      feature_configs: list[models.VideoFeature],
  ) -> dict:
    if all(
        feature.video_segment == models.VideoSegment.FIRST_5_SECS_VIDEO
        for feature in feature_configs
    ):
      return {
          "frame_paths": preprocess_result.first_5_seconds_frames,
          "transcript": preprocess_result.first_5_seconds_transcript,
          "transcript_available": (
              preprocess_result.first_5_seconds_transcript_available
          ),
          "frame_evidence": preprocess_result.first_5_seconds_frame_evidence,
      }

    return {
        "frame_paths": preprocess_result.full_video_frames,
        "transcript": (
            preprocess_result.full_video_transcript
            or preprocess_result.transcript
        ),
        "transcript_available": preprocess_result.transcript_available,
        "frame_evidence": preprocess_result.full_video_frame_evidence,
    }

  def _build_prompt_config(
      self,
      feature_configs: list[models.VideoFeature],
      config,
  ) -> models.PromptConfig:
    feature_questions = self._build_features_prompt(feature_configs, config)
    return models.PromptConfig(
        prompt=(
            "These are the questions that you have to answer for each feature:"
            f"\n{feature_questions}"
        ),
        system_instructions=_SYSTEM_INSTRUCTIONS,
    )

  def _build_features_prompt(
      self,
      feature_configs: list[models.VideoFeature],
      config,
  ) -> str:
    features_prompt = ""
    for feature in feature_configs:
      features_prompt += f"""
            Feature ID: {feature.id}
            Feature Name: {feature.name}
            Feature Category: {feature.category}
            Feature Sub Category: {feature.sub_category}
            Feature Video Segment: {feature.video_segment}
            Feature Evaluation Criteria: {feature.evaluation_criteria}
            Question: {feature.prompt_template}
            {self._augment_instructions(feature, config)} \n\n
        """

    video_metadata = f"""
            Brand Name: {config.brand_name}
            Brand Variations: {config.brand_variations}
            Branded Products: {config.branded_products}
            Branded Product Categories: {config.branded_products_categories}
        """
    return (
        features_prompt.replace("{brand_name}", config.brand_name)
        .replace("{brand_variations}", ", ".join(config.brand_variations))
        .replace("{branded_products}", ", ".join(config.branded_products))
        .replace(
            "{branded_products_categories}",
            ", ".join(config.branded_products_categories),
        )
        .replace(
            "{branded_call_to_actions_str}",
            ", ".join(config.branded_call_to_actions),
        )
        .replace("{metadata_summary}", video_metadata)
    )

  def _validate_feature_response(
      self,
      response_features,
      feature_configs: list[models.VideoFeature],
  ) -> list[dict]:
    """Require exactly one OpenAI result per requested feature ID."""
    if not isinstance(response_features, list):
      raise ValueError("OpenAI response features must be a list")

    expected_ids = [feature.id for feature in feature_configs]
    expected_id_set = set(expected_ids)
    features_by_id = {}
    duplicate_ids = []
    unexpected_ids = []

    for response_feature in response_features:
      if not isinstance(response_feature, dict):
        raise ValueError("OpenAI response feature items must be objects")
      feature_id = response_feature.get("id")
      if feature_id not in expected_id_set:
        unexpected_ids.append(feature_id)
        continue
      if feature_id in features_by_id:
        duplicate_ids.append(feature_id)
        continue
      features_by_id[feature_id] = response_feature

    missing_ids = [
        feature_id for feature_id in expected_ids if feature_id not in features_by_id
    ]
    if missing_ids or unexpected_ids or duplicate_ids:
      details = []
      if missing_ids:
        details.append(f"missing feature ids: {', '.join(missing_ids)}")
      if unexpected_ids:
        details.append(
            "unexpected feature ids: "
            + ", ".join(str(feature_id) for feature_id in unexpected_ids)
        )
      if duplicate_ids:
        details.append(
            "duplicate feature ids: "
            + ", ".join(str(feature_id) for feature_id in duplicate_ids)
        )
      raise ValueError(
          "OpenAI response did not match requested features; "
          + "; ".join(details)
      )

    return [features_by_id[feature_id] for feature_id in expected_ids]

  def _augment_instructions(
      self,
      feature: models.VideoFeature,
      config,
  ) -> str:
    call_to_actions = _CALL_TO_ACTIONS + list(config.branded_call_to_actions)
    return (
        "\n".join(feature.extra_instructions)
        .replace("{criteria}", feature.evaluation_criteria)
        .replace("{call_to_actions}", ", ".join(call_to_actions))
    )

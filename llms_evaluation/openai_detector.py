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

_SYSTEM_INSTRUCTIONS = """
You are an AI Video Analysis Engine. Your goal is to analyze video ad content
and answer questions about specific creative features based only on the visual
and auditory information provided. Do not infer facts from external knowledge.
For each feature, return the requested structured JSON fields, preserve the
exact feature id from the prompt, and cite concrete visual or transcript
evidence whenever possible.
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
    response = self.openai_service.evaluate_features(
        prompt_config=prompt_config,
        preprocess_result=preprocess_result,
        model_name=config.openai_model,
        schema=models.OPENAI_VIDEO_RESPONSE_SCHEMA,
        frame_paths=self._select_frames(preprocess_result, feature_configs),
    )
    return response.get("features", [])

  def _select_frames(
      self,
      preprocess_result: models.VideoPreprocessResult,
      feature_configs: list[models.VideoFeature],
  ) -> list[str]:
    if all(
        feature.video_segment == models.VideoSegment.FIRST_5_SECS_VIDEO
        for feature in feature_configs
    ):
      return preprocess_result.first_5_seconds_frames
    return preprocess_result.full_video_frames

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

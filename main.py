#!/usr/bin/env python3

###########################################################################
#
#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###########################################################################

"""Module to execute the ABCD Detector Assessment"""

import time
import traceback
import logging
import sys
import models
import utils
from configuration import Configuration


def get_video_sources(config: Configuration) -> list[models.VideoSource]:
  """Resolve configured video inputs without loading unused providers."""
  if config.llm_provider_type == models.LLMProviderType.OPENAI:
    from creative_providers.local_creative_provider import LocalCreativeProvider

    return LocalCreativeProvider().get_creative_sources(config)

  from creative_providers import creative_provider_registry

  creative_provider = creative_provider_registry.provider_factory.get_provider(
      config.creative_provider_type.value
  )
  return [
      models.VideoSource(
          original_uri=video_uri,
          local_path=video_uri,
          source_type=config.creative_provider_type.value,
      )
      for video_uri in creative_provider.get_creative_uris(config)
  ]


def _validate_legacy_video_uri(config: Configuration, video_uri: str) -> bool:
  """Validate provider-specific URI shape for legacy providers."""
  if (
      config.creative_provider_type == models.CreativeProviderType.GCS
      and "gs://" not in video_uri
  ):
    logging.error(
        "The creative provider GCS does not match with the video uri"
        f" {video_uri}. Stopping execution. Please check."
    )
    return False

  if (
      config.creative_provider_type == models.CreativeProviderType.YOUTUBE
      and "https://www.youtube.com" not in video_uri
  ):
    logging.error(
        "The creative provider YOUTUBE does not match with the video uri"
        f" {video_uri}. Stopping execution. Please check."
    )
    return False

  return True


def _build_openai_preprocessor(config: Configuration):
  """Build the OpenAI preprocessing pipeline."""
  from llms_evaluation import openai_api_service
  from llms_evaluation import openai_video_preprocessor

  openai_service = openai_api_service.OpenAIAPIService()
  return openai_video_preprocessor.VideoPreprocessor(
      cache_dir=config.cache_dir,
      max_frames=config.max_frames,
      frame_sample_rate=config.frame_sample_rate,
      refresh_cache=config.refresh_cache,
      openai_service=openai_service,
      transcription_model=config.openai_transcription_model,
  )


def _print_assessment_summary(
    video_assessment: models.VideoAssessment,
    long_form_abcd_evaluated_features: list[models.FeatureEvaluation],
    shorts_evaluated_features: list[models.FeatureEvaluation],
) -> None:
  """Print a compact assessment summary without loading Google helpers."""
  for label, evaluated_features in (
      ("Full ABCD", long_form_abcd_evaluated_features),
      ("Shorts", shorts_evaluated_features),
  ):
    if not evaluated_features:
      logging.info("There are not %s evaluated features results to display.", label)
      continue

    print(
        "\n"
        f"{label} assessment for {video_assessment.brand_name} "
        f"({video_assessment.video_uri})"
    )
    for feature_eval in evaluated_features:
      print(
          f"- {feature_eval.feature.id}: detected={feature_eval.detected}, "
          f"confidence={feature_eval.confidence_score}"
      )


def execute_abcd_assessment_for_videos(
    config: Configuration,
) -> list[models.VideoAssessment]:
  """Execute ABCD Assessment for all brand videos retrieved by the Creative Provider"""
  from evaluation_services import video_evaluation_service

  video_sources = get_video_sources(config)
  video_assessments = []
  is_openai = config.llm_provider_type == models.LLMProviderType.OPENAI
  openai_preprocessor = _build_openai_preprocessor(config) if is_openai else None

  for video_source in video_sources:
    video_uri = video_source.original_uri

    # Validate that creative provides match the video uris
    if not is_openai and not _validate_legacy_video_uri(config, video_uri):
      break

    print(f"\n\nProcessing ABCD Assessment for video {video_uri}... \n")

    preprocess_result = None
    if is_openai:
      preprocess_result = openai_preprocessor.preprocess(video_source)
    else:
      from annotations_evaluation import annotations_generation
      from helpers import generic_helpers

      # Generate video annotations for custom features. Annotations are supported only for GCS providers
      if (
          config.use_annotations
          and config.creative_provider_type == models.CreativeProviderType.GCS
      ):
        annotations_generation.generate_video_annotations(config, video_uri)

      # Full ABCD features require 1st_5_secs videos only for GCS providers
      if (
          config.run_long_form_abcd
          and config.creative_provider_type == models.CreativeProviderType.GCS
      ):
        generic_helpers.trim_video(config, video_uri)

    # Execute ABCD Assessment
    long_form_abcd_evaluated_features: models.FeatureEvaluation = []
    shorts_evaluated_features: models.FeatureEvaluation = []

    if config.run_long_form_abcd:
      long_form_abcd_evaluated_features = (
          video_evaluation_service.video_evaluation_service.evaluate_features(
              config=config,
              video_uri=video_uri,
              features_category=models.VideoFeatureCategory.LONG_FORM_ABCD,
              preprocess_result=preprocess_result,
          )
      )

    if config.run_shorts:
      shorts_evaluated_features = (
          video_evaluation_service.video_evaluation_service.evaluate_features(
              config=config,
              video_uri=video_uri,
              features_category=models.VideoFeatureCategory.SHORTS,
              preprocess_result=preprocess_result,
          )
      )

    video_assessment: models.VideoAssessment = models.VideoAssessment(
        brand_name=config.brand_name,
        video_uri=video_uri,
        long_form_abcd_evaluated_features=long_form_abcd_evaluated_features,
        shorts_evaluated_features=shorts_evaluated_features,
        config=config,
    )

    # Print assessments for Full ABCD and Shorts and store results
    if is_openai:
      _print_assessment_summary(
          video_assessment,
          long_form_abcd_evaluated_features,
          shorts_evaluated_features,
      )
    else:
      if len(long_form_abcd_evaluated_features) > 0:
        generic_helpers.print_abcd_assessment(
            video_assessment.brand_name,
            video_assessment.video_uri,
            long_form_abcd_evaluated_features,
        )
      else:
        logging.info(
            "There are not Full ABCD evaluated features results to display."
        )
      if len(shorts_evaluated_features) > 0:
        generic_helpers.print_abcd_assessment(
            video_assessment.brand_name,
            video_assessment.video_uri,
            shorts_evaluated_features,
        )
      else:
        logging.info(
            "There are not Shorts evaluated features results to display."
        )

    if config.bq_table_name and not is_openai:
      generic_helpers.store_in_bq(config, video_assessment)

    # Remove local version of video files
    if not is_openai:
      generic_helpers.remove_local_video_files()

    video_assessments.append(video_assessment)

  return video_assessments


def validate_config(config: Configuration) -> None:
  """Validate CLI configuration before execution."""
  if (
      config.llm_provider_type == models.LLMProviderType.OPENAI
      and config.extract_brand_metadata
      and _missing_brand_metadata(config)
  ):
    raise ValueError(
        "OpenAI metadata extraction is not implemented yet. Provide explicit "
        "brand and product metadata, or omit -extract_brand_metadata."
    )

  if utils.invalid_brand_metadata(config):
    raise ValueError(
        "The Extract Brand Metadata option is disabled and no brand details "
        "were defined. Please enable the option or define brand details."
    )


def _missing_brand_metadata(config: Configuration) -> bool:
  """Return whether required brand metadata is missing."""
  return (
      not config.brand_name
      or len(config.brand_variations) == 0
      or len(config.branded_products) == 0
      or len(config.branded_products_categories) == 0
  )


def main(arg_list: list[str] | None = None, raise_on_error: bool = False) -> None:
  """Main ABCD Assessment execution. See docstring and args.

  Args:
    arg_list: A list of command line arguments
    raise_on_error: Re-raise exceptions so command line execution exits nonzero.

  """

  try:
    args = utils.parse_args(arg_list)

    config = utils.build_abcd_params_config(args)
    validate_config(config)

    start_time = time.time()
    logging.info("Starting ABCD assessment... \n")

    if config.video_uris:
      video_assessments = execute_abcd_assessment_for_videos(config)
      if config.assessment_file and video_assessments:
        from helpers import generic_helpers

        generic_helpers.write_assessments_json(
            video_assessments,
            config.assessment_file,
        )
        generic_helpers.write_assessments_detected_csv(
            video_assessments,
            config.assessment_file,
        )
      logging.info("Finished ABCD assessment. \n")
    else:
      logging.info("There are no videos to process. \n")

    logging.info(
        "ABCD assessment took - %s mins. - \n", (time.time() - start_time) / 60
    )
  except Exception as ex:
    logging.error("ERROR: %s", ex)
    traceback.print_exc()
    if raise_on_error:
      raise


if __name__ == "__main__":
  try:
    main(raise_on_error=True)
  except Exception:
    sys.exit(1)

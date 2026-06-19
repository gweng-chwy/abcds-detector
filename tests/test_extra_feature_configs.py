"""Tests for extra imported-platform feature configs."""

import models
from features_repository import feature_configs_handler
from features_repository.long_form_abcd_features import (
    get_long_form_abcd_feature_configs,
)
from features_repository.shorts_features import get_shorts_feature_configs


def _feature_by_id(features, feature_id):
  matches = [feature for feature in features if feature.id == feature_id]
  assert len(matches) == 1
  return matches[0]


def test_long_form_extra_feature_configs():
  features = get_long_form_abcd_feature_configs()

  assert len(features) == 24
  feature = _feature_by_id(
      features, "e_long_form_max_10_words_per_frame"
  )
  assert feature.name == "Max 10 Words Per Frame"
  assert feature.category == models.VideoFeatureCategory.LONG_FORM_ABCD
  assert feature.sub_category == models.VideoFeatureSubCategory.NONE
  assert feature.video_segment == models.VideoSegment.FULL_VIDEO
  assert feature.evaluation_method == models.EvaluationMethod.LLMS
  assert feature.evaluation_function == ""
  assert feature.include_in_evaluation is True
  assert feature.group_by == models.VideoSegment.FULL_VIDEO


def test_shorts_extra_feature_configs():
  features = get_shorts_feature_configs()

  assert len(features) == 22

  max_words = _feature_by_id(features, "e_shorts_max_10_words_per_frame")
  assert max_words.name == "Max 10 Words Per Frame"
  assert max_words.category == models.VideoFeatureCategory.SHORTS
  assert max_words.sub_category == models.VideoFeatureSubCategory.NONE
  assert max_words.video_segment == models.VideoSegment.FULL_VIDEO
  assert max_words.evaluation_method == models.EvaluationMethod.LLMS
  assert max_words.evaluation_function == ""
  assert max_words.include_in_evaluation is True
  assert max_words.group_by == models.VideoSegment.FULL_VIDEO

  brand_mention = _feature_by_id(
      features, "e_shorts_brand_mention_speech_1st_5_secs"
  )
  assert brand_mention.name == "Brand Mention (Speech) (First 5 seconds)"
  assert brand_mention.category == models.VideoFeatureCategory.SHORTS
  assert brand_mention.sub_category == models.VideoFeatureSubCategory.BRAND
  assert brand_mention.video_segment == models.VideoSegment.FIRST_5_SECS_VIDEO
  assert brand_mention.evaluation_method == models.EvaluationMethod.LLMS
  assert brand_mention.evaluation_function == ""
  assert brand_mention.include_in_evaluation is True
  assert brand_mention.group_by == models.VideoSegment.FIRST_5_SECS_VIDEO


def test_shorts_extra_first_5_feature_groups_with_first_5_segment():
  groups = (
      feature_configs_handler.features_configs_handler
      .get_features_by_category_by_group_config(
          models.VideoFeatureCategory.SHORTS
      )
  )

  first_5_feature_ids = {
      feature.id
      for feature in groups[models.VideoSegment.FIRST_5_SECS_VIDEO.value]
  }

  assert "e_shorts_brand_mention_speech_1st_5_secs" in first_5_feature_ids

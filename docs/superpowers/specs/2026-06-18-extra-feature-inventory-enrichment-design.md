# Extra Feature Inventory Enrichment Design

## Context

The current feature inventory already covers most imported platform attributes
exactly or partially. This change only adds attributes that are totally missing
from the current inventory. Existing partial overlaps remain unchanged.

## Scope

Add three LLM-only `VideoFeature` definitions:

| Video type | Feature id | Feature name | Sub-category | Segment |
|---|---|---|---|---|
| Long form | `e_long_form_max_10_words_per_frame` | Max 10 Words Per Frame | `NONE` | `FULL_VIDEO` |
| Shorts | `e_shorts_max_10_words_per_frame` | Max 10 Words Per Frame | `NONE` | `FULL_VIDEO` |
| Shorts | `e_shorts_brand_mention_speech_1st_5_secs` | Brand Mention (Speech) (First 5 seconds) | `BRAND` | `FIRST_5_SECS_VIDEO` |

## Non-Goals

- Do not modify existing `a_`, `b_`, `c_`, or `d_` feature semantics.
- Do not add partial-overlap attributes such as generic audio, CTA, brand
  throughout, human presence, or scene/motion checks.
- Do not change the evaluation pipeline unless tests show config-only addition
  is insufficient.

## Implementation

Add feature configs directly to:

- `features_repository/long_form_abcd_features.py`
- `features_repository/shorts_features.py`

Use existing config patterns:

- `evaluation_method=EvaluationMethod.LLMS`
- `evaluation_function=""`
- `include_in_evaluation=True`
- `group_by` matching `video_segment`

Update `docs/feature_inventory/` after code so the abstract summary and full
attribute tables include the new `e_` features.

## Testing

Add or update focused tests to verify:

- the three new feature IDs are present in the expected category;
- long form count increases from 23 to 24;
- Shorts count increases from 20 to 22;
- first-5 Shorts brand mention groups under `FIRST_5_SECS_VIDEO`.

Run the narrow relevant tests, then run broader tests if the change touches
shared handler behavior.

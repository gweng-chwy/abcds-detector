# Extra Feature Inventory Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add only the totally missing imported platform attributes as `e_` feature configs, then refresh feature inventory docs.

**Architecture:** This is a config-only feature expansion. Existing feature config handlers group and route `VideoFeature` entries automatically, so no pipeline code should change unless tests prove otherwise.

**Tech Stack:** Python 3.11, repository `VideoFeature` dataclass, pytest, Markdown docs.

---

## File Structure

- Modify `features_repository/long_form_abcd_features.py`: add one long-form `e_` feature under `VideoFeatureSubCategory.NONE`.
- Modify `features_repository/shorts_features.py`: add one Shorts `e_` brand feature under `BRAND` and one Shorts `e_` text-density feature under `NONE`.
- Create `tests/test_extra_feature_configs.py`: assert exact IDs, counts, categories, grouping, and evaluation config.
- Modify `docs/feature_inventory/README.md`: refreshed abstract summary counts and feature names.
- Modify `docs/feature_inventory/long_form_abcd_features.md`: refreshed full attr table for long form.
- Modify `docs/feature_inventory/shorts_features.md`: refreshed full attr table for Shorts.

## Task 1: Add Failing Config Tests

**Files:**
- Create: `tests/test_extra_feature_configs.py`

- [ ] **Step 1: Write the failing test file**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extra_feature_configs.py -q`

Expected: FAIL. `len(features)` assertions see current counts `23` and `20`, and new `e_` IDs are absent.

- [ ] **Step 3: Commit failing test if repository policy requires test-first commits**

Skip commit if keeping red/green in one local change. If committing red tests is acceptable:

```bash
git add tests/test_extra_feature_configs.py
git commit -m "test: cover extra feature configs"
```

## Task 2: Add Feature Configs

**Files:**
- Modify: `features_repository/long_form_abcd_features.py`
- Modify: `features_repository/shorts_features.py`
- Test: `tests/test_extra_feature_configs.py`

- [ ] **Step 1: Add long-form feature config**

Insert before the closing `]` in `get_long_form_abcd_feature_configs()`:

```python
      VideoFeature(
          id="e_long_form_max_10_words_per_frame",
          name="Max 10 Words Per Frame",
          category=VideoFeatureCategory.LONG_FORM_ABCD,
          sub_category=VideoFeatureSubCategory.NONE,
          video_segment=VideoSegment.FULL_VIDEO,
          evaluation_criteria="""
                Every frame with visible on-screen text contains no more than 10 words.
                Frames without visible text pass this criterion.
            """,
          prompt_template="""
                Does every frame with visible on-screen text contain 10 words or fewer?
            """,
          extra_instructions=[
              "Consider the following criteria for your answer: {criteria}",
              (
                  "Look through each frame in the video carefully and count"
                  " visible on-screen words per frame."
              ),
              (
                  "Treat readable brand names, subtitles, captions, supers,"
                  " disclaimers, and product text as visible words."
              ),
              (
                  "Return False if any single frame contains more than 10"
                  " visible words."
              ),
              (
                  "Provide the timestamp or frame evidence for the highest"
                  " word count observed."
              ),
          ],
          evaluation_method=EvaluationMethod.LLMS,
          evaluation_function="",
          include_in_evaluation=True,
          group_by=VideoSegment.FULL_VIDEO,
      ),
```

- [ ] **Step 2: Add Shorts brand first-5 feature config**

Insert in `get_shorts_feature_configs()` after `shorts_product_extreme_closeup` and before the `CONNECT` group starts:

```python
      VideoFeature(
          id="e_shorts_brand_mention_speech_1st_5_secs",
          name="Brand Mention (Speech) (First 5 seconds)",
          category=VideoFeatureCategory.SHORTS,
          sub_category=VideoFeatureSubCategory.BRAND,
          video_segment=VideoSegment.FIRST_5_SECS_VIDEO,
          evaluation_criteria="""
            The brand name is heard in the audio or speech in the first 5 seconds of the video.
            """,
          prompt_template="""
            Does the speech mention the brand {brand_name} in the first 5 seconds of the video?
            """,
          extra_instructions=[
              "Consider the following criteria for your answer: {criteria}",
              (
                  "Only evaluate audio or speech from the first 5 seconds of"
                  " the video."
              ),
              (
                  "Count clear mentions of the brand name or any configured"
                  " brand variation."
              ),
              (
                  "Provide the exact timestamp when the brand {brand_name} is"
                  " heard in speech."
              ),
              (
                  "Return True if and only if the brand {brand_name} or a"
                  " configured brand variation is heard in the first 5 seconds."
              ),
          ],
          evaluation_method=EvaluationMethod.LLMS,
          evaluation_function="",
          include_in_evaluation=True,
          group_by=VideoSegment.FIRST_5_SECS_VIDEO,
      ),
```

- [ ] **Step 3: Add Shorts max-words feature config**

Insert near the Shorts `NONE` features, before `shorts_production_style_index`:

```python
      VideoFeature(
          id="e_shorts_max_10_words_per_frame",
          name="Max 10 Words Per Frame",
          category=VideoFeatureCategory.SHORTS,
          sub_category=VideoFeatureSubCategory.NONE,
          video_segment=VideoSegment.FULL_VIDEO,
          evaluation_criteria="""
            Every frame with visible on-screen text contains no more than 10 words.
            Frames without visible text pass this criterion.
            """,
          prompt_template="""
            Does every frame with visible on-screen text contain 10 words or fewer?
            """,
          extra_instructions=[
              "Consider the following criteria for your answer: {criteria}",
              (
                  "Look through each frame in the video carefully and count"
                  " visible on-screen words per frame."
              ),
              (
                  "Treat readable brand names, subtitles, captions, supers,"
                  " stickers, disclaimers, and product text as visible words."
              ),
              (
                  "Return False if any single frame contains more than 10"
                  " visible words."
              ),
              (
                  "Provide the timestamp or frame evidence for the highest"
                  " word count observed."
              ),
          ],
          evaluation_method=EvaluationMethod.LLMS,
          evaluation_function="",
          include_in_evaluation=True,
          group_by=VideoSegment.FULL_VIDEO,
      ),
```

- [ ] **Step 4: Run focused tests to verify green**

Run: `pytest tests/test_extra_feature_configs.py -q`

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit implementation if committing per task**

```bash
git add features_repository/long_form_abcd_features.py features_repository/shorts_features.py tests/test_extra_feature_configs.py
git commit -m "feat: add extra feature configs"
```

## Task 3: Refresh Feature Inventory Docs

**Files:**
- Modify: `docs/feature_inventory/README.md`
- Modify: `docs/feature_inventory/long_form_abcd_features.md`
- Modify: `docs/feature_inventory/shorts_features.md`

- [ ] **Step 1: Regenerate feature inventory docs from feature configs**

Run:

```bash
python3 - <<'PY'
from collections import OrderedDict
from html import escape
from inspect import cleandoc
from pathlib import Path

from features_repository.long_form_abcd_features import get_long_form_abcd_feature_configs
from features_repository.shorts_features import get_shorts_feature_configs

OUT = Path("docs/feature_inventory")
OUT.mkdir(parents=True, exist_ok=True)

ABCD_ORDER = ["ATTRACT", "BRAND", "CONNECT", "DIRECT", "NONE"]
LABELS = {
    "ATTRACT": "Attract",
    "BRAND": "Brand",
    "CONNECT": "Connect",
    "DIRECT": "Direct",
    "NONE": "Other / None",
}


def md_escape(value: str) -> str:
  return value.replace("\\", "\\\\").replace("|", "\\|")


def inline_code(value: str) -> str:
  return f"`{md_escape(value)}`"


def enum_value(value):
  return getattr(value, "value", str(value))


def clean_block(text):
  if text is None:
    return ""
  return cleandoc(text)


def html_table_text(text):
  text = escape(clean_block(text), quote=False).replace("|", "&#124;")
  return text.replace("\n", "<br>")


def cell_text(value):
  if value is None:
    return ""
  text = html_table_text(value)
  if not text:
    return ""
  return f"<pre><code>{text}</code></pre>"


def list_cell(items):
  if not items:
    return ""
  rendered = []
  for idx, item in enumerate(items, 1):
    rendered.append(f"{idx}. {html_table_text(str(item))}")
  return "<br><br>".join(rendered)


def group_features(features):
  grouped = OrderedDict((key, []) for key in ABCD_ORDER)
  for feature in features:
    grouped.setdefault(feature.sub_category.value, []).append(feature)
  return OrderedDict((key, value) for key, value in grouped.items() if value)


def summary_table(features):
  lines = ["| ABCD | Count | Features |", "|---|---:|---|"]
  for key, group in group_features(features).items():
    names = "; ".join(md_escape(feature.name) for feature in group)
    lines.append(f"| {LABELS.get(key, key.title())} | {len(group)} | {names} |")
  return "\n".join(lines)


def feature_attributes(feature):
  return [
      ("id", inline_code(feature.id)),
      ("name", md_escape(feature.name)),
      ("category", inline_code(feature.category.value)),
      ("sub_category", inline_code(feature.sub_category.value)),
      ("video_segment", inline_code(feature.video_segment.value)),
      ("evaluation_criteria", cell_text(feature.evaluation_criteria)),
      ("prompt_template", cell_text(feature.prompt_template)),
      ("extra_instructions", list_cell(feature.extra_instructions)),
      ("evaluation_method", inline_code(feature.evaluation_method.value)),
      ("evaluation_function", inline_code(feature.evaluation_function or "")),
      ("include_in_evaluation", inline_code(str(feature.include_in_evaluation))),
      ("group_by", inline_code(enum_value(feature.group_by))),
  ]


def detail_doc(title, source_path, features):
  lines = [
      f"# {title}",
      "",
      f"Source: `{source_path}`.",
      "",
      "Each feature below includes every `VideoFeature` config attribute currently defined in code.",
      "",
  ]
  for key, group in group_features(features).items():
    lines.extend([f"## {LABELS.get(key, key.title())}", ""])
    for feature in group:
      lines.extend([
          f"### `{feature.id}` - {feature.name}",
          "",
          "| Attribute | Value |",
          "|---|---|",
      ])
      for attr, value in feature_attributes(feature):
        lines.append(f"| `{attr}` | {value} |")
      lines.append("")
  return "\n".join(lines).rstrip() + "\n"


def readme(long_features, shorts_features):
  return f"""# Feature Inventory

Current feature inventory for ABCDs Detector, grouped by the ABCD paradigm when a feature has an ABCD sub-category.

Sources:

- `features_repository/long_form_abcd_features.py`
- `features_repository/shorts_features.py`

Detailed definitions:

- [Long form full definitions](long_form_abcd_features.md)
- [Shorts full definitions](shorts_features.md)

## Long Form ABCD Summary

{summary_table(long_features)}

## Shorts Summary

{summary_table(shorts_features)}
"""


long_features = get_long_form_abcd_feature_configs()
shorts_features = get_shorts_feature_configs()

(OUT / "README.md").write_text(readme(long_features, shorts_features), encoding="utf-8")
(OUT / "long_form_abcd_features.md").write_text(
    detail_doc("Long Form ABCD Feature Definitions", "features_repository/long_form_abcd_features.py", long_features),
    encoding="utf-8",
)
(OUT / "shorts_features.md").write_text(
    detail_doc("Shorts Feature Definitions", "features_repository/shorts_features.py", shorts_features),
    encoding="utf-8",
)
PY
```

Expected output: no terminal output. Files are rewritten. The abstract summary includes long form `Other / None | 1 | Max 10 Words Per Frame`, Shorts `Brand | 3 | Product Close-Up; Product Extreme Close-Up; Brand Mention (Speech) (First 5 seconds)`, and Shorts `Other / None | 9 | ... Max 10 Words Per Frame ...`.

- [ ] **Step 2: Verify docs contain new IDs**

Run:

```bash
rg -n "e_long_form_max_10_words_per_frame|e_shorts_brand_mention_speech_1st_5_secs|e_shorts_max_10_words_per_frame" docs/feature_inventory
```

Expected: all three IDs appear in full definition docs.

- [ ] **Step 3: Verify docs and source counts agree**

Run:

```bash
python3 -c 'from pathlib import Path
from features_repository.long_form_abcd_features import get_long_form_abcd_feature_configs
from features_repository.shorts_features import get_shorts_feature_configs
checks = [(Path("docs/feature_inventory/long_form_abcd_features.md"), len(get_long_form_abcd_feature_configs())), (Path("docs/feature_inventory/shorts_features.md"), len(get_shorts_feature_configs()))]
for path, expected_features in checks:
  text = path.read_text()
  feature_sections = text.count("### `")
  if feature_sections != expected_features:
    raise SystemExit(f"{path}: expected {expected_features} sections, found {feature_sections}")
  print(f"{path}: {feature_sections} feature sections")
print("feature inventory counts match source configs")'
```

Expected: long form reports `24`, Shorts reports `22`, then prints `feature inventory counts match source configs`.

- [ ] **Step 4: Commit docs if committing per task**

```bash
git add docs/feature_inventory/README.md docs/feature_inventory/long_form_abcd_features.md docs/feature_inventory/shorts_features.md
git commit -m "docs: refresh feature inventory"
```

## Task 4: Final Verification

**Files:**
- Read: `features_repository/long_form_abcd_features.py`
- Read: `features_repository/shorts_features.py`
- Read: `tests/test_extra_feature_configs.py`
- Read: `docs/feature_inventory/README.md`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_extra_feature_configs.py -q`

Expected: `3 passed`.

- [ ] **Step 2: Run OpenAI routing tests touching grouping**

Run: `pytest tests/test_openai_pipeline_routing.py tests/test_openai_detector.py -q`

Expected: all tests pass.

- [ ] **Step 3: Run feature docs count verification**

Run the docs/source count command from Task 3 Step 3.

Expected: long form reports `24`, Shorts reports `22`, then prints `feature inventory counts match source configs`.

- [ ] **Step 4: Review git diff**

Run: `git diff -- features_repository/long_form_abcd_features.py features_repository/shorts_features.py tests/test_extra_feature_configs.py docs/feature_inventory`

Expected: only the three new `e_` configs, focused tests, and regenerated feature inventory docs changed.

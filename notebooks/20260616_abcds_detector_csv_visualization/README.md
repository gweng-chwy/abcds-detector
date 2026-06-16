# ABCDs Detector CSV Visualization

This experiment follows the notebooks convention: one timestamped folder per small, focused experiment.

## Inputs

- Source CSV copy: `data/sample_openai_results_check.csv`
- Source JSON copy: `data/sample_openai_results_check.json`

The JSON copy is used only to split feature IDs into Long-form ABCD and Shorts accurately.

## Tables

- `data/long_form_detected_features.csv`: one row per video, Long-form ABCD feature columns only.
- `data/shorts_detected_features.csv`: one row per video, Shorts feature columns only.
- `data/long_form_detected_features_long.csv`: normalized Long-form ABCD feature table.
- `data/shorts_detected_features_long.csv`: normalized Shorts feature table.

## Figures

- `figures/detected_summary_by_category.svg`: detected true/false summary split by category.
- `figures/long_form_detected_matrix.svg`: Long-form ABCD feature-by-video matrix.
- `figures/shorts_detected_matrix.svg`: Shorts feature-by-video matrix.

## Current Snapshot

- Video rows: 1
- Long-form ABCD features: 23
- Shorts features: 20
- Long-form detected true: 8
- Long-form detected false: 15
- Shorts detected true: 8
- Shorts detected false: 12

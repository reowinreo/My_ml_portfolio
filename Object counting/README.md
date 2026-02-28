# Object Counting

This folder includes helper modules for pseudo-label creation and counting-oriented backbone components.

## Scripts

- `creating_pseudo_dataset.py`
  - Collects image paths, supports split-aware filtering, and interactive exemplar selection.
  - Matches exemplar templates to generate candidate prompts.

- `creating_pseudo_model_gd.py`
  - Wraps local GroundingDINO inference to produce pseudo detection boxes.

- `creating_pseudo_model_sam3.py`
  - Wraps SAM3-related model initialization/inference utilities.

- `training_backbone_ALTGVT_models.py`
  - Defines model structures for ALTGVT-based ranking/counting outputs.

- `training_backbone_ALTGVT_datasets.py`
  - Dataset utilities for ALTGVT training pipeline.

- `training_backbone_ALTGVT_losses.py`
  - Loss-related utilities for ALTGVT training objectives.

## Usage intent

These scripts are experimental components. They are designed as reusable building blocks in manual research workflows rather than an end-to-end packaged application.

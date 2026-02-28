# Knowledge Distillation

This folder stores teacher-student distillation experiments for scene classification.

## Scripts

- `train_resnet152_teacher.py`
  - Trains a ResNet152 teacher model.
  - Saves best and final checkpoints for downstream distillation.

- `distill_kd_ce_uncertainty_weighted_resnet50.py`
  - Distills a ResNet50 student with CE + KD and uncertainty-based weighting.

- `distill_dkd_uncertainty_weighted_resnet50.py`
  - Distills using DKD (TCKD/NCKD) with learnable uncertainty weighting.

- `distill_feature_kd_ce_uncertainty_weighted.py`
  - Combines feature-level alignment with KD/CE style objectives.

- `distill_feature_logit_mse_ce_uncertainty_weighted.py`
  - Uses feature/logit matching (including MSE-style terms) plus supervised objectives.

- `distill_pca_projected_feature_two_stage_resnet50.py`
  - Two-stage distillation pipeline with PCA-projected feature matching.

## Typical pipeline

1. Train or load a high-capacity teacher checkpoint.
2. Build student model and distillation loss.
3. Train student with teacher guidance.
4. Evaluate and compare student accuracy vs. teacher baseline.

## Inputs/outputs

- Typical dataset path: `dataset_raw`.
- Typical split file: `saved_models/split_indices.npz`.
- Typical checkpoint output path: `saved_models/`.

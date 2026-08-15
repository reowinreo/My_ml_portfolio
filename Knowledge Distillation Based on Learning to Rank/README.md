# Knowledge Distillation Based on Learning to Rank

This folder studies knowledge distillation losses and loss-weight hyperparameters, with a focus on aerial/remote-sensing scene classification and ranking/counting tasks.

## Teacher Training

- `teacher_training/train_resnet152_teacher.py`
  - Trains a ResNet152 teacher model on NWPU-RESISC45.
  - Saves best and final checkpoints for downstream distillation.

## DKD Distillation

- `dkd_distillation/distill_dkd_uncertainty_weighted_resnet50.py`
  - Decoupled Knowledge Distillation (DKD) with learnable uncertainty weighting between TCKD and NCKD.

- `dkd_distillation/distill_dkd_manual_weighted_resnet50.py`
  - DKD with manually fixed weights (`alpha=4` for TCKD, `beta=1` for NCKD).
  - Used to study the effect of loss-component weighting.

## Logit Distillation

- `logit_distillation/distill_kd_ce_uncertainty_weighted_resnet50.py`
  - Standard KD (KL on softened logits) combined with hard-label CE and uncertainty-based weighting.

- `logit_distillation/distill_kd_weighted_loss.py`
  - Explores a weighted formulation of the standard KD loss.

- `logit_distillation/distill_kd_weighted_with_hard_label.py`
  - KD + hard-label CE with a tunable `lambda_ce` weight.

- `logit_distillation/distill_logit_mse_resnet50.py`
  - Replaces KL with MSE between student and teacher logits, combined with CE via uncertainty weighting.

- `logit_distillation/distill_logit_cosine_similarity_resnet50.py`
  - Uses cosine-similarity loss on temperature-scaled and normalized logits.

## Feature Distillation

- `feature_distillation/distill_feature_kd_ce_uncertainty_weighted.py`
  - Aligns intermediate features while preserving KD/CE objectives.

- `feature_distillation/distill_feature_logit_mse_ce_uncertainty_weighted.py`
  - Combines feature-level MSE, logit-level MSE, and supervised CE with uncertainty weighting.

- `feature_distillation/distill_feature_two_stage_resnet50.py`
  - Two-stage feature distillation: Stage 1 aligns the backbone; Stage 2 fine-tunes the classifier.

- `feature_distillation/distill_feature_two_stage_resnet50_stage2.py`
  - Standalone Stage 2 script that loads a Stage 1 checkpoint and trains only the classification head.

- `feature_distillation/distill_pca_projected_feature_two_stage_resnet50.py`
  - Two-stage distillation with PCA-projected feature matching.

## Learning-to-Rank Weighting

- `learning_to_rank_weighting/train_rank_model.py`
  - RankNet-based training of a counting/ranking model with global and local pairwise losses.

- `learning_to_rank_weighting/dual_branch_density_rank.py`
  - Demonstrates uncertainty-based dynamic weighting between a ranking loss and a density-map regression loss.

## Notes

- All student experiments use ResNet50 unless otherwise noted; the teacher is ResNet152.
- Most scripts expect `dataset_raw/` and `saved_models/` folders to be present locally.
- The scripts are written in PyTorch and use English comments for readability.

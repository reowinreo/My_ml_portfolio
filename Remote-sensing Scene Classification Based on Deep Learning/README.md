# Remote-sensing Scene Classification Based on Deep Learning

This folder contains a collection of deep-learning approaches for aerial/remote-sensing scene classification, covering four research directions.

## 1. Feature Extraction + SVM

Use classical CNNs as fixed feature extractors and train an SVM classifier on top.

- `feature_extraction_svm/googlenet_feature_svm.py`
  - Loads a pretrained GoogLeNet, removes the final fc layer, and extracts penultimate-layer features.
  - L2-normalizes features and trains a Linear SVM with grid search over `C`.

- `feature_extraction_svm/resnet50_feature_pca_svm.py`
  - Uses a pretrained ResNet50 as a feature extractor.
  - Applies standard scaling and PCA (retaining 95% variance).
  - Trains an RBF-SVM for 45-class scene classification.

## 2. End-to-End Fine-Tuning

Fine-tune popular backbones end-to-end on the NWPU-RESISC45 dataset.

- `end_to_end_finetuning/train_googlenet_finetune.py`
  - Fine-tunes GoogLeNet with a two-group learning-rate schedule.

- `end_to_end_finetuning/train_vgg16_finetune.py`
  - Fine-tunes VGG16 with cross-entropy loss.

- `end_to_end_finetuning/train_vgg16_focal_loss.py`
  - Same VGG16 fine-tuning pipeline but uses Focal Loss (`gamma=2`) to handle class imbalance.

- `end_to_end_finetuning/train_swin_t_finetune.py`
  - Fine-tunes Swin-T (Swin Transformer Tiny).

- `end_to_end_finetuning/train_vit_b16_finetune.py`
  - Fine-tunes ViT-B/16 with head replacement.

## 3. Metric Learning

Learn an embedding space with Siamese/Triplet networks and evaluate with an SVM.

- `metric_learning/siamese_contrastive_bce_embedding.py`
  - Siamese CNN trained with contrastive loss and BCE.
  - Exports embeddings and evaluates with SVM.

- `metric_learning/triplet_metric_learning_embedding.py`
  - CNN embedding network trained with Triplet Loss.
  - Builds triplets from labeled samples and uses SVM for final classification.

## 4. R4C / Learning-to-Rank Multi-Label Classification

A ranking-based perspective for multi-label aerial-image classification and counting.

- `r4c_learning_to_rank/train_rank_model.py`
  - Trains `RankCountingNet` with RankNet-style pairwise ranking loss.
  - Produces global and local ranking scores from a ResNet50 backbone.

- `r4c_learning_to_rank/eval_rank_model.py`
  - Evaluates the ranking model using Spearman correlation and pairwise ranking accuracy.

- `r4c_learning_to_rank/dual_branch_density_rank.py`
  - Jointly trains a ranking branch and a density-map branch with uncertainty-based dynamic weighting.

- `r4c_learning_to_rank/dual_branch_density_rank_with_regression.py`
  - Evaluation script that calibrates `rank_global` to object counts via least-squares regression and fuses ranking and density predictions.

- `r4c_learning_to_rank/creating_pseudo_dataset.py`, `creating_pseudo_model_gd.py`, `creating_pseudo_model_sam3.py`
  - Helpers for generating pseudo labels with GroundingDINO and SAM3.

- `r4c_learning_to_rank/altgvt_components/`
  - Model, dataset, and loss utilities for the ALTGVT-based ranking/counting backbone.

## Notes

- Most scripts assume a local `dataset_raw` folder containing NWPU-RESISC45 or RSOC data.
- Pretrained weights are expected under `pretrained_models/` or `saved_models/`.
- All scripts are written in PyTorch and use English comments for clarity.

# ML Portfolio

This repository is a personal portfolio of machine-learning experiments I completed during my undergraduate study.
It is **not** intended to be a production-ready or deployable software project.

## What this repository demonstrates

- **Remote-sensing Scene Classification Based on Deep Learning**
  - Feature extraction with classical CNNs followed by SVM classification.
  - End-to-end fine-tuning of GoogLeNet, VGG16, Swin-T, and ViT-B/16.
  - Metric learning for aerial-image classification (Siamese / Triplet embeddings).
  - Learning-to-rank multi-label classification and counting (R4C-style).

- **Knowledge Distillation Based on Learning to Rank**
  - Teacher-student distillation for scene classification (ResNet152 → ResNet50).
  - Multiple distillation losses: DKD, logit MSE, cosine similarity, feature distillation, PCA-projected features.
  - Studies on loss-weight hyperparameters: manual weights, uncertainty-based dynamic weighting, and hard-label participation.
  - Learning-to-rank weighting for multi-task ranking-density training.

- **DAVE-based Conversion**
  - Scripts for converting pseudo labels with a DAVE-style inference flow.

## Repository structure

- `Remote-sensing Scene Classification Based on Deep Learning/` — scene classification experiments.
- `Knowledge Distillation Based on Learning to Rank/` — knowledge distillation and ranking/counting experiments.
- `DAVE/` — conversion script for pseudo labels using a DAVE workflow.

Each subfolder contains its own `README.md` with script-level explanations.

## Notes

- Many scripts assume local datasets, checkpoints, or helper modules are already available.
- Paths and hyperparameters are intentionally explicit in scripts because this repo is used as a study record.
- Results may vary by environment, random seed, and dataset split.

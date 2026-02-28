# ML Portfolio

This repository is a personal portfolio of machine learning experiments I completed during my undergraduate study.
It is **not** intended to be a production-ready or deployable software project.

## What this repository demonstrates

- **Scene Classification**: fine-tuning classic and transformer backbones on remote-sensing scene datasets.
- **Knowledge Distillation**: teacher-student pipelines (e.g., ResNet152 -> ResNet50) with multiple KD variants.
- **Metric Learning**: Siamese/Triplet embedding learning followed by SVM evaluation.
- **Object Counting / Pseudo Labels**: pseudo-label generation and counting-oriented pipelines with custom utilities.
- **DAVE-based Conversion**: scripts for converting pseudo labels with DAVE-style inference flows.

## Repository structure

- `Scene classification/` — scene classification training scripts (GoogLeNet, VGG16, ViT-B/16, Swin-T).
- `Knowledge distillation/` — teacher training and several distillation strategies.
- `Metric learning/` — embedding-learning pipelines and downstream SVM experiments.
- `Object counting/` — pseudo dataset/model helpers and counting backbone components.
- `DAVE/` — conversion script for pseudo labels using a DAVE workflow.

Each subfolder contains its own `README.md` with script-level explanations.

## Notes

- Many scripts assume local datasets, checkpoints, or helper modules are already available.
- Paths and hyperparameters are intentionally explicit in scripts because this repo is used as a study record.
- Results may vary by environment, random seed, and dataset split.

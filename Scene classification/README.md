# Scene Classification

This folder contains fine-tuning scripts for remote-sensing scene classification experiments.

## Scripts

- `train_googlenet_finetune.py`
  - Fine-tunes GoogLeNet for 45-class scene recognition.
  - Uses train/validation splitting, standard augmentation, and checkpoint saving.

- `train_vgg16_finetune.py`
  - Fine-tunes VGG16 with similar training/evaluation flow.
  - Useful as a CNN baseline for comparison.

- `train_vit_b16_finetune.py`
  - Fine-tunes ViT-B/16.
  - Includes logic for loading local/online pretrained weights and replacing the classification head.

- `train_swin_t_finetune.py`
  - Fine-tunes Swin-T (Tiny).
  - Includes model-head replacement and two-group learning rate setup.

## Common workflow

1. Build train/validation datasets from local image folders.
2. Load a backbone and adapt its classifier head to the target class count.
3. Train with cross-entropy and a learning-rate scheduler.
4. Save best/final weights and evaluate on validation data.

## Inputs/outputs

- Typical input dataset path: `dataset_raw`.
- Typical output directory: `saved_models/`.

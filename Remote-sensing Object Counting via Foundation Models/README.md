# Remote-sensing Object Counting via Foundation Models

This folder contains research code for a few-shot remote-sensing object-counting pipeline that combines foundation models (Grounding DINO and SAM 3) with learning-to-rank ideas.

## What the code demonstrates

- **Pseudo-label generation** with vision-language foundation models
  - Use Grounding DINO to produce class-aware bounding-box proposals.
  - Refine proposals with SAM 3 to obtain masks and object counts.
  - Generate 4×4 grid pseudo-labels, local ranking labels, and point-grid prompts for small objects.

- **Counting-network training**
  - Train ranking-based counting networks on pseudo labels or real labels.
  - Backbones: ALT-GVT, CBAM-enhanced CNN, and VGG16.
  - Heads: global/local ranking heads, regression heads, and dual-branch ranking+density heads.
  - Losses: ranking loss (RankNet-style), density-map MSE, MAE/RMSE regression, and dynamic multi-task weighting.

- **Inference and evaluation**
  - Run SAM 3 zero-shot counting on new images.
  - Evaluate ranking models with Spearman correlation and pairwise ranking accuracy.
  - Visualize pseudo labels, Grad-CAM attention, and detection results.

- **Model components**
  - `alt_gvt.py`: ALT-GVT-Small backbone and regression head.
  - `cbam.py` / `cbam_safe.py`: Convolutional Block Attention Module implementations.

## Folder layout

```text
Remote-sensing Object Counting via Foundation Models/
├── pseudo_label_generation/   # Build pseudo labels from Grounding DINO + SAM 3
├── counting_network_training/ # Train ranking / regression / density counting models
├── inference_evaluation/      # Test-time inference, checks, and visualizations
└── model_components/          # ALT-GVT and CBAM building blocks
```

## Notes

- The scripts are research prototypes. Paths, API tokens, and dataset roots are hard-coded as constants near the top of each file.
- Expected datasets include RSOC (Building / Ship), VisDrone, DOTA, and similar remote-sensing counting benchmarks.
- Pretrained weights such as `alt_gvt_small.pth` and SAM 3 checkpoints must be downloaded separately.
- Many scripts use `num_workers=0` for compatibility with Windows multiprocessing.

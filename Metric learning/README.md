# Metric Learning

This folder contains embedding-learning experiments that use Siamese/Triplet training and SVM evaluation.

## Scripts

- `triplet_metric_learning_embedding.py`
  - Builds triplets from labeled samples.
  - Trains a CNN embedding network with Triplet Loss.
  - Extracts embeddings and evaluates with an SVM classifier.

- `siamese_contrastive_bce_embedding.py`
  - Builds positive/negative pairs for a Siamese network.
  - Trains with contrastive loss and BCE.
  - Uses learned embeddings for SVM-based classification.

## Workflow summary

1. Load feature/label CSV files.
2. Construct pair/triplet datasets.
3. Train an embedding model.
4. Export embeddings and evaluate with SVM.

## Inputs/outputs

- Expected inputs include CSV files such as `X_train_sat6.csv`, `y_train_sat6.csv`, etc.
- Main output is classification accuracy printed after SVM evaluation.

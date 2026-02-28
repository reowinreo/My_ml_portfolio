# DAVE Utilities

This folder contains a conversion script that uses a DAVE-style workflow for pseudo labels.

## Script

- `Convert_pseudo_labels_using_DAVE.py`
  - Loads pseudo labels from text files.
  - Builds/loads DAVE-related model settings.
  - Selects exemplars and runs inference-style processing.
  - Exports converted predictions and intermediate artifacts.

## Notes

- The script assumes specific local directory structures for datasets and pretrained weights.
- It includes compatibility patches for some runtime/library edge cases.
- This is an experiment-oriented script and may require local adaptation before reuse.

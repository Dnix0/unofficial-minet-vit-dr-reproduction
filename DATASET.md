# Dataset access and preparation

This repository does not contain APTOS 2019 or EyePACS images, labels, or archives.

## APTOS 2019

Obtain the dataset from the authorized Kaggle source referenced by the original paper. Users are responsible for accepting and following the applicable dataset and platform terms.

Expected local manifest columns:

- `image_id`
- `image_path`
- `label`
- `source`

The final reproduction used a label-stratified image-level split:

- Train: 2,563 images
- Validation: 366 images
- Test: 733 images

Do not commit local manifests containing absolute paths. Provide only a synthetic example or a script that regenerates the manifests after the user supplies a local dataset path.

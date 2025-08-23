# Simpsons Character Classification

## At a glance
*   **Task**: Multi-class image classification of cartoon characters.
*   **Data**: "Journey to the Springfields" ([Kaggle](https://www.kaggle.com/c/advanced-dls-spring-2021/)).
*   **Model/Approach**: Transfer learning using a pre-trained [EfficientNetV2-S](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k) model with a custom classification head. Training involves a two-phase fine-tuning strategy.
*   **Key metric(s)**: Micro F1-Score.

## Quick start
*   [Open notebook](./dl_cv_classification_simpsons/main.ipynb)

## How it works
*   Images are resized to 224x224 px and normalized. The training set is augmented using techniques like `RandAugment`, `CutMix`, and `MixUp`.
*   An `EfficientNetV2-S` model pre-trained on ImageNet-21k is adapted by replacing its final layer with a custom classifier.
*   Training is conducted in two phases: first, only the new classifier head is trained for 5 epochs; second, the entire model is unfrozen and fine-tuned for up to 15 epochs.
*   The training process uses an AdamW optimizer, a cosine annealing learning rate scheduler, and early stopping to prevent overfitting.

## Results
*   The model achieves a micro F1-score of **0.99574** on the validation set.
*   Training and validation loss/accuracy curves show effective learning and generalization before early stopping.
*   Qualitative results include visualizations of correct and incorrect predictions, confirming high model confidence on most validation samples.

[← Back to Showcase](./README.md)

# ML Mini-Projects Showcase

## Table of Contents

- [Simpsons Character Classification](#simpsons-character-classification)
  - [At a glance](#at-a-glance)
  - [Quick start](#quick-start)
  - [How it works](#how-it-works)
  - [Results](#results)
- [Image Segmentation with CNNs](#image-segmentation-with-cnns)
  - [At a glance](#at-a-glance-1)
  - [Quick start](#quick-start-1)
  - [How it works](#how-it-works-1)
  - [Results](#results-1)
  - [Citation](#citation)
- [Customer Churn Prediction](#customer-churn-prediction)
  - [At a glance](#at-a-glance-2)
  - [Quick start](#quick-start-2)
  - [How it works](#how-it-works-2)
  - [Results](#results-2)
- [Game of Thrones Character Survival Prediction](#game-of-thrones-character-survival-prediction)
  - [At a glance](#at-a-glance-3)
  - [Quick start](#quick-start-3)
  - [How it works](#how-it-works-3)
  - [Results](#results-3)

## Simpsons Character Classification

### At a glance
*   **Task**: Multi-class image classification of cartoon characters.
*   **Data**: "Journey to the Springfields" ([Kaggle](https://www.kaggle.com/competitions/journey-springfield)).
*   **Model/Approach**: Transfer learning using a pre-trained [EfficientNetV2-S](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k) model with a custom classification head. Training involves a two-phase fine-tuning strategy.
*   **Key metric(s)**: Micro F1-Score.
*   **Runtime/hardware**: Models were trained on a CUDA-enabled GPU.

### Quick start
*   [Open notebook](./dl_cv_classification_simpsons/main.ipynb)

### How it works
*   Images are resized to 224x224 px and normalized. The training set is augmented using techniques like `RandAugment`, `CutMix`, and `MixUp`.
*   An `EfficientNetV2-S` model pre-trained on ImageNet-21k is adapted by replacing its final layer with a custom classifier.
*   Training is conducted in two phases: first, only the new classifier head is trained for 5 epochs; second, the entire model is unfrozen and fine-tuned for up to 15 epochs.
*   The training process uses an AdamW optimizer, a cosine annealing learning rate scheduler, and early stopping to prevent overfitting.

### Results
*   The model achieves a micro F1-score of **0.99574** on the test set.
*   Training and validation loss/accuracy curves show effective learning and generalization before early stopping.
*   Qualitative results include visualizations of correct and incorrect predictions, confirming high model confidence on most validation samples.

## Image Segmentation with CNNs

### At a glance
*   **Task**: Binary segmentation of skin lesions from dermoscopic images.
*   **Data**: PH² Dataset, a public database of 200 annotated dermoscopic images ([ADDI project](https://www.fc.up.pt/addi/ph2%20database.html)).
*   **Model/Approach**: Compared SegNet and U-Net encoder-decoder architectures trained with BCE, Dice, and Focal loss functions.
*   **Key metric(s)**: Intersection over Union (IoU / Jaccard Index).
*   **Runtime/hardware**: Models were trained on a CUDA-enabled GPU.

### Quick start
*   [Open notebook](./dl_semantic_segmentation_medical/main.ipynb)

### How it works
*   The PH² dataset, containing 200 images and corresponding lesion masks, is loaded and resized to 256x256 pixels.
*   The data is split into training (100), validation (50), and test (50) sets.
*   Two convolutional neural network models for semantic segmentation, SegNet and U-Net, are implemented in PyTorch.
*   Models are trained using an AdamW optimizer and evaluated with three different loss functions: `BCEWithLogitsLoss`, `DiceLoss`, and `SigmoidFocalLoss`.
*   Model performance is tracked using the Intersection over Union (IoU) metric on the validation set, with early stopping to prevent overfitting.

### Results
*   The SegNet architecture achieved the best overall performance in the experiments.
*   The highest score was obtained by SegNet trained with DiceLoss, reaching a test IoU of **0.804**.
*   The U-Net model's best performance was an IoU of **0.729** on the test set, achieved using `BCEWithLogitsLoss`.

### Citation
*   Badrinarayanan, V., Kendall, A., & Cipolla, R. (2015). *SegNet: A deep convolutional encoder-decoder architecture for image segmentation*.
*   Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*.
*   Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection*.

## Customer Churn Prediction

### At a glance
*   **Task**: Binary classification to predict customer churn.
*   **Data**: "Предсказание оттока пользователей" ([Kaggle](https://www.kaggle.com/c/advanced-dls-spring-2021/)).
*   **Model/Approach**: A CatBoost Classifier with hyperparameters tuned using the Optuna framework over 100 trials. The optimization process used a `TPESampler` and `HyperbandPruner`.
*   **Key metric(s)**: ROC AUC.
*   **Runtime/hardware**: Models were trained on a CUDA-enabled GPU.

### Quick start
*   [Open notebook](./ml_classification_churn/main.ipynb)

### How it works
*   The pipeline begins with data cleaning, which includes handling missing values in `TotalSpent` and correcting data types.
*   Feature engineering is performed by winsorizing `MonthlySpending` to handle outliers and creating a new interaction feature from five categorical columns.
*   An extensive hyperparameter search is conducted for a CatBoost model using Optuna to maximize the mean ROC AUC score across a 10-fold stratified cross-validation.
*   The final model is trained on the complete dataset using the best parameters found during the optimization study.

### Results
*   The best hyperparameter configuration achieved a mean ROC AUC of 0.848 during cross-validation.
*   On the hold-out test set, the final model scored a ROC AUC of 0.859.

## Game of Thrones Character Survival Prediction

### At a glance
*   **Task**: Binary classification to predict the survival (`isAlive`) of characters in the *Game of Thrones* universe.
*   **Data**: A character dataset sourced from [A Wiki of Ice and Fire](http://awoiaf.westeros.org/), containing features like house, culture, age, and popularity.
*   **Model/Approach**: A `RandomForestClassifier` was selected after a cross-validated comparison of multiple classical ML models. Hyperparameters for the model and preprocessing pipeline were tuned using `RandomizedSearchCV`.
*   **Key metric(s)**: ROC AUC was used for optimization due to class imbalance (78% alive). Final evaluation also includes Accuracy.
*   **Runtime/hardware**: Models were trained on a CUDA-enabled GPU.

### Quick start
*   [Open notebook](./ml_classification_game_of_thrones/main.ipynb)

### How it works
*   The preprocessing pipeline drops uninformative columns and features with a high percentage of missing values (over 85%).
*   New binary features are engineered from raw data, including `isPopular` (from a popularity score) and `boolDeadRelations`. The high-cardinality `culture` feature is simplified by grouping related entries.
*   Categorical data is encoded using a target encoder with Bayesian smoothing. Numerical features are imputed with the median and standardized.
*   Multiple classifiers were benchmarked via 10-fold stratified cross-validation, with `RandomForestClassifier` ultimately selected for final tuning based on its strong performance.

### Results
*   The final tuned model achieved a **ROC AUC of 0.7953** on the held-out test set.
*   The model reached a final **accuracy of 0.8045**, significantly outperforming the baseline accuracy of 0.7788.

<!-- [← Back to Showcase](./README.md) -->

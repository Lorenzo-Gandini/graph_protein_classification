# Deep Learning Hackaton 2025

This repository is heavily reliant on the [winning solution](https://sites.google.com/view/learning-with-noisy-graph-labe/winners) of the challenge [IJCNN 2025 Competition: Learning with Noisy Graph Labels](https://sites.google.com/view/learning-with-noisy-graph-labe?usp=sharing).
 The approach leverages a **Variational Graph Autoencoder (VGAE)** to filter noisy data, a **ensemble of models** strategy to handle different types of noise, and a **weighted voting mechanism** to improve prediction accuracy.

In addition to the methods proposed in the original repository, new losses are explored: SymmetricCrossEntropy (Wang et al. 2019), GeneralizedCrossEntropy (Zhang et al. 2018), Graph Centroid Outlier Discounting (Wani et al. 2023) <br>
Since Graph Centroid Outlier Discounting consistently produced the best results, we adopted it as our primary loss function.
Several other techinques are explored, but not supported by the main script: Curriculum Learning, Co-Teaching, DivideMix <br>
Authors: <br>
   - Flavio Ialongo 2000932
   - Lorenzo Gandini 2235512

---
## Our Procedure 

![alt text](https://github.com/Lorenzo-Gandini/graph_protein_classification/blob/main/method_illustration.png "Method Illustration")
---
## Overview of the Method

The method consists of four key components to handle noisy labels effectively:

1. **Variational Graph Autoencoder (VGAE):**
   - The VGAE is used to filter noisy data by retaining only real patterns in the bottleneck. This is analogous to PCA but operates nonlinearly, making it suitable for complex graph-structured data.

2. **Dropout Regularization:**
   - A 5% dropout is applied to prevent the model from over-relying on potentially noisy features, ensuring robustness.

3. **Simulated Weak Pretraining and Data Augmentation:**
   - A general model is pretrained on all datasets (A, B, C, D) to emulate weak pretraining and large-scale data augmentation. This helps the model generalize better across different types of noise.

4. **Weighted Ensemble of Models:**
   - Multiple models are trained on different subsets of the data, each potentially capturing different noise patterns. The final prediction is a weighted average of the predictions from these models, with weights determined by their F1 scores. This ensemble approach improves robustness to noise and enhances prediction accuracy.

---

## Procedure

1. **Data Preparation:**
   - The datasets (A, B, C, D) are loaded and preprocessed into graph structures.
   - Each graph is represented with node features, edge indices, and edge attributes.

2. **Initial Pretraining on All Datasets:**
   - The model is first pretrained on all datasets (A, B, C, D) to learn general patterns and noise characteristics. This pretraining acts as a form of weak supervision and data augmentation, allowing the model to generalize better when fine-tuned on individual datasets.
   - Example command for pretraining:
     ```bash
     python main.py --train_path "../A/train.json.gz ../B/train.json.gz ../C/train.json.gz ../D/train.json.gz" --num_cycles 5
     ```
   - This generates a pretrained model file (e.g., `model_paths_ABCD.txt`) that can be used for fine-tuning on specific datasets.

3. **Fine-Tuning on Individual Datasets:**
   - After pretraining, the model is fine-tuned on individual datasets (e.g., dataset A) using the pretrained model as a starting point. This allows the model to adapt to the specific noise patterns of the target dataset while retaining the general knowledge learned during pretraining.
   - Example command for fine-tuning on dataset A:
     ```bash
     python main.py --train_path ../A/train.json.gz --test_path ../A/test.json.gz --num_cycles 5 --pretrain_paths model_paths_ABCD.txt --loss_type gcod
     ```

4. **Prediction:**
   - The ensemble of models is used to predict on the test set.
   - Predictions from each model are combined using weighted voting, where the weights are the F1 scores of the models.
---

## Usage

### Pretraining on All Datasets
To pretrain the model on all datasets (A, B, C, D) for 5 cycles:
```bash
python main.py --train_path "../A/train.json.gz ../B/train.json.gz ../C/train.json.gz ../D/train.json.gz" --num_cycles 5
```

### Fine-Tuning on a Specific Dataset
To fine-tune the model on dataset A using the pretrained model (`model_paths_ABCD.txt`):
```bash
python main.py --train_path ../A/train.json.gz --test_path ../A/test.json.gz --num_cycles 5 --pretrain_paths model_paths_ABCD.txt --loss_type gcod
```

### Prediction
To make predictions using the trained models:
```bash
python main.py --test_path ../A/test.json.gz --pretrain_paths model_paths_A.txt
```

---

## Key Components

### 1. Variational Graph Autoencoder (VGAE)
- **Encoder:** The encoder maps the input graph to a latent space, capturing the essential patterns while filtering out noise.
- **Decoder:** The decoder reconstructs the graph from the latent space, ensuring that the learned representations are meaningful.
- **Loss Function:** The loss function combines reconstruction loss, KL divergence, and classification loss to train the model effectively. This composite loss is further enhanced with Graph Centroid Outlier Discounting (GCOD) to achieve superior results.

### 2. Ensemble of Models and Weighted Voting
- The predictions from the ensemble of models are combined using weighted voting, where the weights are the F1 scores of the models. This ensures that models with better performance contribute more to the final prediction.

---

## Code Structure

- **`main.py`:** The main script for training, evaluating, and predicting with the model.
- **`EdgeVGAE`:** The core model class implementing the VGAE with a classification head.
- **`ModelTrainer`:** A utility class for training multiple cycles and managing the ensemble of models.
- **`Config`:** A configuration class for managing hyperparameters and settings.
- **`Losses`:** A file containing the implementation of the differnt losses.
---

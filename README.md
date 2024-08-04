# Text Analysis of Jane Austen vs. Non-Austen Texts

This repository contains an exploratory data analysis (EDA) notebook focused on distinguishing between texts authored by Jane Austen and non-Austen texts. The analysis leverages a dataset of text paragraphs with labeled authorship to understand linguistic patterns and characteristics that differentiate Austen's writing style.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Analysis Plan](#analysis-plan)
4. [Key Findings](#key-findings)
    - [Duplicates Analysis](#duplicates-analysis)
    - [Bias Analysis](#bias-analysis)
    - [Language Analysis](#language-analysis)
        - [Text Length](#text-length)
        - [Most Frequent Terms](#most-frequent-terms)
5. [Training Implementation](#training-implementation)
6. [Conclusion](#conclusion)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

This project aims to analyze and understand the stylistic differences between texts written by Jane Austen and other authors. By examining various linguistic features, we aim to identify characteristics that are unique or prevalent in Austen's work.

## Dataset

The dataset used in this analysis consists of text paragraphs with binary labels indicating whether the text is by Jane Austen (`1`) or not (`0`). The dataset is sourced from the Gutenberg Project and has been pre-processed for this analysis.

## Analysis Plan

The analysis follows these steps:

- **Duplicate Analysis:** Identify and remove duplicate entries within the dataset, especially those that may belong to both Austen and non-Austen categories.
- **Bias Detection:** Examine the distribution of labels to detect any bias that might influence the model's learning process.
- **Language Analysis:** Investigate linguistic features, such as text length, frequent terms, verbs, adjectives, and named entities, to uncover stylistic patterns.

## Key Findings

### Duplicates Analysis

- **Removal of Duplicates:** The dataset contained duplicates, which were identified and removed to ensure the integrity of the analysis.
- **Dual-Labeled Entries:** Instances where the same text appeared under both labels were scrutinized and addressed.

### Bias Analysis

- **Label Distribution:** The dataset exhibits a skew towards non-Austen texts, necessitating strategies to balance the dataset for unbiased model training.

### Language Analysis

#### Text Length

- Austen's texts generally exhibit a wider range of text lengths compared to non-Austen texts.
- Both Austen and non-Austen texts are predominantly shorter, with Austen's texts being, on average, 20 words longer.

#### Most Frequent Terms

- The analysis revealed distinct vocabularies between Austen and non-Austen texts, leveraging term frequency-inverse document frequency (TF-IDF) and natural language processing tools like SpaCy.

## Training Implementation

The training implementation involves using a binary classifier to distinguish between texts written by Jane Austen and other authors. The training process is tracked and logged using MLflow, which helps in monitoring model performance and saving key metrics. Below is a detailed breakdown of the training code:

### Setup

- **MLflow Configuration:** The experiment is set up using MLflow to track experiments and log model parameters, metrics, and artifacts.
- **Dependencies:** The implementation utilizes `transformers`, `torch`, and `mlflow.pytorch` for model training and evaluation.

### Training Loop

1. **Initialize Model and Optimizer:**
   - The model is trained using the AdamW optimizer with a learning rate of `5e-5`.
   - The loss function used is `nn.BCEWithLogitsLoss` to handle binary classification.

2. **Training Process:**
   - The training loop iterates over the number of specified epochs.
   - For each batch in the training data, the model performs a forward pass to compute the loss.
   - The backward pass updates the model weights, and metrics are logged at each step using MLflow.
   - Evaluation metrics (accuracy, precision, recall, F1 score) are computed on the evaluation set at each step and logged to MLflow.

3. **Learning Rate Scheduler:**
   - A linear learning rate scheduler with warmup is used to adjust the learning rate throughout training, which can help stabilize training and improve performance.

4. **Evaluation:**
   - After each epoch, the model is evaluated on the evaluation dataset.
   - Confusion matrices are saved and logged as artifacts to provide insights into model performance.
   - Final evaluation is performed on the test dataset, logging similar metrics and confusion matrices.

5. **Model Saving:**
   - The final trained model is logged to MLflow, allowing for version control and later retrieval for inference or further experimentation.


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
5. [Conclusion](#conclusion)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

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

## Conclusion

The exploratory data analysis highlights significant linguistic differences between Jane Austen's writing and other authors. These insights can guide further machine learning experiments to automatically classify texts based on authorship.

## Usage

To run the analysis, clone this repository and execute the `eda.ipynb` Jupyter Notebook. Ensure you have the necessary dependencies installed, as listed in `requirements.txt`.

```bash
git clone https://github.com/your-repo.git
cd your-repo
pip install -r requirements.txt
jupyter notebook eda.ipynb
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. For significant changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

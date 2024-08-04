import os
import numpy as np
import pandas as pd
import torch
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    default_data_collator,
    pipeline
)
import matplotlib.pyplot as plt
import seaborn as sns

def balance_dataset_labels(df, ratio=1):
    max_samples = int(df['austen'].value_counts().min())
    samples = []
    for label in df['austen'].unique():
        try:
            samples.append(df[df['austen'] == label].sample(max_samples*ratio, replace=False, random_state=42))
        except:
            samples.append(df[df['austen'] == label].sample(max_samples, replace=False, random_state=42))
    return pd.concat(samples)


def load_data(file_path='../data/gutenberg-paragraphs.json', clean=False):
    """
    Load dataset from a JSON file.

    Parameters:
    - file_path: str, path to the JSON file.
    - clean: bool, if True, clean the data.

    Returns:
    - data: DataFrame, the loaded dataset.
    """
    data = pd.read_json(file_path)
    if clean:
        data = clean_data(data)
    return data

def clean_data(data):
    """
    Clean the data by removing duplicates and outliers.

    Parameters:
    - data: DataFrame, the dataset to clean.

    Returns:
    - data_filtered: DataFrame, the cleaned dataset.
    """
    data.drop_duplicates(subset=['text'], inplace=True)
    data = add_text_lengths_to_df(data)
    not_austen = data[data['austen'] == 0]
    austen = data[data['austen'] == 1]
    not_austen_filtered = remove_outliers(not_austen, 'length')
    austen_filtered = remove_outliers(austen, 'length')
    data_filtered = pd.concat([austen_filtered, not_austen_filtered])
    data_filtered.dropna(inplace=True)    
    data_balanced = balance_dataset_labels(data_filtered)
    return data_balanced


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['labels'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # if len(index) == 1:

        #     text = [self.texts[index[0]]]
        #     label = [self.labels[index[0]]]
            
        # Get text and label
        # else:
        text = [self.texts[i] for i in index]
        label = [self.labels[i] for i in index]

        # Tokenize the text
        tokens = self.tokenizer(
            text,
            # add_special_tokens=True,        # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding='max_length',           # Pad to max_length
            truncation=True,                # Truncate to max_length
            return_tensors='pt'             # Return PyTorch tensors
        )

        # Extract input_ids and attention_mask
        input_ids = tokens['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = tokens['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def split(data, tokenizer=None):
    """
    Split data into train, eval, and test sets.

    Parameters:
    - data: DataFrame, the dataset to split.
    - preprocess: bool, if True, preprocess the data.
    - tokenizer: tokenizer object, the tokenizer to use.

    Returns:
    - train, eval, test: DataLoader or Dataset, the data splits.
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    test, eval = train_test_split(test, test_size=0.5, random_state=42)
    train = TextDataset(train[['text', 'labels']], tokenizer=tokenizer, max_length=128)
    eval = TextDataset(eval[['text', 'labels']], tokenizer=tokenizer, max_length=128)
    test = TextDataset(test[['text', 'labels']][:-1], tokenizer=tokenizer, max_length=128)

    # Create DataLoaders
    train = DataLoader(train, batch_size=32)
    eval = DataLoader(eval ,batch_size=32)
    test = DataLoader(test, batch_size=32)

    return train, eval, test

def get_language_data(df, label, max_df=0.4, min_df=0.01, k=20):
    """
    Extract language data for a given author.

    Parameters:
    - df: DataFrame, the dataset.
    - label: int, author label (0 or 1).
    - max_df: float, max document frequency for TF-IDF.
    - min_df: float, min document frequency for TF-IDF.
    - k: int, number of top terms to return.

    Returns:
    - dict, containing most frequent terms, verbs, adjectives, and names.
    """
    nlp_spacy = spacy.load('en_core_web_sm')
    nlp_spacy.max_length = 2100000 
    corpus = df[df['austen']==label]['text'].unique()

    most_freq_terms = get_most_freq_terms(corpus, max_df, min_df, k)
    most_freq_verbs = get_most_freq_verbs(nlp_spacy, corpus)
    most_freq_adjectives = get_most_freq_adjectives(nlp_spacy, corpus)
    most_freq_names = get_most_freq_names(corpus)

    return {
        "most_freq_terms": most_freq_terms,
        "most_freq_verbs": most_freq_verbs,
        "most_freq_adjectives": most_freq_adjectives,
        "most_freq_names": most_freq_names
    }

def get_most_freq_terms(corpus, max_df=0.4, min_df=0.01, k=20):
    """
    Get the most frequent terms in a corpus.

    Parameters:
    - corpus: list of str, the corpus.
    - max_df: float, max document frequency for TF-IDF.
    - min_df: float, min document frequency for TF-IDF.
    - k: int, number of top terms to return.

    Returns:
    - list, top k most frequent terms.
    """
    corpus = [i.lower() for i in corpus]
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    word_count_dict = dict(zip(feature_names, word_counts))
    sorted_word_counts = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    return [term for term, count in sorted_word_counts[:k]]

def get_most_freq_verbs(spacy_client, corpus):
    """
    Get the most frequent verbs in a corpus.

    Parameters:
    - spacy_client: spaCy model, the spaCy language model.
    - corpus: list of str, the corpus.

    Returns:
    - list, most common verbs and their frequencies.
    """
    corpus_verbs = []
    for sample in corpus:
        doc = spacy_client(sample)
        sample_verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        corpus_verbs.extend(sample_verbs)
    verb_counts = Counter(corpus_verbs)
    return verb_counts.most_common()

def get_most_freq_adjectives(spacy_client, corpus):
    """
    Get the most frequent adjectives in a corpus.

    Parameters:
    - spacy_client: spaCy model, the spaCy language model.
    - corpus: list of str, the corpus.

    Returns:
    - list, most common adjectives and their frequencies.
    """
    corpus_adjectives = []
    for sample in corpus:
        doc = spacy_client(sample)
        sample_adjectives = [token.lemma_ for token in doc if token.pos_ == 'ADJ']
        corpus_adjectives.extend(sample_adjectives)
    adjective_counts = Counter(corpus_adjectives)
    return adjective_counts.most_common()

def get_most_freq_names(corpus):
    """
    Get the most frequent named entities in a corpus.

    Parameters:
    - corpus: list of str, the corpus.

    Returns:
    - list, most common named entities and their frequencies.
    """
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp_transformers = pipeline("ner", model=model, tokenizer=tokenizer, device='mps')

    all_entities = []
    for text in corpus:
        ner_results = nlp_transformers(text)
        all_entities.extend([entity['word'] for entity in ner_results])
    entity_counts = Counter(all_entities)
    return entity_counts.most_common()

def less_than_x_words(text, x):
    """
    Check if a text has fewer than x words.

    Parameters:
    - text: str, the text to check.
    - x: int, the threshold.

    Returns:
    - bool, True if text has fewer than x words, else False.
    """
    return len(text.split()) < x

def more_than_x_words(text, x):
    """
    Check if a text has more than x words.

    Parameters:
    - text: str, the text to check.
    - x: int, the threshold.

    Returns:
    - bool, True if text has more than x words, else False.
    """
    return len(text.split()) > x

def plot_pdf_length(df, filter_outliers=True, multiplier=1.5):
    """
    Plot the probability distribution function of text lengths.

    Parameters:
    - df: DataFrame, the dataset.
    - filter_outliers: bool, if True, remove outliers.
    - multiplier: float, IQR multiplier for outlier removal.
    """
    not_austen = df[df['austen'] == 0]
    austen = df[df['austen'] == 1]

    if filter_outliers:
        not_austen = remove_outliers(not_austen, 'length', multiplier)
        austen = remove_outliers(austen, 'length', multiplier)

    not_austen_lengths = not_austen['length']    
    austen_lengths = austen['length']

    mean0, std0 = not_austen_lengths.mean(), not_austen_lengths.std()
    mean1, std1 = austen_lengths.mean(), austen_lengths.std()

    plt.figure(figsize=(14, 7))

    # Plot for non-Austen
    sns.histplot(not_austen_lengths, kde=True, color='blue', label='not austen', stat='density')
    plt.axvline(mean0, color='blue', linestyle='--')
    plt.text(mean0 + std0, 0.025, f'Mean: {mean0:.2f}\nSD: {std0:.2f}', color='blue')

    # Plot for Austen
    sns.histplot(austen_lengths, kde=True, color='orange', label='austen', stat='density')
    plt.axvline(mean1, color='orange', linestyle='--')
    plt.text(mean1 + std1, 0.025, f'Mean: {mean1:.2f}\nSD: {std1:.2f}', color='orange')

    plt.legend()
    plt.title('Probability Distribution of Text Lengths by Austen/non-Austen')
    plt.xlabel('Length of Text')
    plt.ylabel('Density')
    plt.show()

def get_length_text(text):
    """
    Get the number of words in a text.

    Parameters:
    - text: str, the text to analyze.

    Returns:
    - int, number of words in the text.
    """
    return len(text.split())

def remove_outliers(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame based on a column.

    Parameters:
    - df: DataFrame, the dataset.
    - column: str, the column to check for outliers.
    - multiplier: float, IQR multiplier for outlier removal.

    Returns:
    - filtered_df: DataFrame, dataset without outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def add_text_lengths_to_df(df):
    """
    Add a 'length' column to a DataFrame indicating the number of words in 'text' column.

    Parameters:
    - df: DataFrame, the dataset.

    Returns:
    - df: DataFrame, with added 'length' column.
    """
    df['length'] = df['text'].apply(get_length_text)
    return df

def plot_pdf_language(author_data, key, author_name='Author', top_k=20):
    """
    Plot the probability distribution function of language features.

    Parameters:
    - author_data: dict, dictionary containing language features.
    - key: str, key to access specific language feature.
    - author_name: str, name of the author for labeling.
    - top_k: int, number of top features to plot.
    """
    if key not in author_data:
        raise ValueError(f"Key '{key}' not found in the author dictionary.")
    
    data = author_data[key]

    plt.figure(figsize=(14, 7))

    if isinstance(data, list):
        if isinstance(data[0], tuple):  # e.g., most_freq_verbs, most_freq_names
            data = sorted(data, key=lambda x: x[1], reverse=True)[:top_k]
            data_values = [item[1] for item in data]
            data_labels = [item[0] for item in data]

            sns.histplot(data_values, bins=10, kde=False, color='blue', label=f'{author_name}', alpha=0.5)
            sns.kdeplot(data_values, color='blue')
            
            plt.xticks(ticks=range(len(data_labels)), labels=data_labels, rotation=90)
            plt.xlabel('Words')
            plt.ylabel('Frequency')

            for i in range(len(data_values)):
                plt.text(i, data_values[i], str(data_values[i]), ha='center', va='bottom')
        else:  # e.g., most_freq_terms
            data = data[:top_k]
            sns.countplot(x=data, palette='viridis')
            plt.xticks(rotation=90)
            plt.xlabel('Words')
            plt.ylabel('Frequency')
    
    plt.title(f'Top {top_k} {key} for {author_name}')
    plt.legend()
    plt.show()

def plot_term_freq(data_dict, key, k=10):
    """
    Plot the frequency of the k most frequent terms for a given key from a dictionary.

    Parameters:
    - data_dict: dict, containing term frequency data.
    - key: str, key to access specific term frequency data.
    - k: int, number of top frequent terms to plot.
    """
    if key not in data_dict:
        print(f"Key '{key}' not found in the dictionary. Available keys are: {list(data_dict.keys())}")
        return

    term_freq = data_dict[key]
    # remove stop words
    term_freq = [item for item in term_freq if item[0] not in stopwords.words('english')]
    top_terms = sorted(term_freq, key=lambda x: x[1], reverse=True)[:k]
    terms, frequencies = zip(*top_terms)

    plt.figure(figsize=(10, 6))
    plt.bar(terms, frequencies)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Terms')
    plt.ylabel('Frequency')
    plt.title(f'Top {k} Most Frequent Terms for Key: {key}')
    plt.tight_layout()
    plt.show()


# Austen Text Classification

This repository contains my code and analysis for classifying text as either written by Jane Austen or not.

### The Data
Analysis of the Austen text dataset, exploring characteristics and biases.

- **Minimal Duplicates**: Few duplicate entries in the dataset.
- **Class Imbalance**: Significant bias towards the non-Austen class.
- **Text Lengths**:There are some large outliers in text length. After adjustments, Austen's texts are slightly longer and more varied.

#### Probability Distribution of Text Lengths by Austen/non-Austen
<img width="961" alt="image" src="https://github.com/user-attachments/assets/70e4a4cc-a43a-4181-b342-f24ef5d8d691">

*The histogram and overlaid density plots show the distribution of text lengths for Austen and non-Austen classes. Austen's texts tend to be longer on average.*

### The Model
Training and evaluation of a binary classifier to identify Austen vs. non-Austen text.

- **Model Selection**: Selected small yet effective models from the Hugging Face MTEB leaderboard.
- **Model Training**: Opted for a verbose method of training as opposed to using a trainer. Made MLFlow tracking easier.
- **MLflow Tracking**: Used MLflow for neat experiment tracking.
- **Performance**: Achieved a high F1 score of 0.989 on the evaluation set, indicating possible data leakage.

#### Model Training and Evaluation Metrics
![image](https://github.com/user-attachments/assets/7a1f9d9c-9848-48ab-9096-b839bcf36c38)
*This plot from MLFlow shows training loss, f1 and precision after each step. The super high precision and f1 has made me suspect some data leakage.*

### If I Had More Time

- **Clustering for Test Set Selection**: Would implement agglomerative clustering to better manage the dataset split, reducing the risk of data leakage.
- **Manual Data Inspection**: More thorough manual review of the dataset to identify and correct issues affecting performance.
- **Code Refactoring**: Organise the code into classes such as `Load`, `Process`, `Analyse` for better modularity and maintenance. I'd also add in tests.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/yourusername/austen-text-classification.git](https://github.com/goodaytar/recruitment-ml-leedudek.git)
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Data Preprocessing**:
   Open up the notebooks/eda.ipynb file and run the cells
   
4. **Train the Model**:
   Open up the notebooks/train.ipynb file
5. **Evaluate the Model**:
   After running the code in the notebooks/train.ipynb file, you can visit the MLFlow dashboard. First run this to start the server
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5003
   ```
   then go to http://localhost:5003 to track the progress of the model being fine tuned

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

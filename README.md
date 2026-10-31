# Research Article Classification

A machine learning project that automatically classifies research paper abstracts into three academic domains:

- **Science**
- **Business**
- **Social Science**

The project compares six machine learning algorithms, evaluates their performance, and provides an interactive Streamlit application for classifying individual research abstracts or performing batch predictions using CSV files.

## Live Demo

🚀 **Try the application:**  
https://research-paper-classifier-omgcr9q8y8kvtbpf4pvq9f.streamlit.app/

The application provides:
- Model performance dashboard
- Comparison of six machine learning algorithms
- Confusion matrix visualization
- Single research abstract prediction
- Batch prediction through CSV upload
- Downloadable prediction results

## Project Overview

Research articles are published across a wide range of academic disciplines. Manually identifying the domain of a research paper can be time-consuming, especially when dealing with large collections of publications.

This project uses Natural Language Processing (NLP) and supervised machine learning techniques to classify research paper abstracts into one of three predefined categories.

The classification is based on the textual content of research abstracts rather than manually assigned keywords.

## Dataset

The dataset was collected from **Scopus** and consists of research article and review abstracts.

| Category | Number of Abstracts |
|---|---:|
| Science | 1,983 |
| Social Science | 1,766 |
| Business | 1,575 |
| **Total** | **5,324** |

### Dataset Preprocessing

The following preprocessing operations were performed:

1. Combined the three domain-specific datasets.
2. Removed records with missing abstracts.
3. Retained only Article and Review document types.
4. Removed duplicate records using EID.
5. Identified and removed records with conflicting domain labels.
6. Removed duplicate abstracts across categories.
7. Excluded abstracts containing fewer than 30 words.

The final dataset contains **5,324 labeled research abstracts**.

## Methodology

The project follows a standard NLP-based text classification pipeline.

### 1. Text Preprocessing

The following NLP techniques are applied:

- Lowercasing
- Regular expression-based tokenization
- Stopword removal using NLTK
- WordNet lemmatization

### 2. Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert textual abstracts into numerical feature vectors.

Configuration:

| Parameter | Value |
|---|---|
| N-gram range | (1, 2) |
| Maximum features | 20,000 |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |
| Sublinear TF | True |

Unigrams and bigrams are considered to capture both individual words and meaningful two-word expressions.

### 3. Machine Learning Algorithms

Six classification algorithms are implemented and compared:

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model suitable for high-dimensional text data |
| Decision Tree | Tree-based model that learns decision rules |
| Multinomial Naive Bayes | Probabilistic classifier commonly used for text classification |
| K-Nearest Neighbors | Classifies documents based on similarity with neighboring samples |
| Support Vector Machine | Linear SVM classifier using LinearSVC |
| Random Forest | Ensemble of decision trees |

### 4. Model Training and Selection

- Dataset is divided into training and testing sets using an 80:20 stratified split.
- Stratified 3-fold cross-validation is performed on the training data.
- Macro-F1 score is used as the primary model selection metric.
- The model with the highest mean cross-validation Macro-F1 score is selected.
- The selected model is evaluated on the held-out test dataset.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified abstracts.
- **Precision:** Proportion of predicted positive classifications that are correct.
- **Recall:** Proportion of actual positive classifications correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Macro-F1:** Average F1-score across all classes, giving equal importance to each class.

The project also generates:

- Classification report
- Confusion matrix
- Model comparison visualization
- Cross-validation performance comparison

All evaluation results are generated when the training pipeline is executed.

## Project Structure

```text
research_article_classifier/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Research_Article_Classification.ipynb
│
├── data/
│   └── cleaned_scopus_research_abstracts.csv
│
├── artifacts/
│   └── Generated model and evaluation artifacts
│
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── modeling.py
    ├── predict.py
    ├── text_preprocessing.py
    └── train.py
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/DEMONRISHABH/research-paper-classifier.git

cd research-paper-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train and Evaluate Models

Run the training pipeline:

```bash
python -m src.train
```

This will:

- Load and preprocess the dataset.
- Generate TF-IDF features.
- Train all six machine learning algorithms.
- Perform stratified cross-validation.
- Select the best-performing model using Macro-F1.
- Evaluate the selected model on the test dataset.
- Save the trained model and evaluation results.
- Generate model comparison and confusion matrix visualizations.

Generated artifacts are saved in the `artifacts/` directory.

### 5. Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## Application Features

### 1. Performance Dashboard

The dashboard displays:

- Selected machine learning model
- Test accuracy
- Precision, recall and F1-score
- Model comparison
- Confusion matrix
- Dataset and train/test distribution

### 2. Single Abstract Classification

Users can:

- Paste a research paper abstract.
- Upload a `.txt` file containing an abstract.
- Generate a predicted academic category.

The application predicts one of the following:

- Science
- Business
- Social Science

### 3. Batch Classification

Users can upload a CSV file containing multiple research abstracts.

The application allows users to:

1. Upload a CSV file.
2. Select the column containing abstracts.
3. Generate predictions for all records.
4. Download the resulting CSV with predicted categories.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset manipulation |
| NumPy | Numerical operations |
| NLTK | Text preprocessing and lemmatization |
| Scikit-learn | Machine learning and evaluation |
| TF-IDF | Text feature extraction |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Streamlit | Interactive web application |
| Joblib | Model serialization |
| Google Colab / Jupyter Notebook | Model development and experimentation |

## Limitations

- The model is trained on Scopus research abstracts and may not generalize equally well to papers from other publication sources.
- Classification depends on the quality and completeness of the abstract.
- Some research papers naturally belong to multiple academic disciplines, whereas this project uses a single-label classification approach.
- TF-IDF captures word-frequency patterns but does not fully understand contextual meaning or deeper semantic relationships.
- Model performance may vary depending on the distribution of unseen research abstracts.

## Future Improvements

- Experiment with transformer-based models such as BERT and SciBERT.
- Explore contextual embeddings for improved semantic understanding.
- Extend the classifier to support multi-label classification.
- Increase dataset diversity across academic domains.
- Introduce confidence scores and explainable predictions.
- Deploy the application with a scalable backend.

## Contributors

**Rishabh Sachan**

M.Tech Data Science  
Indian Institute of Technology Roorkee

GitHub: [DEMONRISHABH](https://github.com/DEMONRISHABH)

---

⭐ If you find this project useful, consider giving the repository a star!

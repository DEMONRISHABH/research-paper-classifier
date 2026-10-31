# Research Article Classification

An end-to-end machine learning project that classifies research-paper abstracts into:

- Science
- Business
- Social Science

The project compares six classifiers required by the assignment: Logistic Regression, Decision Tree, Naive Bayes, k-Nearest Neighbors (kNN), Support Vector Machine (SVM), and Random Forest.

## Project structure

```text
research_article_classifier/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── cleaned_scopus_research_abstracts.csv
├── artifacts/
│   └── (generated after training)
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── modeling.py
    ├── predict.py
    ├── text_preprocessing.py
    └── train.py
```

## Dataset

The included cleaned dataset is derived from the three Scopus-exported category files supplied for this assignment. It contains an `Abstract` column and a `label` column.

The labels are:

- `Science`
- `Business`
- `Social Science`

The training script reads the CSV, validates required columns, removes empty abstracts and duplicate abstracts, and uses a stratified train-test split.

Before submission, retain the original Scopus export provenance and follow the applicable Scopus/Elsevier terms for storage, redistribution and publication of abstracts.

## Models

The following six models are evaluated:

1. Logistic Regression
2. Decision Tree
3. Multinomial Naive Bayes
4. k-Nearest Neighbors
5. Linear Support Vector Machine
6. Random Forest

## NLP and feature extraction

NLTK is used for:

- Lowercase normalization
- Regular-expression tokenization
- English stopword removal
- WordNet lemmatization

TF-IDF converts the cleaned tokens into numerical text features using unigrams and bigrams. The TF-IDF vectorizer is inside each scikit-learn Pipeline, so it is fitted separately inside each cross-validation fold and only on training data during final evaluation.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**macOS/Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

NLTK resources are downloaded automatically when the preprocessing module is first used. If your environment blocks automatic downloads, run:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Train and compare models

From the project root:

```bash
python -m src.train
```

Optional arguments:

```bash
python -m src.train --data data/cleaned_scopus_research_abstracts.csv --test-size 0.2 --cv-folds 3 --random-state 42
```

The script:

1. Loads and validates the labeled dataset.
2. Creates a stratified train-test split.
3. Runs stratified cross-validation for each classifier.
4. Selects the model with the highest mean cross-validation Macro-F1.
5. Fits the selected model on the full training split.
6. Evaluates it once on the held-out test split.
7. Saves the model, metrics, classification report, model comparison and confusion matrix.

Generated files are saved in `artifacts/`:

- `best_model.joblib`
- `results.json`
- `model_comparison.csv`
- `model_comparison.png`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`

The test set is not used for model selection. Macro-F1 is the primary selection metric because it gives equal weight to each class.

## Run the Streamlit app

Train the models first. Then run:

```bash
streamlit run app.py
```

### App features

**Landing page**
- Selected model and held-out test metrics
- Sample counts for Science, Business and Social Science
- Cross-validation model comparison
- Confusion matrix for the selected model

**Single abstract prediction**
- Upload one `.txt` file or paste an abstract
- Receive the predicted category
- Display a confidence value only when the selected estimator supports `predict_proba`

**Batch CSV prediction**
- Upload a CSV containing an abstract column
- Select the abstract column if it is not automatically detected
- Predict labels for all non-empty rows
- Download a CSV containing the original data and predicted labels

## Google Colab notebook

The companion notebook `Research_Article_Classification.ipynb` contains the exploratory data analysis, preprocessing, cross-validation, model comparison, final evaluation and artifact export workflow.

Upload `cleaned_scopus_research_abstracts.csv` when prompted if it is not already in the Colab working directory.

## Methodological notes and limitations

- The three labels are inherited from the supplied source datasets. Interdisciplinary papers may not have an unambiguous single label.
- The cleaned dataset should be checked for duplicate abstracts and conflicting identifiers before final submission.
- A stratified split preserves class proportions but does not remove possible source, journal or author-related bias.
- TF-IDF is a sparse lexical representation, not a contextual language-model embedding.
- The classifier predicts one of the three classes; it does not establish the true disciplinary identity of an interdisciplinary paper.
- Confidence values from `predict_proba` are model scores and should not automatically be interpreted as calibrated probabilities.
- Results must be generated by running the notebook or training script; do not report illustrative or invented scores.

## Reproducibility

The random seed defaults to 42. The training script records class counts, split sizes, cross-validation scores and held-out test metrics in the generated artifacts.

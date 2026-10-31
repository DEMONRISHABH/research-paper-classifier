from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from .text_preprocessing import nltk_tokenize

def build_vectorizer():
    return TfidfVectorizer(
        tokenizer=nltk_tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype="float32"
    )

def build_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=40,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "kNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine",
            algorithm="brute",
            n_jobs=-1
        ),
        "SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        )
    }

def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("classifier", classifier)
    ])

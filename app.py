import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.predict import load_model, predict_abstracts

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "results.json"
MODEL_PATH = ARTIFACTS / "best_model.joblib"

st.set_page_config(
    page_title="Research Article Classifier",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Article Classification")
st.caption("Classify research abstracts into Science, Business and Social Science.")

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

def read_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

results = read_results()

if results is None or not MODEL_PATH.exists():
    st.warning(
        "Model artifacts are not available yet. Train the models first by running "
        "`python -m src.train` from the project directory."
    )
    st.stop()

model = get_model()

st.header("Model performance dashboard")

metrics = results["metrics"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected model", results["selected_model"])
m2.metric("Test accuracy", f"{metrics['accuracy']:.3f}")
m3.metric("Macro precision", f"{metrics['precision_macro']:.3f}")
m4.metric("Macro F1-score", f"{metrics['f1_macro']:.3f}")

left, right = st.columns(2)

with left:
    st.subheader("Dataset sample counts")
    counts = pd.Series(results["class_counts"], name="Samples")
    st.bar_chart(counts)
    split_counts = pd.DataFrame({
        "Train samples": pd.Series(results["train_class_counts"]),
        "Test samples": pd.Series(results["test_class_counts"])
    }).fillna(0).astype(int)
    st.caption("Class counts in the full modeling dataset and the stratified split.")
    st.dataframe(split_counts, use_container_width=True)
    st.caption(
        f"Total samples used: {results['dataset_rows']:,} | "
        f"Train: {results['train_rows']:,} | Test: {results['test_rows']:,}"
    )

with right:
    st.subheader("Classifier comparison")
    comparison_path = ARTIFACTS / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(
            comparison.style.format({
                "CV Macro-F1 Mean": "{:.4f}",
                "CV Macro-F1 Std": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    image_path = ARTIFACTS / "model_comparison.png"
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

st.subheader("Confusion matrix")
cm_path = ARTIFACTS / "confusion_matrix.png"
if cm_path.exists():
    st.image(str(cm_path), use_container_width=True)
cm_csv = ARTIFACTS / "confusion_matrix.csv"
if cm_csv.exists():
    with st.expander("View confusion matrix values"):
        st.dataframe(pd.read_csv(cm_csv, index_col=0), use_container_width=True)

st.divider()
st.header("Make predictions")

single_tab, batch_tab = st.tabs([
    "Single abstract prediction",
    "Batch CSV prediction"
])

with single_tab:
    st.subheader("Predict one research abstract")
    uploaded_txt = st.file_uploader(
        "Upload a .txt file containing one abstract",
        type=["txt"],
        key="single_abstract_file"
    )

    uploaded_text = ""
    if uploaded_txt is not None:
        uploaded_text = uploaded_txt.getvalue().decode("utf-8", errors="replace")

    abstract = st.text_area(
        "Or paste the abstract here",
        value=uploaded_text,
        height=220,
        placeholder="Paste a research paper abstract..."
    )

    if st.button("Predict category", type="primary", key="single_predict"):
        if not abstract.strip():
            st.error("Please upload a text file or enter an abstract.")
        else:
            prediction = predict_abstracts([abstract], model=model)[0]
            st.success(f"Predicted category: **{prediction['predicted_label']}**")
            if "confidence" in prediction:
                st.metric("Model confidence", f"{prediction['confidence']:.1%}")
            else:
                st.info(
                    "The selected classifier does not provide calibrated "
                    "probabilities; no probability estimate is displayed."
                )

with batch_tab:
    st.subheader("Predict categories for a CSV file")
    st.write("Upload a CSV containing a column with research abstracts.")
    batch_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_csv_file"
    )

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        st.write("Preview:")
        st.dataframe(batch_df.head(), use_container_width=True)

        possible_columns = [
            col for col in batch_df.columns
            if str(col).strip().lower() in ["abstract", "text", "abstract_text"]
        ]

        if not possible_columns:
            selected_column = st.selectbox(
                "Select the abstract text column",
                options=list(batch_df.columns)
            )
        else:
            selected_column = st.selectbox(
                "Select the abstract text column",
                options=list(batch_df.columns),
                index=list(batch_df.columns).index(possible_columns[0])
            )

        if st.button("Predict batch", type="primary"):
            texts = batch_df[selected_column].fillna("").astype(str).tolist()
            valid_mask = [bool(text.strip()) for text in texts]

            predictions = [None] * len(texts)
            valid_texts = [text for text, valid in zip(texts, valid_mask) if valid]

            if valid_texts:
                valid_predictions = predict_abstracts(valid_texts, model=model)
                iterator = iter(valid_predictions)
                for index, valid in enumerate(valid_mask):
                    if valid:
                        predictions[index] = next(iterator)

            batch_df["predicted_label"] = [
                item["predicted_label"] if item else ""
                for item in predictions
            ]

            if any(item and "confidence" in item for item in predictions):
                batch_df["prediction_confidence"] = [
                    item.get("confidence", None) if item else None
                    for item in predictions
                ]

            st.success(f"Predictions completed for {len(valid_texts):,} non-empty abstracts.")
            st.dataframe(batch_df, use_container_width=True)

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv_bytes,
                file_name="research_abstract_predictions.csv",
                mime="text/csv"
            )

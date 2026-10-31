from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"

def load_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )
    return joblib.load(model_path)

def predict_abstracts(abstracts, model=None):
    model = model or load_model()
    cleaned = [str(item) for item in abstracts]
    predictions = model.predict(cleaned)

    output = []
    classifier = model.named_steps["classifier"]

    probabilities = None
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(cleaned)

    for index, prediction in enumerate(predictions):
        row = {"predicted_label": str(prediction)}
        if probabilities is not None:
            row["confidence"] = float(probabilities[index].max())
        output.append(row)

    return output

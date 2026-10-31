import pandas as pd

LABELS = ["Science", "Business", "Social Science"]

def find_column(df, candidates):
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def load_dataset(path):
    df = pd.read_csv(path)
    abstract_col = find_column(df, ["Abstract", "abstract", "text"])
    label_col = find_column(df, ["label", "category", "class"])

    if abstract_col is None:
        raise ValueError("Dataset must contain an 'Abstract' column.")
    if label_col is None:
        raise ValueError("Dataset must contain a 'label' column.")

    eid_col = find_column(df, ["EID", "eid", "Scopus ID"])
    doc_type_col = find_column(df, ["Document Type", "document_type"])

    if eid_col is not None:
        conflicting_eids = (
            df.dropna(subset=[eid_col])
              .groupby(eid_col)[label_col]
              .nunique()
        )
        conflicting_eids = conflicting_eids[conflicting_eids > 1].index
        df = df[~df[eid_col].isin(conflicting_eids)].copy()

    if doc_type_col is not None:
        df = df[
            df[doc_type_col].astype(str).str.strip().str.lower().isin(
                ["article", "review"]
            )
        ].copy()

    data = df[[abstract_col, label_col]].copy()
    data.columns = ["abstract", "label"]
    data["abstract"] = data["abstract"].fillna("").astype(str).str.strip()
    data["label"] = data["label"].fillna("").astype(str).str.strip()

    data = data[
        (data["abstract"].str.len() > 0) &
        (data["label"].isin(LABELS))
    ].copy()

    normalized = data["abstract"].str.lower().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    data["_normalized"] = normalized
    conflicting_abstracts = (
        data.groupby("_normalized")["label"].nunique()
    )
    conflicting_abstracts = conflicting_abstracts[
        conflicting_abstracts > 1
    ].index

    data = data[~data["_normalized"].isin(conflicting_abstracts)]
    data = data.drop_duplicates(subset=["_normalized"])
    data = data.drop(columns=["_normalized"]).reset_index(drop=True)

    if data["label"].nunique() < 2:
        raise ValueError("At least two valid classes are required.")

    return data

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_RESOURCE_READY = False
_STOP_WORDS = None
_LEMMATIZER = WordNetLemmatizer()
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

def ensure_nltk_resources(download=True):
    global _RESOURCE_READY, _STOP_WORDS
    if _RESOURCE_READY:
        return

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not download:
                raise RuntimeError(
                    f"NLTK resource '{package}' is missing. "
                    f"Run: python -c \"import nltk; nltk.download('{package}')\""
                )
            nltk.download(package, quiet=True)

    _STOP_WORDS = set(stopwords.words('english'))
    _RESOURCE_READY = True

def nltk_tokenize(text):
    ensure_nltk_resources()
    if text is None:
        return []

    text = str(text).lower()
    tokens = _TOKEN_PATTERN.findall(text)
    stop_words = _STOP_WORDS

    cleaned = []
    for token in tokens:
        if token in stop_words or len(token) < 2:
            continue
        cleaned.append(_LEMMATIZER.lemmatize(token))

    return cleaned

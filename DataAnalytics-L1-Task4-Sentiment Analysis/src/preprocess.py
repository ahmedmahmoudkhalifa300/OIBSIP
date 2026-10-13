"""
Preprocess Module
-----------------
Handles text cleaning, emoji translation, contraction expansion,
tokenization, lemmatization, and stopword removal with sentiment preservation.
"""

import sys
import os

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure required NLTK resources are available
for resource, path in [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab')
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(path, quiet=True)

# Common English Contractions Dictionary
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "might've": "might have", "must've": "must have",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
    "who'd": "who would", "who'll": "who will", "who's": "who is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

# Negation and sentiment-critical tokens to preserve when removing stopwords
SENTIMENT_PRESERVE_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing', 'nowhere',
    'very', 'most', 'more', 'too', 'against', 'but', 'however', 'cannot', 'good', 'bad'
}

DEFAULT_STOPWORDS = set(stopwords.words('english')) - SENTIMENT_PRESERVE_WORDS


def expand_contractions(text: str) -> str:
    """Expand short-form English contractions into full words."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    expanded = []
    for word in words:
        clean_word = word.lower().strip(string.punctuation)
        if clean_word in CONTRACTIONS:
            expanded.append(CONTRACTIONS[clean_word])
        else:
            expanded.append(word)
    return " ".join(expanded)


def convert_emojis(text: str) -> str:
    """Convert emoji symbols into descriptive textual representations."""
    if not isinstance(text, str):
        return ""
    # Convert emojis to text representation like :smiling_face_with_heart_eyes:
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Replace underscores with spaces
    text = text.replace("_", " ")
    return text


def clean_text(
    text: str,
    lemmatize: bool = True,
    remove_stopwords: bool = True
) -> str:
    """
    Comprehensive text cleaning pipeline:
    1. Lowercase conversion
    2. Emoji translation to words
    3. Contraction expansion
    4. URL and username mentions removal
    5. Hashtag symbol removal (preserving tag words)
    6. Non-alphanumeric character removal
    7. Lemmatization using WordNet
    8. Sentiment-aware Stopwords removal
    """
    if not isinstance(text, str):
        return ""

    # Convert to lower case
    text = text.lower()

    # Translate emojis before stripping punctuation
    text = convert_emojis(text)

    # Expand contractions
    text = expand_contractions(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove user mentions (@username)
    text = re.sub(r'@\w+', ' ', text)

    # Clean hashtag markers but keep text (e.g. #Happy -> Happy)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove non-alphabetical characters (retain spaces and standard letters)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Normalize whitespace
    tokens = text.split()

    # Lemmatizer
    lemmatizer = WordNetLemmatizer() if lemmatize else None

    processed_tokens = []
    for token in tokens:
        # Filter stopwords
        if remove_stopwords and token in DEFAULT_STOPWORDS:
            continue
        if len(token) <= 1 and token not in {'a', 'i'}:
            continue
        # Lemmatize
        if lemmatizer:
            token = lemmatizer.lemmatize(token)
            # Try verb lemmatization if unchanged
            token = lemmatizer.lemmatize(token, pos='v')
        processed_tokens.append(token)

    return " ".join(processed_tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer for text preprocessing.
    Can be seamlessly plugged into an sklearn Pipeline.
    """
    def __init__(self, lemmatize: bool = True, remove_stopwords: bool = True):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "tolist"):
            X = X.tolist()
        return [clean_text(str(doc), self.lemmatize, self.remove_stopwords) for doc in X]


if __name__ == "__main__":
    sample = "Enjoying a beautiful day at the park! 😊 @User123 can't wait for #Vacation https://example.com"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))

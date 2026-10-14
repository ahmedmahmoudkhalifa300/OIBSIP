"""
Predict Module
--------------
Inference engine and interactive CLI for real-time sentiment analysis.
Can be executed as a standalone CLI tool or imported as a class in other applications.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
from typing import Union, List, Dict, Any
import joblib
import numpy as np

# Import preprocessing
from src.preprocess import clean_text


class SentimentPredictor:
    """Production-ready inference wrapper for the trained Sentiment Analysis model."""

    SENTIMENT_ICONS = {
        'Positive': '🟢 [POSITIVE]',
        'Negative': '🔴 [NEGATIVE]',
        'Neutral': '🟡 [NEUTRAL]'
    }

    def __init__(self, model_path: str = "outputs/models/best_sentiment_model.joblib"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Please run 'python src/train.py' first to train and save the model."
            )
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def predict_single(self, raw_text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single input text string.
        Returns:
            dict containing original_text, cleaned_text, predicted_sentiment,
            confidence_score, and class_probabilities (if available).
        """
        cleaned = clean_text(raw_text)
        prediction = str(self.model.predict([cleaned])[0])

        probabilities = {}
        confidence = 1.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([cleaned])[0]
            classes = self.model.classes_
            probabilities = {str(cls): round(float(p), 4) for cls, p in zip(classes, probs)}
            confidence = round(float(max(probs)), 4)
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function([cleaned])[0]
            classes = self.model.classes_
            probabilities = {str(cls): round(float(s), 4) for cls, s in zip(classes, scores)}

        badge = self.SENTIMENT_ICONS.get(prediction, f"[{prediction}]")

        return {
            "original_text": raw_text,
            "cleaned_text": cleaned,
            "predicted_sentiment": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "badge": badge
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for a list of texts."""
        return [self.predict_single(t) for t in texts]


def interactive_cli(predictor: SentimentPredictor):
    """Interactive loop for CLI text sentiment evaluation."""
    print("=" * 65)
    print("🌟 REAL-TIME SENTIMENT ANALYSIS ENGINE")
    print("Type any text or tweet to analyze its sentiment.")
    print("Type 'exit' or 'quit' to terminate.")
    print("=" * 65)

    sample_prompts = [
        "Just finished an amazing workout! Super energized! 💪",
        "The flight was delayed by 6 hours, worst service ever.",
        "Attending a virtual meeting about cloud computing."
    ]
    print("\n💡 Example test sentences you can try:")
    for i, s in enumerate(sample_prompts, 1):
        print(f"  {i}. {s}")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nEnter text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Exiting. Have a great day!")
                break

            result = predictor.predict_single(user_input)
            print("\n" + "-" * 50)
            print(f"  Result     : {result['badge']} {result['predicted_sentiment']}")
            print(f"  Confidence : {result['confidence'] * 100:.1f}%")
            if result['probabilities']:
                print("  Class Probabilities:")
                for cls, prob in result['probabilities'].items():
                    bar = "█" * int(prob * 20)
                    print(f"    - {cls:10s}: {prob:6.1%} | {bar}")
            print(f"  Cleaned text: '{result['cleaned_text']}'")
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference CLI")
    parser.add_argument("--text", type=str, help="Text string to evaluate", default=None)
    parser.add_argument("--model", type=str, default="outputs/models/best_sentiment_model.joblib", help="Path to joblib model")
    args = parser.parse_args()

    predictor = SentimentPredictor(model_path=args.model)

    if args.text:
        result = predictor.predict_single(args.text)
        print(f"\nText      : {result['original_text']}")
        print(f"Cleaned   : {result['cleaned_text']}")
        print(f"Sentiment : {result['badge']} {result['predicted_sentiment']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        if result['probabilities']:
            print("Probabilities:", result['probabilities'])
    else:
        interactive_cli(predictor)


if __name__ == "__main__":
    main()

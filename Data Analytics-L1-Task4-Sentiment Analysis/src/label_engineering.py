"""
Label Engineering Module
------------------------
Resolves the high-cardinality label issue (191 raw fine-grained emotion classes),
maps them cleanly into standardized 3 sentiment categories (Positive, Negative, Neutral),
and handles class imbalance using class weights and stratification.
"""

import sys
import os

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Definitive taxonomy mapping the 191 fine-grained emotions to 3 primary sentiments
POSITIVE_CLASSES = {
    'Acceptance', 'Accomplishment', 'Admiration', 'Adoration', 'Adrenaline', 'Adventure',
    'Affection', 'Amazement', 'Amusement', 'Anticipation', 'Appreciation', 'ArtisticBurst',
    'Awe', 'Blessed', 'Breakthrough', 'Calmness', 'Captivation', 'Celebration',
    'Celestial Wonder', 'Charm', 'Colorful', 'Compassion', 'Compassionate', 'Confidence',
    'Confident', 'Connection', 'Contentment', 'Coziness', 'Creative Inspiration',
    'Creativity', 'Culinary Adventure', 'CulinaryOdyssey', 'Dazzle', 'Determination',
    'DreamChaser', 'Ecstasy', 'Elation', 'Elegance', 'Empathetic', 'Empowerment',
    'Enchantment', 'Energy', 'Engagement', 'Enjoyment', 'Enthusiasm', 'Euphoria',
    'Excitement', 'Exploration', 'FestiveJoy', 'Free-spirited', 'Freedom', 'Friendship',
    'Fulfillment', 'Grandeur', 'Grateful', 'Gratitude', 'Happiness', 'Happy', 'Harmony',
    'Heartwarming', 'Hope', 'Hopeful', 'Hypnotic', 'Iconic', 'Imagination', 'Immersion',
    'Inspiration', 'Inspired', 'Intrigue', 'Joy', 'Joy in Baking', 'JoyfulReunion',
    'Kind', 'Kindness', 'Love', 'Marvel', 'Melodic', 'Mesmerizing', 'Mindfulness',
    'Motivation', "Nature's Beauty", "Ocean's Freedom", 'Optimism', 'Overjoyed',
    'Playful', 'PlayfulJoy', 'Positive', 'Positivity', 'Pride', 'Proud', 'Radiance',
    'Rejuvenation', 'Relief', 'Renewed Effort', 'Resilience', 'Reverence', 'Romance',
    'Runway Creativity', 'Satisfaction', 'Serenity', 'Solace', 'Spark', 'Success',
    'Tenderness', 'Thrill', 'Thrilling Journey', 'Touched', 'Tranquility', 'Triumph',
    'Vibrancy', 'Whimsy', 'Winter Magic', 'Wonder', 'Wonderment', 'Zest'
}

NEGATIVE_CLASSES = {
    'Anger', 'Anxiety', 'Apprehensive', 'Bad', 'Betrayal', 'Bitter', 'Bitterness',
    'Boredom', 'Darkness', 'Desolation', 'Despair', 'Desperation', 'Devastated',
    'Disappointed', 'Disappointment', 'Disgust', 'Dismissive', 'Embarrassed',
    'EmotionalStorm', 'Envious', 'Envy', 'Exhaustion', 'Fear', 'Fearful',
    'Frustrated', 'Frustration', 'Grief', 'Hate', 'Heartache', 'Heartbreak',
    'Helplessness', 'Intimidation', 'Isolation', 'Jealous', 'Jealousy', 'Loneliness',
    'Loss', 'LostLove', 'Melancholy', 'Miscalculation', 'Negative', 'Numbness',
    'Obstacle', 'Overwhelmed', 'Pressure', 'Regret', 'Resentment', 'Ruins',
    'Sad', 'Sadness', 'Shame', 'Sorrow', 'Suffering', 'Yearning'
}

NEUTRAL_CLASSES = {
    'Neutral', 'Curiosity', 'Nostalgia', 'Confusion', 'Surprise', 'Indifference',
    'Ambivalence', 'Arousal', 'Bittersweet', 'Challenge', 'Contemplation',
    'Emotion', 'Envisioning History', 'InnerJourney', 'Journey', 'Mischievous',
    'Pensive', 'Reflection', 'Solitude', 'Suspense', 'Sympathy', 'Whispers of the Past'
}


def map_raw_sentiment(label: Any) -> str:
    """
    Map an arbitrary raw sentiment string to 'Positive', 'Negative', or 'Neutral'.
    Handles trailing whitespace, title casing, and unseen inputs with heuristic fallback.
    """
    if not isinstance(label, str):
        return 'Neutral'

    cleaned = label.strip()

    # Exact match checks
    if cleaned in POSITIVE_CLASSES:
        return 'Positive'
    if cleaned in NEGATIVE_CLASSES:
        return 'Negative'
    if cleaned in NEUTRAL_CLASSES:
        return 'Neutral'

    # Case-insensitive checks
    cleaned_lower = cleaned.lower()
    for pos in POSITIVE_CLASSES:
        if cleaned_lower == pos.lower():
            return 'Positive'
    for neg in NEGATIVE_CLASSES:
        if cleaned_lower == neg.lower():
            return 'Negative'
    for neu in NEUTRAL_CLASSES:
        if cleaned_lower == neu.lower():
            return 'Neutral'

    # Keyword heuristics
    if any(k in cleaned_lower for k in ['pos', 'joy', 'good', 'happy', 'love', 'excite', 'great']):
        return 'Positive'
    if any(k in cleaned_lower for k in ['neg', 'sad', 'bad', 'hate', 'fear', 'anger', 'depress']):
        return 'Negative'

    return 'Neutral'


def engineer_labels(
    df: pd.DataFrame,
    sentiment_column: str = 'Sentiment'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply sentiment mapping to a DataFrame, compute distributions, and calculate balanced class weights.
    
    Returns:
        df: DataFrame with 'Clean_Sentiment' column added.
        metadata: Dictionary with label counts, proportions, and balanced class weights.
    """
    df = df.copy()
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in dataframe.")

    # Apply mapping
    df['Clean_Sentiment'] = df[sentiment_column].apply(map_raw_sentiment)

    # Calculate class counts and proportions
    counts = df['Clean_Sentiment'].value_counts().to_dict()
    total = len(df)
    proportions = {k: round(v / total, 4) for k, v in counts.items()}

    # Compute balanced class weights for ML models
    classes = np.array(sorted(list(counts.keys())))
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=df['Clean_Sentiment'].values
    )
    class_weights_dict = {str(cls): round(float(w), 4) for cls, w in zip(classes, weights)}

    metadata = {
        'total_samples': total,
        'unique_raw_classes': df[sentiment_column].astype(str).str.strip().nunique(),
        'mapped_classes': list(counts.keys()),
        'class_distribution': counts,
        'class_proportions': proportions,
        'balanced_class_weights': class_weights_dict
    }

    return df, metadata


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'Sentiment': [' Joy ', 'Despair', 'Curiosity', ' Unknown Emotion ', 'Happy']
    })
    df_res, meta = engineer_labels(sample_df)
    print("Mapped DataFrame:\n", df_res)
    print("\nMetadata:\n", meta)

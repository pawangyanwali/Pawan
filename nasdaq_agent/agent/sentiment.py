import logging
from textblob import TextBlob
from agent.data_fetcher import fetch_news

logger = logging.getLogger(__name__)

# Amplify sentiment for finance-specific keywords
BULLISH_KEYWORDS = {
    "beats", "beat", "exceeds", "record", "upgrade", "buy", "outperform",
    "growth", "surge", "rally", "breakout", "profit", "strong", "positive",
    "bullish", "upside", "partnership", "contract", "approval", "dividend",
}
BEARISH_KEYWORDS = {
    "misses", "miss", "downgrade", "sell", "underperform", "loss", "decline",
    "fall", "drop", "weak", "negative", "bearish", "lawsuit", "recall",
    "investigation", "layoff", "cut", "warning", "risk", "concern",
}


def _score_text(text: str) -> float:
    """Return a sentiment score in [-1, +1] for a piece of text."""
    if not text:
        return 0.0

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity   # base TextBlob score

    lower = text.lower().split()
    word_set = set(lower)
    bull_hits = len(word_set & BULLISH_KEYWORDS)
    bear_hits = len(word_set & BEARISH_KEYWORDS)
    keyword_boost = (bull_hits - bear_hits) * 0.15

    return max(-1.0, min(1.0, polarity + keyword_boost))


def score_sentiment(ticker: str) -> tuple[float, list[str]]:
    """
    Fetch recent news for `ticker`, score each headline, and return
    (aggregate_score, list_of_headlines).
    Score is in [-1, +1]; 0.0 is returned when no news is found.
    """
    news_items = fetch_news(ticker)
    if not news_items:
        return 0.0, []

    scores: list[float] = []
    headlines: list[str] = []

    for item in news_items:
        title = item.get("title", "")
        if not title:
            continue
        headlines.append(title)
        scores.append(_score_text(title))

    if not scores:
        return 0.0, headlines

    # Weight recent news more heavily (first item is most recent)
    weights = [1 / (i + 1) for i in range(len(scores))]
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return round(float(weighted_score), 4), headlines

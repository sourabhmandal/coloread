"""Production-oriented keyphrase extraction for PDF highlighting.

The pipeline combines package-backed NLP components:

- RAKE candidate extraction from ``rake_nltk``
- TF-IDF and latent semantic analysis from ``scikit-learn``
- lead / wrap and cue-phrase feature engineering for ranking

The output is a deterministic list of verbatim substrings from the source
document so the downstream PDF annotator can highlight them directly.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rake_nltk import Rake
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

_CUE_PHRASES = (
    "in conclusion",
    "in summary",
    "therefore",
    "thus",
    "significantly",
    "importantly",
    "notably",
    "moreover",
    "furthermore",
    "as a result",
    "the takeaway",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_MAX_INPUT_CHARS = 200_000


@dataclass(frozen=True)
class _Candidate:
    phrase: str
    score: float
    sentence_index: int
    offset: int


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", normalized).strip()


def _split_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _locate_phrase(text: str, phrase: str) -> tuple[str, int] | None:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    return text[match.start() : match.end()], match.start()


def _sentence_index_for_phrase(sentences: list[str], phrase: str) -> tuple[int, int]:
    phrase_lower = phrase.lower()
    best_index = 0
    best_offset = 0
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        offset = lowered.find(phrase_lower)
        if offset >= 0:
            return index, offset
        if index == 0:
            best_index = 0
            best_offset = 0
    return best_index, best_offset


def _build_tfidf_profile(sentences: list[str]) -> tuple[dict[str, float], np.ndarray]:
    vectorizer = TfidfVectorizer(
        stop_words=sorted(_STOPWORDS),
        lowercase=True,
        token_pattern=r"[A-Za-z][A-Za-z0-9'-]*",
        ngram_range=(1, 2),
    )

    try:
        sentence_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return {}, np.zeros(len(sentences), dtype=float)

    feature_names = vectorizer.get_feature_names_out()
    term_weights = np.asarray(sentence_matrix.mean(axis=0)).ravel()
    tfidf_profile = dict(zip(feature_names, term_weights, strict=False))

    if sentence_matrix.shape[0] < 2 or sentence_matrix.shape[1] < 2:
        sentence_scores = np.asarray(sentence_matrix.sum(axis=1)).ravel()
    else:
        n_components = min(4, sentence_matrix.shape[0] - 1, sentence_matrix.shape[1] - 1)
        if n_components < 1:
            sentence_scores = np.asarray(sentence_matrix.sum(axis=1)).ravel()
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            latent_matrix = svd.fit_transform(sentence_matrix)
            sentence_scores = np.linalg.norm(latent_matrix, axis=1)

    if sentence_scores.size:
        min_score = float(sentence_scores.min())
        max_score = float(sentence_scores.max())
        if max_score > min_score:
            sentence_scores = (sentence_scores - min_score) / (max_score - min_score)
        else:
            sentence_scores = np.ones_like(sentence_scores)

    return tfidf_profile, sentence_scores


def _phrase_tfidf_score(phrase: str, tfidf_profile: dict[str, float]) -> float:
    tokens = _tokenize_words(phrase)
    if not tokens:
        return 0.0

    unigram_score = sum(tfidf_profile.get(token, 0.0) for token in tokens)
    bigram_score = sum(
        tfidf_profile.get(f"{left} {right}", 0.0)
        for left, right in zip(tokens, tokens[1:])
    )
    return (unigram_score + bigram_score) / max(1, len(tokens))


def _cue_boost(sentence: str) -> float:
    lowered = sentence.lower().lstrip()
    if any(lowered.startswith(cue) for cue in _CUE_PHRASES):
        return 0.35
    if any(cue in lowered[:90] for cue in _CUE_PHRASES):
        return 0.15
    return 0.0


def _position_boost(sentence_index: int, sentence_count: int) -> float:
    if sentence_count <= 1:
        return 0.2

    lead_cutoff = max(1, sentence_count // 5)
    wrap_cutoff = max(1, sentence_count - lead_cutoff)
    if sentence_index < lead_cutoff or sentence_index >= wrap_cutoff:
        return 0.2
    return 0.0


def _build_candidates(text: str) -> list[_Candidate]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    tfidf_profile, sentence_scores = _build_tfidf_profile(sentences)
    rake = Rake(
        stopwords=sorted(_STOPWORDS),
        sentence_tokenizer=_split_sentences,
        word_tokenizer=_tokenize_words,
    )
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases_with_scores()

    if not ranked_phrases:
        return []

    max_rake_score = max(score for score, _ in ranked_phrases) or 1.0
    candidates: list[_Candidate] = []

    for rake_score, phrase in ranked_phrases:
        located = _locate_phrase(text, phrase)
        if not located:
            continue

        exact_phrase, offset = located
        sentence_index, _ = _sentence_index_for_phrase(sentences, phrase)
        sentence = sentences[min(sentence_index, len(sentences) - 1)]
        sentence_score = float(sentence_scores[sentence_index]) if sentence_scores.size else 0.0
        tfidf_score = _phrase_tfidf_score(phrase, tfidf_profile)
        length_bonus = min(0.3, 0.08 * max(0, len(_tokenize_words(phrase)) - 1))
        cue_bonus = _cue_boost(sentence)
        position_bonus = _position_boost(sentence_index, len(sentences))

        score = (
            0.45 * (rake_score / max_rake_score)
            + 0.25 * tfidf_score
            + 0.20 * sentence_score
            + position_bonus
            + cue_bonus
            + length_bonus
        )

        if len(_tokenize_words(phrase)) == 1:
            score *= 0.75

        candidates.append(
            _Candidate(
                phrase=exact_phrase,
                score=score,
                sentence_index=sentence_index,
                offset=offset,
            )
        )

    return candidates


def identify_highlights(text: str, model: Optional[str] = None) -> list[str]:
    """Return verbatim phrases that are worth highlighting."""
    del model

    normalized_text = _normalize_text(text[:_MAX_INPUT_CHARS])
    if not normalized_text:
        return []

    candidates = _build_candidates(normalized_text)
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.sentence_index,
            candidate.offset,
            candidate.phrase.lower(),
        ),
    )

    selected: list[_Candidate] = []
    seen: set[str] = set()

    for candidate in ranked:
        key = candidate.phrase.lower()
        if key in seen:
            continue
        if len(_tokenize_words(candidate.phrase)) > 1:
            selected.append(candidate)
            seen.add(key)

    if len(selected) < 5:
        for candidate in ranked:
            key = candidate.phrase.lower()
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= min(20, max(5, len(ranked))):
                break

    if len(selected) < 5:
        for candidate in ranked:
            key = candidate.phrase.lower()
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= min(20, max(5, len(ranked))):
                break

    return [candidate.phrase for candidate in selected[:20]]
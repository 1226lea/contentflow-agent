"""TF-IDF recommender ported from ContentFlow (core/recommendations.py).

The original works on Django model instances and also mixes in
behavioural signals (likes, posts, applications). This port keeps the
static-profile part — the same four weighted dimensions and the same
TF-IDF + cosine-similarity comparison — but takes plain dicts, so the
agent can call it as a tool.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Same dimensions and weights as the original ContentFlow recommender.
DIMENSIONS = {
    "interests": {"label": "Interests", "weight": 0.35, "strategy_field": "suitable_interests"},
    "skills": {"label": "Skills", "weight": 0.30, "strategy_field": "suitable_skills"},
    "goals": {"label": "Goals", "weight": 0.20, "strategy_field": "suitable_goals"},
    "constraints": {
        "label": "Creative preferences and resources",
        "weight": 0.15,
        "strategy_field": "suitable_constraints",
    },
}


def compare_texts(query_text: str, candidate_texts: list[str], top_terms: int = 4):
    """Return (similarities, matched_terms) of query_text against each candidate.

    Same logic as ContentFlow's _compare_dimension: fit TF-IDF on the
    query plus all candidates, take cosine similarity, and list the
    shared terms with the highest combined weight as the match reason.
    """
    empty = ([0.0] * len(candidate_texts), [[] for _ in candidate_texts])
    if not query_text or not query_text.strip():
        return empty

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([query_text, *candidate_texts])
    except ValueError:  # e.g. only stop words, or text TF-IDF can't tokenise
        return empty

    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    features = vectorizer.get_feature_names_out()
    query_weights = matrix[0].toarray().flatten()
    query_terms = {features[i] for i, w in enumerate(query_weights) if w > 0}

    matches = []
    for row in range(len(candidate_texts)):
        weights = matrix[row + 1].toarray().flatten()
        shared = [
            features[i] for i, w in enumerate(weights)
            if w > 0 and features[i] in query_terms
        ]
        shared.sort(key=lambda term: (
            -query_weights[vectorizer.vocabulary_[term]]
            * weights[vectorizer.vocabulary_[term]],
            term,
        ))
        matches.append(shared[:top_terms])

    return [float(s) for s in similarities], matches


def recommend_strategies(profile: dict, strategies: list[dict], limit: int = 3) -> list[dict]:
    """Rank strategies for a profile with keys interests/skills/goals/constraints."""
    if not strategies or limit <= 0:
        return []

    active_weight = sum(
        d["weight"] for key, d in DIMENSIONS.items() if (profile.get(key) or "").strip()
    )
    if active_weight == 0:
        return []

    scores = [0.0] * len(strategies)
    reasons = [[] for _ in strategies]
    dimension_scores = [{} for _ in strategies]

    for key, details in DIMENSIONS.items():
        text = (profile.get(key) or "").strip()
        if not text:
            continue
        similarities, matches = compare_texts(
            text, [s[details["strategy_field"]] for s in strategies]
        )
        weight = details["weight"] / active_weight  # re-normalise over filled dimensions
        for i, similarity in enumerate(similarities):
            scores[i] += similarity * weight
            dimension_scores[i][details["label"]] = round(similarity * 100, 1)
            if matches[i]:
                reasons[i].append({"dimension": details["label"], "terms": matches[i]})

    results = []
    for i, strategy in enumerate(strategies):
        if scores[i] <= 0:
            continue
        score = round(scores[i] * 100, 1)
        label = "Strong match" if score >= 50 else "Good match" if score >= 30 else "Potential match"
        results.append({
            "name": strategy["name"],
            "niche": strategy.get("niche"),
            "score": score,
            "match_label": label,
            "matched_terms": reasons[i],
            "dimension_scores": dimension_scores[i],
        })

    results.sort(key=lambda r: (-r["score"], r["name"].lower()))
    return results[:limit]


def collaboration_text(collab: dict) -> str:
    return " ".join([
        collab.get("title", ""),
        collab.get("description", ""),
        " ".join(collab.get("required_skills", [])),
        collab.get("niche") or "",
    ])


def search_collaborations(query: str, collaborations: list[dict], limit: int = 3) -> list[dict]:
    """Rank open collaboration requests against a free-text query."""
    if not collaborations or limit <= 0:
        return []
    similarities, matches = compare_texts(
        query, [collaboration_text(c) for c in collaborations]
    )
    results = [
        {
            "id": c["id"],
            "title": c["title"],
            "author": c.get("author"),
            "niche": c.get("niche"),
            "required_skills": c.get("required_skills", []),
            "score": round(similarities[i] * 100, 1),
            "matched_terms": matches[i],
        }
        for i, c in enumerate(collaborations)
        if similarities[i] > 0
    ]
    results.sort(key=lambda r: (-r["score"], r["title"].lower()))
    return results[:limit]

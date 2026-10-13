"""Generic Question-Focused Evidence Reranker & Relevance Gate.

Evaluates candidate chunks for question relevance vs mere topical drift, enforces entity
consistency, scores multi-factor relevance, filters out TOPICALLY_RELEVANT_ONLY chunks,
and constructs the accepted Evidence Set before answer generation.
"""

from datetime import datetime, timezone
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas.research import (
    DebugTrace,
    EvidenceItem,
    QuestionType,
    QueryUnderstandingProfile,
    RelevanceClassification,
    ScoringWeights,
    SourceTier,
    TemporalScope,
)

logger = logging.getLogger("research_engine.agent.evidence_ranker")

# Generic Causal / Mechanism indicators (domain-agnostic: biology, engineering, physics, finance, social/economic)
CAUSAL_MECHANISM_MARKERS = [
    r"\b(because|causes?|causing|caused by|mechanism of|underlying principle|reasons?|root causes?)\b",
    r"\b(due to|results in|leads to|triggers?|triggering|activates?|activation|sparked|sparking)\b",
    r"\b(pathway|binds to|receptors?|inhibit|inhibiting|induces?|mediates?)\b",
    r"\b(enables?|architecture|component|operates by|works by|functioning via)\b",
    r"\b(produces?|drives?|catalyzes?|modulates?|generates?|drivers?|factors?|stemming from)\b",
    r"\b(demanding|demands|discontent|protests?|unrest|clashes|disputes?|tensions?|grievances?|outrage|conflict)\b",
]

# Generic Topical Drift markers (general statistics, other variants, superficial prevalence)
TOPICAL_DRIFT_MARKERS = [
    r"\b(prevalence|case numbers|death toll|statistics show|general population)\b",
    r"\b(unrelated|other alternatives|different product|market valuation)\b",
    r"\b(side effects reported|mild symptoms|manufacturing delay)\b",
]

# Correlation vs Causation distinction markers
CORRELATION_ONLY_MARKERS = [
    r"\b(associated with|correlated with|linked to|observational study observed)\b",
    r"\b(statistical association|co-occurs with|no causal link established)\b",
]


def chunk_source_content(
    source: Dict[str, Any],
    max_chunk_chars: int = 600,
    overlap_chars: int = 100,
) -> List[Dict[str, Any]]:
    """Splits raw source content into candidate chunks for fine-grained ranking.
    
    Guaranteed termination and lightweight processing without CPU hang.
    """
    content = source.get("content") or source.get("snippet") or ""
    url = source.get("url", "")
    title = source.get("title", url)
    tier = source.get("tier", SourceTier.TIER_2_SECONDARY.value)
    domain = source.get("domain", "")

    if not content.strip():
        return []

    # Clean redundant whitespace
    cleaned = re.sub(r"\s+", " ", content).strip()
    if len(cleaned) <= max_chunk_chars:
        return [{
            "text": cleaned,
            "source_url": url,
            "source_title": title,
            "tier": tier,
            "domain": domain,
            "chunk_index": 0,
        }]

    # Guaranteed forward progress sliding window
    step = max(max_chunk_chars - overlap_chars, 200)
    chunks = []
    idx = 0
    for start in range(0, len(cleaned), step):
        end = min(start + max_chunk_chars, len(cleaned))
        chunk_text = cleaned[start:end].strip()
        if len(chunk_text) >= 50:
            chunks.append({
                "text": chunk_text,
                "source_url": url,
                "source_title": title,
                "tier": tier,
                "domain": domain,
                "chunk_index": idx,
            })
            idx += 1
            if idx >= 15:  # Keep processing lightweight: max 15 chunks per source
                break

    return chunks


def compute_semantic_similarity(query: str, chunk_text: str) -> float:
    """Computes lexical/token-overlap semantic similarity [0.0 - 1.0]."""
    q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    c_words = set(re.findall(r"\b\w{3,}\b", chunk_text.lower()))
    if not q_words or not c_words:
        return 0.0

    intersection = q_words.intersection(c_words)
    # Jaccard-Dice hybrid
    dice = (2.0 * len(intersection)) / (len(q_words) + len(c_words))
    # Coverage of query words
    coverage = len(intersection) / len(q_words)
    return min(1.0, 0.4 * dice + 0.6 * coverage)


def compute_entity_match(
    entities: List[str],
    comparison_targets: List[str],
    chunk_text: str,
    is_comparison: bool,
) -> Tuple[float, str, Optional[str]]:
    """Evaluates entity presence and checks for entity drift.
    
    Returns:
        (entity_score, matched_entity, drift_warning)
    """
    text_lower = chunk_text.lower()
    if not entities:
        return 0.5, "general", None

    matched = []
    for ent in entities:
        ent_lower = ent.lower().strip()
        if ent_lower in text_lower:
            matched.append(ent)
            continue
        # Support multi-word entities (e.g. 'civil unrest in India') via token overlap
        words = [w for w in re.findall(r"\b\w{3,}\b", ent_lower) if w not in {"the", "and", "for", "with", "from", "right", "now"}]
        if words:
            matching_words = [w for w in words if w in text_lower]
            if len(matching_words) >= max(1, math.ceil(len(words) * 0.5)):
                matched.append(ent)

    if is_comparison and len(comparison_targets) >= 2:
        t1, t2 = comparison_targets[0].lower(), comparison_targets[1].lower()
        t1_toks = [tok for tok in re.findall(r"\b\w{3,}\b", t1) if tok not in {"and", "the", "for", "with"}]
        t2_toks = [tok for tok in re.findall(r"\b\w{3,}\b", t2) if tok not in {"and", "the", "for", "with"}]
        has_t1 = (t1 in text_lower) or (bool(t1_toks) and any(tok in text_lower for tok in t1_toks))
        has_t2 = (t2 in text_lower) or (bool(t2_toks) and any(tok in text_lower for tok in t2_toks))
        if has_t1 and has_t2:
            return 1.0, f"{comparison_targets[0]} & {comparison_targets[1]}", None
        elif has_t1:
            return 0.85, comparison_targets[0], None
        elif has_t2:
            return 0.85, comparison_targets[1], None
        else:
            return 0.1, "", "Neither comparison entity mentioned"

    # Single entity or primary entity check
    if matched:
        primary = matched[0]
        # Check if text heavily discusses an alternative competing entity (entity drift)
        return 1.0, primary, None

    # Penalty for missing entity
    return 0.1, "", f"Target entity ({', '.join(entities)}) not found in chunk"


def compute_intent_match(
    question_type: QuestionType,
    chunk_text: str,
) -> Tuple[float, str]:
    """Scores whether the chunk contains textual features matching the question intent."""
    text_lower = chunk_text.lower()

    if question_type in (QuestionType.CAUSAL_QUESTION, QuestionType.HOW_IT_WORKS):
        causal_hits = sum(1 for p in CAUSAL_MECHANISM_MARKERS if re.search(p, text_lower))
        if causal_hits >= 2:
            return 1.0, "mechanism"
        elif causal_hits == 1:
            return 0.75, "mechanism"
        # Check if only correlation
        if any(re.search(p, text_lower) for p in CORRELATION_ONLY_MARKERS):
            return 0.4, "correlational_observation"
        return 0.2, "empirical_observation"

    elif question_type == QuestionType.QUANTITATIVE:
        # Check for numbers, percentages, units
        has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?%?|\b\d+\s*(?:mg|ml|kg|wh/kg|tflops|ms|seconds|usd|gb|mb)\b", text_lower))
        return (0.9, "quantitative_metric") if has_numbers else (0.2, "qualitative_statement")

    elif question_type == QuestionType.CURRENT_LATEST:
        has_recent_year = bool(re.search(r"\b(2024|2025|2026|recently|current|now|update)\b", text_lower))
        return (0.9, "recent_status") if has_recent_year else (0.4, "status")

    elif question_type == QuestionType.DEFINITION:
        has_def = bool(re.search(r"\b(is defined as|refers to|is a|means|denotes|characterized by)\b", text_lower))
        return (0.9, "definition") if has_def else (0.3, "discussion")

    elif question_type == QuestionType.PROCEDURE_HOW_TO:
        has_steps = bool(re.search(r"\b(step|first|then|next|finally|install|configure|execute|verify)\b", text_lower))
        return (0.9, "procedural_step") if has_steps else (0.3, "guidance")

    elif question_type == QuestionType.COMPARISON:
        has_comparison = bool(re.search(r"\b(whereas|compared to|while|faster than|higher than|differs from|contrast)\b", text_lower))
        return (0.9, "comparative_evaluation") if has_comparison else (0.4, "single_entity_data")

    return 0.6, "factual"


def compute_question_relevance_and_classification(
    query: str,
    profile: QueryUnderstandingProfile,
    chunk_text: str,
    entity_score: float,
    intent_score: float,
    evidence_type: str,
) -> Tuple[float, RelevanceClassification, str]:
    """Core Question-Relevance Gate: Distinguishes direct answers from topical drift."""
    text_lower = chunk_text.lower()
    q_type = profile.question_type

    # 1. Entity check: If entity score is negligible, cannot directly answer specific question
    if entity_score < 0.3:
        return 0.1, RelevanceClassification.IRRELEVANT, "Target entity missing from evidence chunk"

    # 2. Check for explicit topical drift patterns (e.g. general stats or off-target drift)
    is_drift = any(re.search(p, text_lower) for p in TOPICAL_DRIFT_MARKERS)

    # 3. Handle Causal / How-it-works questions
    if q_type in (QuestionType.CAUSAL_QUESTION, QuestionType.HOW_IT_WORKS):
        has_mechanism = any(re.search(p, text_lower) for p in CAUSAL_MECHANISM_MARKERS)
        if has_mechanism and entity_score >= 0.7:
            return 0.95, RelevanceClassification.DIRECTLY_RELEVANT, "Explains underlying mechanism/causal pathway for target entity"
        elif has_mechanism:
            return 0.70, RelevanceClassification.PARTIALLY_RELEVANT, "Mentions mechanism but partial entity alignment"
        elif is_drift or "effectiveness" in text_lower or "%" in text_lower:
            # Contains outcome stats (e.g. 70% effectiveness) but NO mechanism!
            return 0.35, RelevanceClassification.TOPICALLY_RELEVANT_ONLY, "Topically relevant (outcome/statistics) but lacks causal mechanism"
        else:
            return 0.25, RelevanceClassification.TOPICALLY_RELEVANT_ONLY, "Mentions topic without providing causal explanation"

    # 4. Handle Comparison questions
    elif q_type == QuestionType.COMPARISON:
        targets = profile.comparison_targets
        if len(targets) >= 2:
            t1, t2 = targets[0].lower(), targets[1].lower()
            t1_toks = [tok for tok in re.findall(r"\b\w{3,}\b", t1) if tok not in {"and", "the", "for", "with"}]
            t2_toks = [tok for tok in re.findall(r"\b\w{3,}\b", t2) if tok not in {"and", "the", "for", "with"}]
            t1_found = (t1 in text_lower) or (bool(t1_toks) and any(tok in text_lower for tok in t1_toks))
            t2_found = (t2 in text_lower) or (bool(t2_toks) and any(tok in text_lower for tok in t2_toks))
            if t1_found and t2_found:
                return 0.95, RelevanceClassification.DIRECTLY_RELEVANT, f"Directly compares both {targets[0]} and {targets[1]}"
            elif t1_found or t2_found:
                found_t = targets[0] if t1_found else targets[1]
                return 0.70, RelevanceClassification.PARTIALLY_RELEVANT, f"Provides comparable metrics for {found_t}"
            else:
                return 0.2, RelevanceClassification.IRRELEVANT, "Neither comparison entity present"

    # 5. Handle Quantitative questions
    elif q_type == QuestionType.QUANTITATIVE:
        has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?%?|\b\d+\s*(?:mg|ml|kg|wh/kg|tflops|ms|seconds|usd)\b", text_lower))
        if has_numbers and entity_score >= 0.7:
            return 0.95, RelevanceClassification.DIRECTLY_RELEVANT, "Contains requested quantitative metric and measurement units"
        elif has_numbers:
            return 0.60, RelevanceClassification.PARTIALLY_RELEVANT, "Contains numbers but lacks full target entity context"
        else:
            return 0.30, RelevanceClassification.TOPICALLY_RELEVANT_ONLY, "Topically related discussion without requested numeric metrics"

    # 6. Handle Current/Latest questions
    elif q_type == QuestionType.CURRENT_LATEST:
        has_date = bool(re.search(r"\b(2024|2025|2026)\b", text_lower))
        if has_date and entity_score >= 0.7:
            return 0.95, RelevanceClassification.DIRECTLY_RELEVANT, "Provides verified recent status with timestamp anchor"
        elif entity_score >= 0.7:
            return 0.60, RelevanceClassification.PARTIALLY_RELEVANT, "Discusses current state but lacks explicit 2024-2026 date anchor"
        else:
            return 0.25, RelevanceClassification.TOPICALLY_RELEVANT_ONLY, "Mentions topic without contemporary update"

    # 7. General question handling
    semantic_sim = compute_semantic_similarity(query, chunk_text)
    if semantic_sim >= 0.55 and entity_score >= 0.7:
        return 0.85, RelevanceClassification.DIRECTLY_RELEVANT, "Direct high-similarity evidence matching target question"
    elif semantic_sim >= 0.35 and entity_score >= 0.5:
        return 0.60, RelevanceClassification.PARTIALLY_RELEVANT, "Partial evidence relevant to question context"
    elif entity_score >= 0.5:
        return 0.35, RelevanceClassification.TOPICALLY_RELEVANT_ONLY, "Topically related to entity but does not directly answer question"
    else:
        return 0.15, RelevanceClassification.IRRELEVANT, "Unrelated content"


def compute_recency_score(temporal_scope: TemporalScope, text: str) -> float:
    """Computes recency score based on query temporal requirements."""
    if temporal_scope == TemporalScope.STATIC:
        return 0.5  # Neutral for timeless principles

    if temporal_scope == TemporalScope.HISTORICAL:
        has_hist = bool(re.search(r"\b(19\d{2}|20[0-1]\d|2020|2021)\b", text))
        return 0.9 if has_hist else 0.4

    # RECENT (2024-2026)
    if re.search(r"\b202[5-6]\b", text):
        return 1.0
    elif "2024" in text:
        return 0.8
    elif any(k in text.lower() for k in ["recent", "latest", "update", "current"]):
        return 0.7
    return 0.3


def rank_and_build_evidence_set(
    sources: List[Dict[str, Any]],
    profile: QueryUnderstandingProfile,
    weights: Optional[ScoringWeights] = None,
    max_evidence_items: int = 12,
) -> Tuple[List[EvidenceItem], DebugTrace, bool]:
    """Executes multi-factor reranking, applies Direct-Answer Gate, and produces the Evidence Set.
    
    Returns:
        (evidence_set, debug_trace, is_evidence_insufficient)
    """
    if weights is None:
        weights = ScoringWeights()

    w1 = weights.w_semantic
    w2 = weights.w_entity
    w3 = weights.w_intent
    w4 = weights.w_question
    w5 = weights.w_source
    w6 = weights.w_recency

    is_comparison = profile.question_type == QuestionType.COMPARISON
    entities = profile.entities
    query = profile.main_topic

    all_chunks: List[Dict[str, Any]] = []
    for s in sources:
        chunks = chunk_source_content(s)
        all_chunks.extend(chunks)

    evaluated_candidates = []
    rejected_chunks = []
    accepted_items: List[EvidenceItem] = []

    # Map tier string to tier score
    tier_scores = {
        SourceTier.TIER_1_PRIMARY.value: 1.0,
        SourceTier.TIER_2_SECONDARY.value: 0.7,
        SourceTier.TIER_3_TERTIARY.value: 0.4,
    }

    item_counter = 1

    for c in all_chunks:
        chunk_text = c["text"]
        tier_val = c["tier"]

        # 1. Semantic Similarity
        sem_sim = compute_semantic_similarity(query, chunk_text)

        # 2. Entity Match
        ent_score, matched_entity, drift_warn = compute_entity_match(
            entities, profile.comparison_targets, chunk_text, is_comparison
        )

        # 3. Intent Match
        intent_score, evidence_type = compute_intent_match(profile.question_type, chunk_text)

        # 4. Question Relevance & Classification
        q_rel_score, classification, rationale = compute_question_relevance_and_classification(
            query, profile, chunk_text, ent_score, intent_score, evidence_type
        )

        # 5. Source Quality Score
        source_score = tier_scores.get(tier_val, 0.5)

        # 6. Recency Score
        recency_score = compute_recency_score(profile.temporal_scope, chunk_text)

        # Final Combined Score
        final_score = (
            w1 * sem_sim +
            w2 * ent_score +
            w3 * intent_score +
            w4 * q_rel_score +
            w5 * source_score +
            w6 * recency_score
        )

        candidate_record = {
            "source_url": c["source_url"],
            "source_title": c["source_title"],
            "tier": tier_val,
            "text_preview": chunk_text[:120] + "...",
            "sem_sim": round(sem_sim, 3),
            "ent_score": round(ent_score, 3),
            "intent_score": round(intent_score, 3),
            "q_rel_score": round(q_rel_score, 3),
            "source_score": round(source_score, 3),
            "recency_score": round(recency_score, 3),
            "final_score": round(final_score, 3),
            "classification": classification.value,
            "rationale": rationale,
            "drift_warning": drift_warn,
        }
        evaluated_candidates.append(candidate_record)

        # GATE: Only accept DIRECTLY_RELEVANT and PARTIALLY_RELEVANT
        if classification in (RelevanceClassification.DIRECTLY_RELEVANT, RelevanceClassification.PARTIALLY_RELEVANT):
            # Also require reasonable overall threshold
            if final_score >= 0.35:
                ev_id = f"EVD-{item_counter:02d}"
                item_counter += 1
                try:
                    tier_enum = SourceTier(tier_val)
                except Exception:
                    tier_enum = SourceTier.TIER_2_SECONDARY

                accepted_items.append(
                    EvidenceItem(
                        evidence_id=ev_id,
                        text=chunk_text,
                        source_url=c["source_url"],
                        source_title=c["source_title"],
                        entity=matched_entity or (entities[0] if entities else "General"),
                        relevance_classification=classification,
                        relevance_score=round(final_score, 4),
                        source_tier=tier_enum,
                        evidence_type=evidence_type,
                        publication_date=c.get("publication_date"),
                    )
                )
            else:
                rejected_chunks.append({
                    **candidate_record,
                    "rejection_reason": f"Final score ({final_score:.2f}) below minimum acceptance threshold (0.35)",
                })
        else:
            # Filtered out due to topical drift or irrelevance
            rejection_reason = f"Filtered out: Classified as {classification.value} ({rationale})"
            if drift_warn:
                rejection_reason += f" | {drift_warn}"
            rejected_chunks.append({
                **candidate_record,
                "rejection_reason": rejection_reason,
            })

    # Sort accepted evidence items by relevance score descending
    accepted_items.sort(key=lambda x: x.relevance_score, reverse=True)

    # For comparison questions: Ensure balanced evidence representation across entities
    if is_comparison and len(profile.comparison_targets) >= 2:
        t1, t2 = profile.comparison_targets[0].lower(), profile.comparison_targets[1].lower()
        items_t1 = [it for it in accepted_items if t1 in it.text.lower() or t1 in it.entity.lower()]
        items_t2 = [it for it in accepted_items if t2 in it.text.lower() or t2 in it.entity.lower()]
        # Interleave to preserve balance
        interleaved: List[EvidenceItem] = []
        max_len = max(len(items_t1), len(items_t2))
        seen_ids = set()
        for i in range(max_len):
            if i < len(items_t1) and items_t1[i].evidence_id not in seen_ids:
                interleaved.append(items_t1[i])
                seen_ids.add(items_t1[i].evidence_id)
            if i < len(items_t2) and items_t2[i].evidence_id not in seen_ids:
                interleaved.append(items_t2[i])
                seen_ids.add(items_t2[i].evidence_id)
        # Add any joint items
        for it in accepted_items:
            if it.evidence_id not in seen_ids:
                interleaved.append(it)
                seen_ids.add(it.evidence_id)
    # If accepted items are completely empty, promote the top scoring candidate chunks from diverse sources
    if len(accepted_items) == 0 and evaluated_candidates:
        seen_urls = {it.source_url for it in accepted_items}
        # Find raw chunks matching candidate URLs
        url_to_chunk = {c["source_url"]: c for c in all_chunks}
        for cand in sorted(evaluated_candidates, key=lambda c: c["final_score"], reverse=True):
            url = cand["source_url"]
            if (
                url not in seen_urls
                and cand["final_score"] >= 0.25
                and cand.get("ent_score", 0) >= 0.4
                and cand.get("classification") != RelevanceClassification.IRRELEVANT.value
            ):
                raw_c = url_to_chunk.get(url, {})
                chunk_full_text = raw_c.get("text") or cand.get("text_preview", "")
                ev_id = f"EVD-{item_counter:02d}"
                item_counter += 1
                try:
                    tier_enum = SourceTier(cand["tier"])
                except Exception:
                    tier_enum = SourceTier.TIER_2_SECONDARY
                accepted_items.append(
                    EvidenceItem(
                        evidence_id=ev_id,
                        text=chunk_full_text,
                        source_url=url,
                        source_title=cand["source_title"],
                        entity=entities[0] if entities else "General",
                        relevance_classification=RelevanceClassification.PARTIALLY_RELEVANT,
                        relevance_score=round(cand["final_score"], 4),
                        source_tier=tier_enum,
                        evidence_type="empirical_observation",
                    )
                )
                seen_urls.add(url)
                if len(accepted_items) >= 6:
                    break

    final_evidence_set = accepted_items[:max_evidence_items]

    # Insufficient Evidence Detection
    # If no DIRECTLY_RELEVANT chunks exist, or accepted evidence is strictly empty / sparse
    directly_relevant_count = sum(
        1 for it in final_evidence_set if it.relevance_classification == RelevanceClassification.DIRECTLY_RELEVANT
    )
    insufficient_evidence = (len(final_evidence_set) == 0) or (directly_relevant_count == 0 and profile.question_type in (
        QuestionType.CAUSAL_QUESTION, QuestionType.HOW_IT_WORKS, QuestionType.QUANTITATIVE, QuestionType.CURRENT_LATEST
    ))

    debug_trace = DebugTrace(
        parsed_query={
            "query": query,
            "entities": profile.entities,
            "question_type": profile.question_type.value,
            "intent": profile.intent.value,
            "temporal_scope": profile.temporal_scope.value,
            "required_evidence": profile.required_evidence,
            "exclude_topics": profile.exclude_topics,
        },
        candidate_chunks_evaluated=len(all_chunks),
        accepted_evidence_count=len(final_evidence_set),
        rejected_chunks_count=len(rejected_chunks),
        rejected_chunks=rejected_chunks[:25],  # Keep representative sample for debug inspection
        scoring_weights={
            "w_semantic": w1,
            "w_entity": w2,
            "w_intent": w3,
            "w_question": w4,
            "w_source": w5,
            "w_recency": w6,
        },
        insufficient_evidence_flag=insufficient_evidence,
    )

    return final_evidence_set, debug_trace, insufficient_evidence

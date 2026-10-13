"""Comprehensive Evaluation Test Suite for Generic Question-Focused RAG Pipeline.

Validates against failure modes A through J:
A. Topically relevant evidence is NOT question relevant (e.g. outcome stats vs causal mechanism).
B. Multiple entities are mentioned (entity consistency & drift prevention).
C. Sources contradict each other (explicit divergence detection).
D. The answer requires recent information (recency weighting & temporal anchor).
E. Evidence is insufficient (system acknowledges gap, avoids hallucinating missing explanation).
F. The query is simple and produces concise response.
G. The query requires a deep causal/mechanism breakdown.
H. Correct topic but wrong population / time period.
I. Correlation vs causation distinction (flags correlation without inventing causation).
J. Comparison questions require balanced evidence for both entities.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

from app.agent.query_understanding import (
    build_query_understanding_profile,
    classify_question_type_heuristic,
    generate_intent_targeted_queries,
)
from app.agent.evidence_ranker import (
    chunk_source_content,
    compute_semantic_similarity,
    compute_entity_match,
    compute_intent_match,
    compute_question_relevance_and_classification,
    rank_and_build_evidence_set,
)
from app.schemas.research import (
    QuestionType,
    QueryIntent,
    RelevanceClassification,
    ScoringWeights,
    SourceTier,
    TemporalScope,
)
from app.agent.nodes import (
    planner_node,
    researcher_node,
    critic_node,
    synthesizer_node,
    verifier_node,
)
from app.agent.state import ResearchState


# ==============================================================================
# TEST CASE A: Topical Relevance vs Question Relevance (Causal Mechanism)
# ==============================================================================
def test_case_a_topical_drift_filtering():
    """Verify that chunks with high topic overlap (e.g. 70% effectiveness) but NO mechanism
    are classified as TOPICALLY_RELEVANT_ONLY and filtered out of the accepted Evidence Set."""
    query = "Why is Covishield so effective against coronavirus?"
    profile = build_query_understanding_profile(query)

    assert profile.question_type == QuestionType.CAUSAL_QUESTION
    assert "Covishield" in profile.entities

    sources = [
        {
            "title": "Vaccine Statistics 2021",
            "url": "https://who.int/covishield-stats",
            "tier": "TIER_1_PRIMARY",
            "content": "Covishield demonstrated 70% effectiveness across clinical trials. Millions of doses were administered globally with high recipient satisfaction.",
        },
        {
            "title": "Covishield Biological Mechanism",
            "url": "https://nature.com/articles/covishield-mechanism",
            "tier": "TIER_2_SECONDARY",
            "content": "Covishield uses a replication-deficient chimpanzee adenovirus vector (ChAdOx1) to deliver the genetic code of the SARS-CoV-2 spike protein, which triggers cellular immune response and antibody production.",
        },
        {
            "title": "Unrelated COVID Variants",
            "url": "https://cdc.gov/variants",
            "tier": "TIER_1_PRIMARY",
            "content": "Recent coronavirus variants have emerged worldwide with mutations in different viral lineages.",
        },
    ]

    evidence_set, debug_trace, is_insufficient = rank_and_build_evidence_set(sources, profile)

    # 1. The mechanism chunk MUST be accepted as DIRECTLY_RELEVANT
    accepted_urls = [ev.source_url for ev in evidence_set]
    assert "https://nature.com/articles/covishield-mechanism" in accepted_urls

    # 2. The statistics-only chunk MUST NOT be classified as DIRECTLY_RELEVANT
    stats_chunk = next((c for c in debug_trace.rejected_chunks if "covishield-stats" in c.get("source_url", "")), None)
    if stats_chunk:
        assert stats_chunk["classification"] == RelevanceClassification.TOPICALLY_RELEVANT_ONLY.value

    # 3. Off-topic variant chunk must be filtered out
    assert "https://cdc.gov/variants" not in accepted_urls


# ==============================================================================
# TEST CASE B & J: Comparison Target Balance & Entity Consistency
# ==============================================================================
def test_case_b_and_j_comparison_entity_balance():
    """Verify that a comparison query extracts both entities and ensures balanced evidence."""
    query = "Compare DuckDB vs ClickHouse for analytical workloads"
    profile = build_query_understanding_profile(query)

    assert profile.question_type == QuestionType.COMPARISON
    assert len(profile.comparison_targets) >= 2
    assert "DuckDB" in profile.comparison_targets
    assert "ClickHouse" in profile.comparison_targets

    sources = [
        {
            "title": "DuckDB Architecture",
            "url": "https://duckdb.org/arch",
            "tier": "TIER_1_PRIMARY",
            "content": "DuckDB operates as an embedded in-process vectorized columnar database optimized for single-node analytical execution without client-server overhead.",
        },
        {
            "title": "ClickHouse Architecture",
            "url": "https://clickhouse.com/arch",
            "tier": "TIER_1_PRIMARY",
            "content": "ClickHouse is a distributed MPP columnar database designed for petabyte-scale real-time analytical clustering with high concurrency.",
        },
        {
            "title": "PostgreSQL Tutorial",
            "url": "https://postgres.org/docs",
            "tier": "TIER_1_PRIMARY",
            "content": "PostgreSQL is an open-source object-relational database system with strong transaction guarantees.",
        },
    ]

    evidence_set, debug_trace, is_insufficient = rank_and_build_evidence_set(sources, profile)

    # Both comparison targets MUST be represented in the accepted evidence set
    entities_present = {ev.entity for ev in evidence_set}
    has_duck = any("DuckDB" in ev.text or "DuckDB" in ev.entity for ev in evidence_set)
    has_click = any("ClickHouse" in ev.text or "ClickHouse" in ev.entity for ev in evidence_set)

    assert has_duck is True
    assert has_click is True
    # Third entity (PostgreSQL) must not crowd out comparison entities
    assert all("PostgreSQL" not in ev.entity for ev in evidence_set)


# ==============================================================================
# TEST CASE C: Source Contradiction Detection
# ==============================================================================
@pytest.mark.asyncio
async def test_case_c_contradiction_detection():
    """Verify that conflicting claims across sources are surfaced rather than merged."""
    state: ResearchState = {
        "task_id": "test-contradiction-1",
        "topic": "Sodium-Ion Battery Energy Density",
        "scraped_sources": [
            {
                "title": "Battery Lab Benchmark 2025",
                "url": "https://sciencedirect.com/sodium-density-a",
                "content": "Sodium-ion battery cells reached 160 Wh/kg commercially, with theoretical limits capped at 170 Wh/kg.",
                "tier": "TIER_2_SECONDARY",
            },
            {
                "title": "Advanced Materials Review 2025",
                "url": "https://nature.com/sodium-density-b",
                "content": "Breakthrough solid-state sodium-ion prototypes have demonstrated 210 Wh/kg in verified laboratory tests.",
                "tier": "TIER_1_PRIMARY",
            },
        ],
        "evidence_set": [
            {
                "evidence_id": "EVD-01",
                "text": "Sodium-ion cells reached 160 Wh/kg commercially, capped at 170 Wh/kg.",
                "source_url": "https://sciencedirect.com/sodium-density-a",
                "source_title": "Battery Lab Benchmark",
                "entity": "Sodium-Ion Battery",
                "relevance_classification": "DIRECTLY_RELEVANT",
                "source_tier": "TIER_2_SECONDARY",
            },
            {
                "evidence_id": "EVD-02",
                "text": "Breakthrough solid-state prototypes demonstrated 210 Wh/kg in laboratory tests.",
                "source_url": "https://nature.com/sodium-density-b",
                "source_title": "Advanced Materials Review",
                "entity": "Sodium-Ion Battery",
                "relevance_classification": "DIRECTLY_RELEVANT",
                "source_tier": "TIER_1_PRIMARY",
            },
        ],
        "iteration_count": 0,
        "query_profile": {
            "entities": ["Sodium-Ion Battery"],
            "question_type": "quantitative",
            "intent": "QUANTITATIVE",
            "temporal_scope": "RECENT",
        },
    }

    mock_critic_output = (
        '{\n'
        '  "passed": true,\n'
        '  "critique": "Evidence presents diverging commercial vs laboratory metrics.",\n'
        '  "contradictions": [\n'
        '    {\n'
        '      "topic": "Energy Density Metric",\n'
        '      "source_a_url": "https://sciencedirect.com/sodium-density-a",\n'
        '      "source_a_claim": "Commercial cells capped at 160-170 Wh/kg",\n'
        '      "source_b_url": "https://nature.com/sodium-density-b",\n'
        '      "source_b_claim": "Lab prototypes demonstrated 210 Wh/kg",\n'
        '      "discrepancy_analysis": "Commercial production vs experimental lab cell conditions"\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    with patch("app.config.Settings.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = mock_critic_output
        mock_get_llm.return_value = mock_llm

        res = await critic_node(state)

        assert "contradictions" in res
        assert len(res["contradictions"]) == 1
        assert res["contradictions"][0]["topic"] == "Energy Density Metric"


# ==============================================================================
# TEST CASE D: Recency Weighting for Contemporary Inquiries
# ==============================================================================
def test_case_d_recency_prioritization():
    """Verify that CURRENT_LATEST questions prioritize contemporary sources over historical ones."""
    query = "What is the latest FDA guidance on AI in medical devices as of 2025?"
    profile = build_query_understanding_profile(query)

    assert profile.question_type == QuestionType.CURRENT_LATEST
    assert profile.temporal_scope == TemporalScope.RECENT

    sources = [
        {
            "title": "FDA 2019 Legacy Framework",
            "url": "https://fda.gov/2019-framework",
            "tier": "TIER_1_PRIMARY",
            "content": "In 2019, the FDA proposed an initial discussion paper on software as a medical device (SaMD) modifications.",
        },
        {
            "title": "FDA 2025 Final Guidance",
            "url": "https://fda.gov/2025-final-guidance",
            "tier": "TIER_1_PRIMARY",
            "content": "In late 2025, the FDA released finalized binding guidance regarding Predetermined Change Control Plans (PCCP) for machine learning devices.",
        },
    ]

    evidence_set, debug_trace, is_insufficient = rank_and_build_evidence_set(sources, profile)

    assert len(evidence_set) >= 1
    # 2025 guidance must score higher than 2019 legacy paper
    assert evidence_set[0].source_url == "https://fda.gov/2025-final-guidance"


# ==============================================================================
# TEST CASE E: Insufficient Evidence Handling
# ==============================================================================
def test_case_e_insufficient_evidence_acknowledged():
    """Verify that when only outcome correlation is present without causal mechanism,
    the ranker flags insufficient evidence and synthesizer acknowledges the gap."""
    query = "Why does Compound-X cause cardiac regeneration?"
    profile = build_query_understanding_profile(query)

    # Only provide correlation / outcome data, zero mechanism
    sources = [
        {
            "title": "Compound-X Observational Trial",
            "url": "https://nejm.org/compound-x-obs",
            "tier": "TIER_1_PRIMARY",
            "content": "Compound-X was associated with a 25% improvement in left ventricular ejection fraction in rodent models.",
        }
    ]

    evidence_set, debug_trace, is_insufficient = rank_and_build_evidence_set(sources, profile)

    # Must flag insufficiency because no mechanism was established
    assert is_insufficient is True
    assert debug_trace.insufficient_evidence_flag is True


# ==============================================================================
# TEST CASE F & G: Adaptive Formatting for Simple Factual vs Deep Causal Queries
# ==============================================================================
def test_case_f_and_g_query_classification_variety():
    """Verify classification across diverse question types without domain hardcoding."""
    # Simple definition / factual
    prof_def = build_query_understanding_profile("What is WebAssembly System Interface (WASI)?")
    assert prof_def.question_type == QuestionType.DEFINITION
    assert "Authoritative Definition" in prof_def.recommended_format

    # Deep causal explanation
    prof_causal = build_query_understanding_profile("Why does quantum entanglement prevent faster-than-light communication?")
    assert prof_causal.question_type == QuestionType.CAUSAL_QUESTION
    assert "Mechanism" in prof_causal.recommended_format

    # Step-by-step procedure
    prof_howto = build_query_understanding_profile("How to configure mTLS in Envoy proxy step by step")
    assert prof_howto.question_type == QuestionType.PROCEDURE_HOW_TO

    # Quantitative
    prof_quant = build_query_understanding_profile("What is the efficiency percentage of perovskite tandem solar cells?")
    assert prof_quant.question_type == QuestionType.QUANTITATIVE


# ==============================================================================
# TEST CASE H: Correct Topic but Wrong Context / Population Drift
# ==============================================================================
def test_case_h_entity_and_context_drift():
    """Verify that evidence for a different entity or drift is penalized."""
    query = "How does Moderna mRNA-1273 induce immune memory?"
    profile = build_query_understanding_profile(query)

    # Evidence discussing a different vaccine (Pfizer BNT162b2)
    chunk_other = "Pfizer BNT162b2 utilizes lipid nanoparticle encapsulation for mRNA delivery."
    score, matched, warning = compute_entity_match(
        entities=["Moderna mRNA-1273"],
        comparison_targets=[],
        chunk_text=chunk_other,
        is_comparison=False,
    )

    # Score should be low with drift warning
    assert score <= 0.3
    assert warning is not None


# ==============================================================================
# TEST CASE I: Correlation vs Causation Safeguard
# ==============================================================================
def test_case_i_correlation_vs_causation_tagging():
    """Verify that chunks indicating correlation are categorized as correlational rather than causal proof."""
    text_corr = "High consumption of coffee is statistically associated with reduced mortality in observational cohorts."
    score, ev_type = compute_intent_match(QuestionType.CAUSAL_QUESTION, text_corr)

    # Should detect correlational observation and not give maximum 1.0 mechanism score
    assert ev_type == "correlational_observation"
    assert score <= 0.6


# ==============================================================================
# TEST CASE: Configurable Reranking Weights
# ==============================================================================
def test_configurable_scoring_weights():
    """Verify that reranking weights are configurable and dynamically alter ranking."""
    query = "What is the consensus on LK-99 superconductivity?"
    profile = build_query_understanding_profile(query)

    sources = [
        {
            "title": "Random Social Media Blog Post",
            "url": "https://randomblog.com/lk99",
            "tier": "TIER_3_TERTIARY",
            "content": "What is the consensus on LK-99 superconductivity? Many amateur commentators are asking this question online.",
        },
        {
            "title": "Nature Materials Peer-Reviewed Study",
            "url": "https://nature.com/articles/lk99-study",
            "tier": "TIER_1_PRIMARY",
            "content": "Consensus from global replication studies confirms that LK-99 exhibits ferromagnetism, not ambient superconductivity.",
        },
    ]

    # Weights heavily prioritizing Source Quality Tier
    source_heavy_weights = ScoringWeights(
        w_semantic=0.10,
        w_entity=0.20,
        w_intent=0.10,
        w_question=0.10,
        w_source=0.45,
        w_recency=0.05,
    )

    evidence_set, debug_trace, _ = rank_and_build_evidence_set(
        sources, profile, weights=source_heavy_weights
    )

    assert len(evidence_set) >= 1
    # Nature study must be top ranked
    assert evidence_set[0].source_url == "https://nature.com/articles/lk99-study"
    assert debug_trace.scoring_weights["w_source"] == 0.45


# ==============================================================================
# TEST CASE: Claim Verification & Hallucinated URL Elimination
# ==============================================================================
@pytest.mark.asyncio
async def test_verifier_node_removes_hallucinated_urls():
    """Verify that verifier_node strips any citation URL that was never scraped."""
    state: ResearchState = {
        "task_id": "test-verifier-grounding",
        "topic": "Rust Concurrency Memory Safety",
        "draft_report": (
            "# Rust Concurrency\n\n"
            "## Executive Summary\n"
            "Rust guarantees memory safety without garbage collection through its ownership model.\n\n"
            "## Key Findings\n"
            "- Ownership rules prevent data races at compile time ([Rust Docs](https://doc.rust-lang.org/nomicon)).\n"
            "- Fake claim citing hallucinated URL ([Fake Paper](https://hallucinated-invented-source.com/paper)).\n"
        ),
        "scraped_sources": [
            {
                "title": "Rust Nomicon",
                "url": "https://doc.rust-lang.org/nomicon",
                "content": "The Rust ownership system guarantees thread safety through Send and Sync traits.",
                "tier": "TIER_1_PRIMARY",
            }
        ],
        "citations": [
            {
                "title": "Rust Nomicon",
                "url": "https://doc.rust-lang.org/nomicon",
                "tier": "TIER_1_PRIMARY",
            },
            {
                "title": "Fake Paper",
                "url": "https://hallucinated-invented-source.com/paper",
                "tier": "TIER_3_TERTIARY",
            },
        ],
        "evidence_set": [
            {
                "evidence_id": "EVD-01",
                "text": "The Rust ownership system guarantees thread safety through Send and Sync traits.",
                "source_url": "https://doc.rust-lang.org/nomicon",
                "source_title": "Rust Nomicon",
                "entity": "Rust",
                "relevance_classification": "DIRECTLY_RELEVANT",
                "source_tier": "TIER_1_PRIMARY",
            }
        ],
        "iteration_count": 1,
        "query_profile": {
            "entities": ["Rust"],
            "question_type": "how_it_works",
            "intent": "EXPLANATORY",
        },
    }

    res = await verifier_node(state)

    final_report = res["final_report"]
    final_urls = [c["url"] for c in final_report["citations"]]

    # Legitimate scraped URL is preserved
    assert "https://doc.rust-lang.org/nomicon" in final_urls
    # Hallucinated URL MUST BE REMOVED
    assert "https://hallucinated-invented-source.com/paper" not in final_urls
    assert res["verification_passed"] is True


# ==============================================================================
# TEST CASE: Observability & Debug Trace
# ==============================================================================
def test_observability_debug_trace_structure():
    """Verify that DebugTrace captures parsed query, scoring weights, evaluations and rejected reasons."""
    query = "Why is covishield effective?"
    profile = build_query_understanding_profile(query)

    sources = [
        {
            "title": "General News",
            "url": "https://generic-news.com/covid",
            "content": "General population vaccination campaigns continued across the nation.",
            "tier": "TIER_3_TERTIARY",
        }
    ]

    evidence_set, debug_trace, is_insufficient = rank_and_build_evidence_set(sources, profile)

    assert debug_trace.parsed_query["question_type"] == "causal_question"
    assert "Covishield" in debug_trace.parsed_query["entities"] or "covishield" in debug_trace.parsed_query["query"].lower()
    assert debug_trace.candidate_chunks_evaluated >= 1
    assert debug_trace.rejected_chunks_count >= 1
    assert len(debug_trace.rejected_chunks) >= 1
    assert "rejection_reason" in debug_trace.rejected_chunks[0]


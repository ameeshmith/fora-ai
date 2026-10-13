"""Generic Question-Focused Query Understanding & Intent Classification Layer.

Provides structured intent decomposition, entity extraction, required evidence profiling,
and topical exclusion rules across all domains (medical, scientific, legal, financial, technical).
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.research import (
    QuestionType,
    QueryIntent,
    QueryUnderstandingProfile,
    TemporalScope,
)

logger = logging.getLogger("research_engine.agent.query_understanding")

# Domain indicator dictionaries (purely heuristic, non-restrictive, domain-agnostic fallback)
DOMAIN_PATTERNS = {
    "medical_life_sciences": [
        r"\b(vaccin|pathway|protein|clinical|dosage|disease|drug|virus|cell|syndrome|therapy|fda|mrna|infection|antibody)\b"
    ],
    "scientific_physical": [
        r"\b(quantum|semiconductor|superconduct|physics|entropy|qubit|isotope|spectroscopy|particle|gravity|thermal)\b"
    ],
    "technical_software": [
        r"\b(api|architecture|framework|database|latency|compiler|algorithm|concurrency|kubernetes|microservice|python|rust)\b"
    ],
    "financial_economic": [
        r"\b(inflation|gdp|liquidity|valuation|equity|ebitda|bond|yield|hedge|portfolio|interest rate|market cap)\b"
    ],
    "legal_regulatory": [
        r"\b(compliance|statute|regulation|gdpr|jurisdiction|liability|copyright|patent|treaty|enforcement|directive)\b"
    ],
}


def classify_question_type_heuristic(query: str) -> Tuple[QuestionType, QueryIntent]:
    """Classifies a research question into one of 16 generic question types and query intent."""
    q_lower = query.lower().strip()

    # Comparison questions
    if any(k in q_lower for k in ["compare", " vs ", " versus ", "difference between", "better than", "trade-offs between", "benchmark against"]):
        return QuestionType.COMPARISON, QueryIntent.COMPARISON

    # Causal / Why questions (Mechanisms, reasons, root causes)
    if re.search(r"\bwhy\b", q_lower) or any(k in q_lower for k in [
        "what causes", "what is the reason", "how come", "why does", "why is", "why do",
        "root cause of", "mechanism of why"
    ]):
        return QuestionType.CAUSAL_QUESTION, QueryIntent.CAUSAL_ANALYSIS

    # How it works / Mechanisms
    if any(k in q_lower for k in ["how does it work", "how does", "how do", "mechanism of action", "inner workings", "underlying process", "architecture of how"]):
        return QuestionType.HOW_IT_WORKS, QueryIntent.EXPLANATORY

    # Quantitative / Metrics / Numerical questions
    if any(k in q_lower for k in ["how much", "how many", "what percentage", "what is the rate", "what is the cost", "cost of", "efficiency percentage", "benchmark score", "statistics of", "metric for"]):
        return QuestionType.QUANTITATIVE, QueryIntent.QUANTITATIVE

    # Current / Latest / Status updates
    if any(k in q_lower for k in ["latest", "recent", "current status", "newest", "as of 2024", "as of 2025", "as of 2026", "what is happening with", "state of the art"]):
        return QuestionType.CURRENT_LATEST, QueryIntent.RECENT_DEVELOPMENTS

    # Historical / Epoch questions
    if any(k in q_lower for k in ["history of", "in 19", "in 20", "originated", "historical background", "timeline of", "how did it start"]):
        # Check if year is in the past
        year_match = re.search(r"\b(19\d{2}|20[0-1]\d|202[0-3])\b", q_lower)
        if year_match or "history" in q_lower:
            return QuestionType.HISTORICAL, QueryIntent.EXPLANATORY

    # Definition / Conceptual
    if any(k in q_lower for k in ["what is ", "what are ", "define ", "definition of", "meaning of", "what does "]) and not any(k in q_lower for k in ["how", "why"]):
        return QuestionType.DEFINITION, QueryIntent.DEFINITION

    # Procedure / How to
    if any(k in q_lower for k in ["how to ", "how do i ", "step by step", "procedure for", "tutorial", "guide to", "implementation steps"]):
        return QuestionType.PROCEDURE_HOW_TO, QueryIntent.HOW_TO

    # Regulatory / Legal
    if any(k in q_lower for k in ["regulation", "legal", "fda approval", "compliance", "mandate", "statutory", "policy for", "gdpr", "guidelines"]):
        return QuestionType.REGULATORY_LEGAL, QueryIntent.REQUIREMENTS

    # Troubleshooting / Diagnostics
    if any(k in q_lower for k in ["troubleshoot", "debug", "issue with", "why is it failing", "error", "fix", "mitigate problem"]):
        return QuestionType.TROUBLESHOOTING, QueryIntent.HOW_TO

    # Recommendation
    if any(k in q_lower for k in ["should i use", "recommend", "best option for", "which one to choose", "recommendation for"]):
        return QuestionType.RECOMMENDATION, QueryIntent.COMPARISON

    # Evidence / Research / Literature
    if any(k in q_lower for k in ["evidence for", "studies on", "research papers on", "clinical trials for", "literature on"]):
        return QuestionType.EVIDENCE_RESEARCH, QueryIntent.LITERATURE_RETRIEVAL

    # Summarization
    if any(k in q_lower for k in ["summarize", "summary of", "brief overview of", "tldr"]):
        return QuestionType.SUMMARIZATION, QueryIntent.EXPLANATORY

    # Explanation / General
    if any(k in q_lower for k in ["explain", "overview", "describe", "elucidate"]):
        return QuestionType.EXPLANATION, QueryIntent.EXPLANATORY

    # Default fallback: Factual question
    return QuestionType.FACTUAL_QUESTION, QueryIntent.DIRECT_FACT


def extract_entities_and_context(query: str) -> Dict[str, Any]:
    """Extracts entities, topics, exclusion zones, and constraints generically."""
    q_clean = query.strip()
    words = q_clean.split()

    # Comparison target extraction
    comparison_targets: List[str] = []
    vs_match = re.search(r"(.+?)\s+(?:vs\.?|versus|compared to)\s+(.+?)(?:\?|$)", q_clean, re.IGNORECASE)
    if vs_match:
        target1 = re.sub(r"^(?:compare|what is the difference between)\s+", "", vs_match.group(1), flags=re.IGNORECASE).strip()
        target2 = vs_match.group(2).strip()
        # Clean trailing contextual prepositions (e.g. 'ClickHouse for analytical workloads' -> 'ClickHouse')
        target1 = re.sub(r"\s+(?:for|in|on|with|under|regarding)\s+.*$", "", target1, flags=re.IGNORECASE).strip()
        target2 = re.sub(r"\s+(?:for|in|on|with|under|regarding)\s+.*$", "", target2, flags=re.IGNORECASE).strip()
        comparison_targets = [target1, target2]

    # Entity candidate extraction (capitalized words, quoted strings, or distinctive phrases)
    quotes = re.findall(r'"([^"]+)"', q_clean)
    capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9\-_]{2,}\b", q_clean)

    # Stopwords to filter from capitalized entities
    stopwords = {"What", "Why", "How", "When", "Where", "Who", "Explain", "Compare", "Which", "Should", "Could", "Does", "Is", "Are"}
    cleaned_entities = [e for e in capitalized if e not in stopwords]
    entities = list(dict.fromkeys(quotes + comparison_targets + cleaned_entities))

    if not entities and len(words) > 0:
        # Fallback: strip leading question words and take core subject noun phrase
        core_phrase = re.sub(r"^(?:why is|why does|how does|what is|how to|explain|compare|what are)\s+", "", q_clean, flags=re.IGNORECASE)
        core_phrase = re.sub(r"\s+(?:effective|work|so popular|different)\??$", "", core_phrase, flags=re.IGNORECASE).strip(" ?.")
        if core_phrase:
            entities = [core_phrase]

    # Domain detection
    detected_domain = "general"
    for dom_name, patterns in DOMAIN_PATTERNS.items():
        if any(re.search(pat, q_clean, re.IGNORECASE) for pat in patterns):
            detected_domain = dom_name
            break

    # Time requirement extraction
    time_req = None
    time_match = re.search(r"\b(20[1-2]\d|19\d{2}|latest|recent|current|historical|in \d{4})\b", q_clean, re.IGNORECASE)
    if time_match:
        time_req = time_match.group(0)

    # Temporal scope
    if any(k in q_clean.lower() for k in ["latest", "recent", "current", "2024", "2025", "2026", "now"]):
        temporal_scope = TemporalScope.RECENT
    elif any(k in q_clean.lower() for k in ["history", "originated", "in 19", "in 201", "in 2020", "in 2021"]):
        temporal_scope = TemporalScope.HISTORICAL
    else:
        temporal_scope = TemporalScope.STATIC

    return {
        "entities": entities,
        "comparison_targets": comparison_targets,
        "domain": detected_domain,
        "time_requirement": time_req,
        "temporal_scope": temporal_scope,
    }


def build_required_and_excluded_evidence(
    question_type: QuestionType,
    entities: List[str],
    query: str,
) -> Tuple[List[str], List[str]]:
    """Generic evidence requirement builder and topical drift exclusion rule generator."""
    entity_str = ", ".join(entities) if entities else "target entity"

    required_evidence: List[str] = []
    exclude_topics: List[str] = []

    if question_type in (QuestionType.CAUSAL_QUESTION, QuestionType.HOW_IT_WORKS):
        required_evidence = [
            f"Specific mechanism of action / causal pathway of {entity_str}",
            f"Direct cause-and-effect link explaining why/how {entity_str} works",
            "Functional components, active agents, or operating principles",
        ]
        exclude_topics = [
            f"General high-level outcome statistics lacking mechanism for {entity_str}",
            "Unrelated peer entities, alternative products, or competing platforms",
            "Generic context, general prevalence, or tangential trivia without causal link",
            "General safety, side-effects, or regulatory announcements unless directly explaining mechanism",
        ]

    elif question_type == QuestionType.COMPARISON:
        targets = entities if len(entities) >= 2 else ["Entity A", "Entity B"]
        required_evidence = [
            f"Direct comparative metrics between {targets[0]} and {targets[1]} under same criteria",
            f"Architectural or functional advantages of {targets[0]}",
            f"Architectural or functional advantages of {targets[1]}",
            "Identical benchmark conditions or evaluation context",
        ]
        exclude_topics = [
            "One-sided promotion omitting comparable metrics for the other entity",
            "Unrelated third entities not requested in comparison",
            "General tutorials without comparative evaluation",
        ]

    elif question_type == QuestionType.QUANTITATIVE:
        required_evidence = [
            f"Exact quantitative metric, numeric value, rate, or percentage for {entity_str}",
            "Specific units of measurement, sample size, or measurement methodology",
            "Time period or conditions under which measurement was conducted",
        ]
        exclude_topics = [
            "Qualitative marketing claims without verifiable numerical figures",
            "Metrics measured on different populations, versions, or hardware without clarification",
        ]

    elif question_type == QuestionType.CURRENT_LATEST:
        required_evidence = [
            f"Most recent official status, release, or guidance regarding {entity_str}",
            "Explicit publication timestamp or date of update (2024-2026)",
            "Changelog, what changed compared to prior status",
        ]
        exclude_topics = [
            "Outdated historical versions or deprecated announcements",
            "Speculative rumors not grounded in official or Tier 1/2 releases",
        ]

    elif question_type == QuestionType.DEFINITION:
        required_evidence = [
            f"Formal, authoritative definition of {entity_str}",
            "Core characteristics and defining boundaries",
            "Primary use cases or standard specification",
        ]
        exclude_topics = [
            "Long tangential tutorials or implementation details",
            "Market speculation or unrelated news items",
        ]

    elif question_type == QuestionType.PROCEDURE_HOW_TO:
        required_evidence = [
            f"Step-by-step procedural instructions for {entity_str}",
            "Required prerequisites, dependencies, or inputs",
            "Common failure modes, pitfalls, or validation checks",
        ]
        exclude_topics = [
            "Theoretical history or marketing overview",
            "Deprecated workflows or unverified forum scripts",
        ]

    elif question_type == QuestionType.REGULATORY_LEGAL:
        required_evidence = [
            f"Official statutory framework, regulatory mandate, or guidance for {entity_str}",
            "Enforcement jurisdiction, compliance deadlines, or penalties",
            "Official agency or standards body publication references",
        ]
        exclude_topics = [
            "Unofficial personal opinions or non-legal commentaries",
            "Regulations applicable to other jurisdictions without disclosure",
        ]

    else:
        # Generic explanation / factual
        required_evidence = [
            f"Authoritative factual evidence directly addressing '{query}'",
            f"Empirical data or primary documentation regarding {entity_str}",
            "Key contextual parameters and boundary limitations",
        ]
        exclude_topics = [
            "Superficial mentions failing to provide concrete evidence",
            "Tangential discussions of loosely related topics",
        ]

    return required_evidence, exclude_topics


def build_query_understanding_profile(
    query: str,
    llm_analysis: Optional[Dict[str, Any]] = None,
) -> QueryUnderstandingProfile:
    """Constructs a complete QueryUnderstandingProfile combining heuristics and LLM response."""
    # 1. Heuristic baseline
    q_type, q_intent = classify_question_type_heuristic(query)
    extracted = extract_entities_and_context(query)

    entities = extracted["entities"]
    comparison_targets = extracted["comparison_targets"]
    detected_domain = extracted["domain"]
    time_req = extracted["time_requirement"]
    temporal_scope = extracted["temporal_scope"]

    # 2. Refine with LLM analysis if provided
    if llm_analysis and isinstance(llm_analysis, dict):
        if "question_type" in llm_analysis:
            try:
                q_type = QuestionType(llm_analysis["question_type"])
            except Exception:
                pass
        if "intent" in llm_analysis:
            try:
                q_intent = QueryIntent(llm_analysis["intent"])
            except Exception:
                pass
        if "entities" in llm_analysis and isinstance(llm_analysis["entities"], list) and llm_analysis["entities"]:
            entities = [str(e) for e in llm_analysis["entities"]]
        if "comparison_targets" in llm_analysis and isinstance(llm_analysis["comparison_targets"], list):
            comparison_targets = [str(c) for c in llm_analysis["comparison_targets"]]
        if "domain" in llm_analysis:
            detected_domain = str(llm_analysis["domain"])
        if "temporal_scope" in llm_analysis:
            try:
                temporal_scope = TemporalScope(llm_analysis["temporal_scope"])
            except Exception:
                pass

    required_ev, exclude_top = build_required_and_excluded_evidence(q_type, entities, query)

    # Search strategy and format recommendations
    entity_label = " vs ".join(entities) if entities else query
    if q_type == QuestionType.COMPARISON:
        search_strategy = f"Balanced dual-entity retrieval for {entity_label} across common benchmark axes"
        recommended_format = "Comparative Matrix + Key Differences + Dual-Entity Evidence"
    elif q_type in (QuestionType.CAUSAL_QUESTION, QuestionType.HOW_IT_WORKS):
        search_strategy = f"Mechanism-focused deep retrieval targeting causal pathways and operating principles for {entity_label}"
        recommended_format = "Direct Answer + Mechanism & Causal Process + Supporting Evidence + Scope & Limitations"
    elif q_type == QuestionType.CURRENT_LATEST:
        search_strategy = f"Recency-filtered retrieval prioritizing 2024-2026 releases and official updates for {entity_label}"
        recommended_format = "Current Status + Timeline/Dates + Key Changes + Source Attributions"
    elif q_type == QuestionType.QUANTITATIVE:
        search_strategy = f"Metric-specific precision retrieval preserving numerical units and methodology for {entity_label}"
        recommended_format = "Precise Metric Summary + Measurement Context & Units + Supporting Datasets"
    elif q_type == QuestionType.PROCEDURE_HOW_TO:
        search_strategy = f"Procedural step-by-step retrieval detailing requirements and pitfalls for {entity_label}"
        recommended_format = "Prerequisites + Step-by-Step Procedure + Pitfalls & Validation"
    elif q_type == QuestionType.DEFINITION:
        search_strategy = f"Authoritative terminology and specification lookup for {entity_label}"
        recommended_format = "Authoritative Definition + Core Characteristics + Scope Boundaries"
    elif q_type == QuestionType.FACTUAL_QUESTION:
        search_strategy = f"Direct fact-checking and empirical confirmation for {entity_label}"
        recommended_format = "Concise Answer + Verified Evidence + Source Citation"
    else:
        search_strategy = f"Multi-angle research investigation into {entity_label}"
        recommended_format = "Executive Findings + Evidence Breakdown + Limitations"

    return QueryUnderstandingProfile(
        entities=entities,
        main_topic=query,
        question_type=q_type,
        intent=q_intent,
        requested_information=f"Evidence directly addressing: {query}",
        required_evidence=required_ev,
        exclude_topics=exclude_top,
        constraints=[f"Domain: {detected_domain}", f"Temporal: {temporal_scope.value}"],
        time_requirement=time_req,
        comparison_targets=comparison_targets,
        domain=detected_domain,
        temporal_scope=temporal_scope,
        primary_entities=entities,
        search_strategy=search_strategy,
        recommended_format=recommended_format,
    )


def generate_intent_targeted_queries(
    profile: QueryUnderstandingProfile,
    depth: str = "quick",
) -> List[str]:
    """Generates precise, multi-angle search queries tailored to question type and entities."""
    topic = profile.main_topic
    entities = profile.entities
    q_type = profile.question_type
    query_count = 2 if depth == "quick" else 4
    entity_str = " ".join(entities[:2]) if entities else topic

    queries: List[str] = []

    # Clean user query as base search term
    clean_topic = re.sub(r'^(why is there|why are there|what is|how does|why do|why did)\s+', '', topic, flags=re.I)
    clean_topic = re.sub(r'[?!\"\'`]', '', clean_topic).strip()
    if clean_topic:
        queries.append(clean_topic)

    if q_type == QuestionType.COMPARISON and len(profile.comparison_targets) >= 2:
        t1, t2 = profile.comparison_targets[0], profile.comparison_targets[1]
        queries.append(f"{t1} vs {t2} comparison differences")
        queries.append(f"{t1} vs {t2} benchmark")
        queries.append(f"{t1} architecture overview")
        queries.append(f"{t2} architecture overview")

    elif q_type in (QuestionType.CAUSAL_QUESTION, QuestionType.HOW_IT_WORKS):
        if profile.domain == "medical_life_sciences":
            queries.append(f"{entity_str} mechanism of action pathway")
            queries.append(f"why {entity_str} biological mechanism efficacy")
            queries.append(f"{entity_str} underlying principles how it works")
        elif profile.domain == "technical_software":
            queries.append(f"{entity_str} how it works architecture")
            queries.append(f"{entity_str} implementation mechanism")
            queries.append(f"{entity_str} technical explanation")
        else:
            queries.append(f"{entity_str} causes reasons analysis")
            queries.append(f"why {entity_str} background and causes")
            queries.append(f"{entity_str} recent news developments reasons")
            queries.append(f"{entity_str} key factors explained")

    elif q_type == QuestionType.CURRENT_LATEST:
        queries.append(f"{entity_str} latest news update")
        queries.append(f"{entity_str} recent developments what changed")
        queries.append(f"{entity_str} current status overview")

    elif q_type == QuestionType.QUANTITATIVE:
        queries.append(f"{entity_str} statistics numbers data")
        queries.append(f"{entity_str} metrics benchmark figures")

    elif q_type == QuestionType.REGULATORY_LEGAL:
        queries.append(f"{entity_str} regulation legal rules requirements")
        queries.append(f"{entity_str} policy compliance overview")

    elif q_type == QuestionType.PROCEDURE_HOW_TO:
        queries.append(f"how to {entity_str} step by step guide")
        queries.append(f"{entity_str} instructions tutorial")

    elif q_type == QuestionType.DEFINITION:
        queries.append(f"what is {entity_str} definition meaning")
        queries.append(f"{entity_str} overview explanation")

    else:
        queries.append(f"{entity_str} overview and analysis")
        queries.append(f"{entity_str} facts and details")

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped[:query_count]

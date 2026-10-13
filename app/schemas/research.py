"""Pydantic v2 schemas for research requests, reports, claim-level grounding, and adaptive synthesis."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class QuestionType(str, Enum):
    """Generic question-intent classification supported across all domains."""
    FACTUAL_QUESTION = "factual_question"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    CAUSAL_QUESTION = "causal_question"          # "Why does X work?", root causes
    HOW_IT_WORKS = "how_it_works"                # Mechanisms, underlying processes
    COMPARISON = "comparison"                    # Comparative criteria across multiple entities
    RECOMMENDATION = "recommendation"
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURE_HOW_TO = "procedure_how_to"        # Step-by-step procedural guides
    CURRENT_LATEST = "current_latest"            # Contemporary updates requiring date verification
    HISTORICAL = "historical"                    # Epoch-specific historical queries
    QUANTITATIVE = "quantitative"                # Metric, numeric, measurement questions
    EVIDENCE_RESEARCH = "evidence_research"      # Scientific study/literature retrieval
    REGULATORY_LEGAL = "regulatory_legal"        # Compliance, mandates, legal frameworks
    OPINION_INTERPRETATION = "opinion_interpretation"
    SUMMARIZATION = "summarization"


class RelevanceClassification(str, Enum):
    """Direct question relevance classification to prevent topical drift."""
    DIRECTLY_RELEVANT = "DIRECTLY_RELEVANT"          # Directly answers the specific question / mechanism
    PARTIALLY_RELEVANT = "PARTIALLY_RELEVANT"        # Provides necessary partial context or step
    TOPICALLY_RELEVANT_ONLY = "TOPICALLY_RELEVANT_ONLY"# Same topic, but fails to answer specific question
    IRRELEVANT = "IRRELEVANT"                        # Off-topic or unrelated context


class QueryIntent(str, Enum):
    """Categorization of what the user is actually seeking (mapped from QuestionType)."""
    EXPLANATORY = "EXPLANATORY"                  # "Explain X", core mechanisms
    COMPARISON = "COMPARISON"                    # "Compare X and Y", trade-offs
    RECENT_DEVELOPMENTS = "RECENT_DEVELOPMENTS"  # "What is the latest on X?"
    REQUIREMENTS = "REQUIREMENTS"                # "What are the requirements for X?"
    CAUSAL_ANALYSIS = "CAUSAL_ANALYSIS"          # "Why did X happen / why is X effective?"
    LITERATURE_RETRIEVAL = "LITERATURE_RETRIEVAL"# "Give me papers / studies on X"
    FACT_VERIFICATION = "FACT_VERIFICATION"      # "Is X true / feasible?"
    HOW_TO = "HOW_TO"                            # "How to implement / build X?"
    DIRECT_FACT = "DIRECT_FACT"                  # Simple factual question
    QUANTITATIVE = "QUANTITATIVE"                # Numeric/metric inquiries
    DEFINITION = "DEFINITION"                    # Authoritative terminology definitions


class TemporalScope(str, Enum):
    """Temporal sensitivity of the research query."""
    STATIC = "STATIC"            # Timeless foundational knowledge (e.g. TCP/IP)
    RECENT = "RECENT"            # Contemporary, fast-moving updates (2024-2026)
    HISTORICAL = "HISTORICAL"    # Specific past epoch or event


class SourceTier(str, Enum):
    """Categorization of source quality and authority."""
    TIER_1_PRIMARY = "TIER_1_PRIMARY"      # Official docs, standards, gov, original arXiv
    TIER_2_SECONDARY = "TIER_2_SECONDARY"  # Major journals, Reuters, Bloomberg, reputable analysis
    TIER_3_TERTIARY = "TIER_3_TERTIARY"    # Blogs, generic web, community summaries


class SupportLevel(str, Enum):
    """Evidence support confidence rating for each claim."""
    WELL_SUPPORTED = "WELL_SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INFERENCE = "INFERENCE"


class ScoringWeights(BaseModel):
    """Configurable weights for hybrid multi-factor candidate chunk reranking."""
    w_semantic: float = Field(default=0.20, description="Weight for semantic/vector similarity")
    w_entity: float = Field(default=0.20, description="Weight for exact entity and alias matching")
    w_intent: float = Field(default=0.20, description="Weight for question-type keyword/intent match")
    w_question: float = Field(default=0.20, description="Weight for direct question answering capability")
    w_source: float = Field(default=0.10, description="Weight for authority source tier")
    w_recency: float = Field(default=0.10, description="Weight for temporal proximity when applicable")


class QueryUnderstandingProfile(BaseModel):
    """Structured decomposition and deep intent profile of the user directive."""
    entities: List[str] = Field(default_factory=list, description="Extracted target subjects / entities")
    main_topic: str = Field(default="", description="High-level subject domain or theme")
    question_type: QuestionType = Field(default=QuestionType.EXPLANATION, description="Classified generic question type")
    intent: QueryIntent = Field(default=QueryIntent.EXPLANATORY, description="High-level intent mapping")
    requested_information: str = Field(default="", description="Specific information required to answer query")
    required_evidence: List[str] = Field(default_factory=list, description="Concrete evidence facets needed")
    exclude_topics: List[str] = Field(default_factory=list, description="Topical drift areas to explicitly filter out")
    constraints: List[str] = Field(default_factory=list, description="Target conditions, populations, or scopes")
    time_requirement: Optional[str] = Field(default=None, description="Specific era or recency constraint if any")
    comparison_targets: List[str] = Field(default_factory=list, description="Entities involved if comparison")
    domain: Optional[str] = Field(default="general", description="Detected scientific/technical/legal/general domain")
    temporal_scope: TemporalScope = Field(default=TemporalScope.RECENT, description="Temporal boundary needed")
    primary_entities: List[str] = Field(default_factory=list, description="Key subjects (alias for entities)")
    search_strategy: str = Field(default="Targeted multi-angle retrieval", description="Strategy for query fan-out")
    recommended_format: str = Field(default="Direct Answer + Mechanism + Evidence", description="Structural layout")


class EvidenceItem(BaseModel):
    """Structured evidence item verified and accepted into the Evidence Set before answer generation."""
    evidence_id: str = Field(..., description="Unique evidence identifier, e.g. EVD-01")
    text: str = Field(..., description="Verbatim extracted excerpt containing the proof")
    source_url: str = Field(..., description="Canonical source URL")
    source_title: str = Field(default="", description="Page title")
    entity: str = Field(default="", description="Target entity this evidence applies to")
    relevance_classification: RelevanceClassification = Field(..., description="Direct or partial relevance classification")
    relevance_score: float = Field(default=0.0, description="Calculated multi-factor score [0.0 - 1.0]")
    source_tier: SourceTier = Field(default=SourceTier.TIER_2_SECONDARY, description="Source authority tier")
    evidence_type: str = Field(default="empirical", description="Category: mechanism, quantitative_metric, definition, etc.")
    publication_date: Optional[str] = Field(default=None, description="Extracted or detected publication timestamp")


class DebugTrace(BaseModel):
    """Developer observability trace detailing retrieval and verification decisions."""
    parsed_query: Dict[str, Any] = Field(default_factory=dict)
    candidate_chunks_evaluated: int = Field(default=0)
    accepted_evidence_count: int = Field(default=0)
    rejected_chunks_count: int = Field(default=0)
    rejected_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    scoring_weights: Dict[str, float] = Field(default_factory=dict)
    insufficient_evidence_flag: bool = Field(default=False)


class Citation(BaseModel):
    """Verified reference source supporting research findings."""
    model_config = ConfigDict(extra="ignore")

    title: str = Field(..., description="Title of the source webpage or article")
    url: str = Field(..., description="Canonical URL of the source")
    snippet: str = Field(default="", description="Relevant textual excerpt")
    tier: SourceTier = Field(default=SourceTier.TIER_2_SECONDARY, description="Source quality tier")
    domain: Optional[str] = Field(default=None, description="Extracted domain hostname")


class SourceItem(BaseModel):
    """Raw scraped source item cached during agent execution."""
    title: str
    url: str
    content: str
    query: str
    tier: SourceTier = SourceTier.TIER_2_SECONDARY
    domain: Optional[str] = None


class ClaimGrounding(BaseModel):
    """Fine-grained atomic claim mapped to exact backing sources."""
    claim_id: str = Field(..., description="Unique claim tag (e.g. CLM-01)")
    claim_text: str = Field(..., description="The factual assertion")
    support_level: SupportLevel = Field(..., description="Evidence verification status")
    source_urls: List[str] = Field(default_factory=list, description="URLs backing this claim")
    supporting_quote: Optional[str] = Field(default=None, description="Direct quote from scraped evidence")


class ContradictionItem(BaseModel):
    """Explicit disagreement between retrieved sources."""
    topic: str = Field(..., description="The contested topic or metric")
    source_a_url: str = Field(..., description="First source URL")
    source_a_claim: str = Field(..., description="Claim made by source A")
    source_b_url: str = Field(..., description="Second source URL")
    source_b_claim: str = Field(..., description="Contradicting claim made by source B")
    discrepancy_analysis: str = Field(..., description="Root cause of the divergence (methodology, timeline, etc.)")


class ResearchRequest(BaseModel):
    """Payload for submitting a new autonomous research task."""
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research topic, question, or thesis to investigate",
        examples=["OpenAI Swarm vs LangGraph vs CrewAI architecture benchmark"],
    )
    depth: Literal["quick", "deep"] = Field(
        default="quick",
        description="'quick' performs 2 targeted queries; 'deep' performs comprehensive fan-out with iterative reflection",
    )
    max_sources: int = Field(
        default=10,
        ge=2,
        le=30,
        description="Maximum number of distinct sources to retrieve and analyze",
    )
    time_limit_seconds: int = Field(
        default=120,
        ge=15,
        le=300,
        description="Global timeout in seconds for autonomous research execution",
    )


class ResearchStartResponse(BaseModel):
    """Immediate response after launching a research task."""
    task_id: str = Field(..., description="Unique UUID identifier for tracking the task")
    status: Literal["queued", "processing", "completed", "failed", "cancelled"] = Field(
        default="processing", description="Current execution state"
    )
    message: str = Field(
        default="Research task launched successfully.",
        description="Descriptive confirmation message",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of initiation",
    )


class CancelResponse(BaseModel):
    """Confirmation payload returned when terminating an active research job."""
    task_id: str = Field(..., description="Terminated task identifier")
    status: Literal["cancelled"] = Field(default="cancelled")
    message: str = Field(..., description="Termination status summary")


class TaskSummaryItem(BaseModel):
    """Condensed summary item for session mission history."""
    task_id: str
    topic: str
    status: str
    depth: str
    query_intent: Optional[str] = None
    created_at: str
    sources_count: int = 0


class ResearchReport(BaseModel):
    """Synthesized research output delivered to the client."""
    task_id: str = Field(..., description="Task identifier associated with this report")
    topic: str = Field(..., description="Research topic queried")
    query_profile: Optional[QueryUnderstandingProfile] = Field(
        default=None, description="Intent and temporal classification"
    )
    executive_summary: str = Field(
        ..., description="High-level overview and synthesis of findings"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Bulleted list of validated factual discoveries"
    )
    markdown_content: str = Field(
        ..., description="Full production markdown report with adaptive structuring"
    )
    claims: List[ClaimGrounding] = Field(
        default_factory=list, description="Claim-level evidence grounding map"
    )
    contradictions: List[ContradictionItem] = Field(
        default_factory=list, description="Detected source disagreements and contradictions"
    )
    overall_uncertainty: SupportLevel = Field(
        default=SupportLevel.WELL_SUPPORTED, description="System-level confidence rating"
    )
    citations: List[Citation] = Field(
        default_factory=list, description="Verified references supporting the report"
    )
    evidence_set: List[EvidenceItem] = Field(
        default_factory=list, description="Curated accepted evidence units passed to answer generation"
    )
    debug_trace: Optional[DebugTrace] = Field(
        default=None, description="Observability trace with scores, rejected chunks, and evaluations"
    )
    insufficient_evidence_acknowledged: bool = Field(
        default=False, description="Flag indicating if the system recognized evidence insufficiency"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when synthesis completed",
    )
    iterations_used: int = Field(
        default=1, description="Number of reflection/correction loops executed"
    )


class TaskStatusResponse(BaseModel):
    """Live state of an ongoing or completed research task."""
    task_id: str = Field(..., description="Unique task identifier")
    status: Literal["queued", "processing", "completed", "failed", "cancelled"] = Field(
        ..., description="Execution status"
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Active node in LangGraph (planner, researcher, critic, synthesizer, verifier)",
    )
    active_substep: Optional[str] = Field(
        default=None,
        description="Fine-grained action description (e.g. 'Classifying query intent')",
    )
    iteration: int = Field(
        default=1, description="Current reflection loop iteration count"
    )
    query_intent: Optional[str] = Field(
        default=None, description="Classified intent (e.g. COMPARISON, FACT_VERIFICATION)"
    )
    logs: List[dict] = Field(
        default_factory=list,
        description="Live telemetry timeline of agent actions",
    )
    error: Optional[str] = Field(
        default=None, description="Error detail if execution encountered a failure"
    )
    report_available: bool = Field(
        default=False, description="Whether final report is generated and queryable"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last checkpoint timestamp",
    )


class QuestionRequest(BaseModel):
    """Payload for asking grounded follow-up questions over a research report."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Follow-up question regarding the research findings",
    )


class AnswerResponse(BaseModel):
    """Grounded, zero-hallucination response to a follow-up question."""
    task_id: str = Field(..., description="Task identifier")
    question: str = Field(..., description="User question asked")
    answer: str = Field(..., description="Grounded answer based strictly on verified research sources")
    grounded_sources: List[str] = Field(
        default_factory=list,
        description="List of canonical source URLs supporting this specific answer",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of generation",
    )

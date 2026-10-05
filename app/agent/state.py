"""TypedDict state definition for LangGraph research workflow."""

from typing import Any, Dict, List, Optional, TypedDict


class ResearchState(TypedDict, total=False):
    """Execution state tracked across all nodes in the LangGraph workflow."""

    task_id: str
    topic: str
    depth: str  # "quick" or "deep"
    query_profile: Dict[str, Any]  # Intent, temporal scope, entities, strategy, format
    plan: List[str]  # Search queries planned for investigation
    queries_searched: List[str]  # Queries already executed
    scraped_sources: List[Dict[str, Any]]  # List of {title, url, snippet, content, tier, domain}
    citations: List[Dict[str, Any]]  # Ranked and canonicalized citations
    draft_report: str  # Intermediate findings or draft
    critique: str  # Feedback and critique from reflection node
    critique_passed: bool  # Whether research meets sufficiency standards
    contradictions: List[Dict[str, Any]]  # Disagreements detected between sources
    claims: List[Dict[str, Any]]  # Claim-level evidence grounding entries
    overall_uncertainty: str  # WELL_SUPPORTED, PARTIALLY_SUPPORTED, etc.
    iteration_count: int  # Current loop iteration index (1-indexed)
    current_step: str  # Current active node for progress reporting
    active_substep: str  # Fine-grained action description (e.g. 'Classifying query intent')
    logs: List[Dict[str, Any]]  # Live telemetry timeline of agent actions
    final_report: Optional[Dict[str, Any]]  # Synthesized final report payload
    verification_passed: bool  # Final claim verification gate status
    error: Optional[str]  # Error message if failure encountered

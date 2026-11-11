"""H-EDGE: Hierarchical knowledge hypergraphs and evidence-driven exploration."""
from .knowledge_hypergraph import HierarchicalKnowledgeHypergraph
from .evidence_slots import DependencyAwareEvidenceSlots
from .hierarchical_retrieval import HierarchicalRetrieval
from .source_verified_binding import SourceVerifiedBinding
from .evidence_repair import EvidenceGroundedRepair
from .source_bridge import SourceBridge
from .pipeline import HEDGE, Settings

__all__ = ["HEDGE", "Settings", "HierarchicalKnowledgeHypergraph",
           "DependencyAwareEvidenceSlots", "HierarchicalRetrieval", "SourceVerifiedBinding",
           "EvidenceGroundedRepair", "SourceBridge"]

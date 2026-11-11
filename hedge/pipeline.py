"""H-EDGE: Evidence-Driven Graph Exploration, Algorithm 1."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from components.source_qa.reasoning import read_answer
from .evidence_slots import DependencyAwareEvidenceSlots
from .hierarchical_retrieval import HierarchicalRetrieval
from .knowledge_hypergraph import VisitBudget
from .source_bridge import SourceBridge
from .source_verified_binding import SourceVerifiedBinding
from .evidence_witnesses import compose_evidence, reader_packet
from .evidence_repair import EvidenceGroundedRepair


@dataclass(frozen=True)
class Settings:
    visit_budget: int = 256
    document_budget: int = 5
    max_anchors: int = 3
    fact_ports: bool = True
    source_mentions: bool = True
    implicit_anchors: bool = True
    persist_repairs: bool = True

    def __post_init__(self):
        if (type(self.document_budget) is not int or not 1 <= self.document_budget <= 5
                or type(self.visit_budget) is not int or self.visit_budget < 1
                or type(self.max_anchors) is not int or self.max_anchors < 1):
            raise ValueError("H-EDGE needs a positive visit budget and at most five reader documents")


class _Allocation:
    """A local cap that charges the shared question budget exactly once."""
    def __init__(self, parent, limit):
        self.parent, self.left = parent, max(0, limit)

    def take(self, kind):
        if self.left <= 0 or not self.parent.take(kind):
            return False
        self.left -= 1
        return True


class HEDGE:
    def __init__(self, memory, llm, dense, fact_scorer, settings=None):
        self.memory, self.llm, self.dense = memory, llm, dense
        self.fact_scorer = fact_scorer
        self.settings = settings or Settings()

    def _ranking(self, question):
        scores = self.dense.scores(question)
        if (not scores or any(d not in self.memory.documents for d in scores)
                or any(not math.isfinite(v) for v in scores.values())):
            raise ValueError("dense retrieval must return finite corpus-aligned scores")
        return sorted(scores, key=lambda d: (-scores[d], d))

    def run(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("nonempty question required")
        settings, memory = self.settings, self.memory
        version = memory.version
        budget = VisitBudget(settings.visit_budget)
        slots = DependencyAwareEvidenceSlots(self.llm)
        planned = slots.construct(question, memory)
        hierarchical = HierarchicalRetrieval(memory, self.fact_scorer)
        bridge = SourceBridge(memory.source_store)
        binder = SourceVerifiedBinding(self.llm, memory)
        repair = EvidenceGroundedRepair(self.llm, memory)
        initial = self._ranking(question)
        accepted, witnesses, steps, proposals = {}, {}, [], []
        goals = planned["requirements"]
        for offset, template in enumerate(goals):
            if not slots.ready(template, accepted):
                steps.append({"goal_id": template["id"], "accepted": False, "error": "unclosed_parent"})
                continue
            if not budget.remaining:
                steps.append({"goal_id": template["id"], "accepted": False, "error": "visit_budget_exhausted"})
                continue
            goal = slots.instantiate(template, accepted)
            # Allocate across remaining goals, reserving a visit for a recovered
            # source assertion. Both structural routes share the same B.
            allowance = max(1, budget.remaining // (len(goals) - offset))
            start_used = budget.used
            route = hierarchical.retrieve(goal, _Allocation(budget, (allowance - 1) // 2))
            dense = self._ranking(goal["text"])
            bridged = bridge.select(goal["text"], dense, top_k=settings.document_budget,
                                    max_anchors=settings.max_anchors,
                                    fact_ports=settings.fact_ports, source_mentions=settings.source_mentions,
                                    implicit_anchors=settings.implicit_anchors,
                                    budget=_Allocation(budget, allowance - 1 - (budget.used - start_used)))
            structural_docs = [s["doc_idx"] for f in route["candidates"] for s in f["sources"]]
            docs = list(dict.fromkeys(structural_docs + bridged["indices"]))
            step = binder.solve(question, goal, docs, route["candidates"],
                                [accepted[d] for d in template["depends_on"]],
                                _Allocation(budget, allowance - (budget.used - start_used)))
            step.update(goal_id=goal["id"], route={"hierarchy": route, "bridge": bridged})
            if step["accepted"]:
                witness = compose_evidence(goal, step, [witnesses[d] for d in template["depends_on"]])
                step["witness_id"] = witness["id"]
                accepted[goal["id"]] = step
                witnesses[goal["id"]] = witness
                proposals.extend(repair.propose(step, witness))
            steps.append(step)
        docs, reader_witnesses, chain = reader_packet(steps, list(witnesses.values()),
                                                     initial, settings.document_budget)
        closed = len(accepted) == len(goals)
        objects = {**memory.facts, **{s["fact"]["id"]: s["fact"] for s in accepted.values()},
                   **{w["id"]: w for w in witnesses.values()}}
        reader_objects = {}

        def retain(fid):
            if fid in reader_objects:
                return
            obj = objects[fid]
            for member in obj.get("members", []):
                retain(member)
            reader_objects[fid] = obj

        for witness in reader_witnesses:
            retain(witness["id"])
        answer = read_answer(self.llm, memory.source_store, question, docs, chain,
                             chain_closed=closed and len(chain) == len(goals),
                             operation=planned["operation"], witnesses=reader_witnesses,
                             occurrences=list(reader_objects.values()))
        # No current-question component is run against the updated memory.
        ledger = []
        if settings.persist_repairs and answer["error"] is None:
            self.memory, ledger = repair.admit_after_answer(proposals, answer_completed=True)
        return {"method_identity": "H-EDGE", "question": question, "plan": planned,
                "steps": steps, "witnesses": list(witnesses.values()),
                "reader_witnesses": reader_witnesses, "chain_closed": closed,
                "reader_occurrences": list(reader_objects.values()),
                "final_indices": docs, "output": answer,
                "repair_proposals": proposals, "repair_ledger": ledger,
                "memory_version_answered": version, "memory_version_after": self.memory.version,
                "visits": {"limit": budget.limit, "used": budget.used, "by_kind": dict(budget.counts)},
                "settings": asdict(settings)}

"""Hierarchical Retrieval and Binding: candidate exploration (Section IV-B)."""
from __future__ import annotations

from .knowledge_hypergraph import entails
from components.source_qa.store import norm


class HierarchicalRetrieval:
    def __init__(self, memory, fact_scorer):
        self.memory, self.fact_scorer = memory, fact_scorer

    def retrieve(self, goal, budget):
        memory = self.memory
        candidates, visited_schemas, visited_facts, routes = [], set(), set(), []
        anchored = memory.anchored_occurrences(goal["inputs"])
        concepts = sorted(memory.concepts.values(),
                          key=lambda c: (c["id"] != goal["domain"], c["id"]))
        for concept in concepts:
            if not budget.take("concept"):
                break
            for sid in concept["schemas"]:
                if sid in visited_schemas:
                    continue
                if not budget.take("schema"):
                    break
                visited_schemas.add(sid)
                schema = memory.schemas[sid]
                if norm(goal["predicate"]) not in {norm(p) for p in schema["predicates"]}:
                    continue
                for fid in memory.memberships[sid]:
                    if fid in visited_facts or fid not in anchored:
                        continue
                    fact = memory.facts[fid]
                    if not budget.take("composition" if fact["depth"] else "fact"):
                        break
                    visited_facts.add(fid)
                    # Exposed values and qualifiers narrow candidates. The
                    # source verifier establishes directed role alignment.
                    if (all(any(memory.same_value(v, x) for x in fact["roles"].values())
                            for v in goal["inputs"].values())
                            and entails(fact["qualifiers"], goal["qualifiers"])):
                        candidates.append(fact)
                        routes.append({"concept": concept["id"], "schema": sid,
                                       "occurrence": fid, "sources": fact["sources"]})
        scores = self.fact_scorer(goal["text"], candidates) if candidates else {}
        if set(scores) != {f["id"] for f in candidates}:
            raise ValueError("occurrence scorer must preserve candidate identity")
        import math
        if any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("non-finite occurrence similarity")
        candidates.sort(key=lambda f: (-scores[f["id"]], f["id"]))
        return {"candidates": candidates, "routes": routes,
                "scores": scores, "visited_schemas": sorted(visited_schemas)}

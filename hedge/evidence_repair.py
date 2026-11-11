"""Evidence-Grounded Repair (IV-C): proposals, admission, then H_t -> H_t+1."""
from __future__ import annotations

import copy

from .knowledge_hypergraph import HierarchicalKnowledgeHypergraph, identity
from components.source_qa.store import norm


ADMIT = """Check a proposed persistent relation using ONLY the supplied original
source spans. Verify the predicate, each named role, argument identity/direction,
qualifiers, and constituent joins. Source overlap or an answer string alone is
insufficient. Return JSON {"supported":true|false,"identities_consistent":true|false,
"qualifiers_supported":true|false,"reason":"brief explanation"}.
"""


class EvidenceGroundedRepair:
    def __init__(self, llm, memory):
        self.llm, self.memory = llm, memory

    def propose(self, step, witness):
        if step.get("accepted") is not True:
            return []
        proposals = []
        for fact in (step["fact"], witness):
            if fact["id"] in self.memory.facts:
                continue
            proposals.append({"id": identity("repair", fact),
                              "kind": "composition" if fact["depth"] else "fact",
                              "residual_requirement": step["goal"],
                              "supported_answer": step["answer"],
                              "sources": copy.deepcopy(fact["sources"]),
                              "assertion": copy.deepcopy(fact)})
        fact = step["fact"]
        if not fact["depth"]:
            # Title uniqueness proposes a correspondence; a separate source
            # check must establish identity before it becomes a confirmed port.
            for role, value in fact["roles"].items():
                targets = self.memory.title_documents.get(norm(value), [])
                if len(targets) != 1 or any(p.get("role") == role and p.get("identity_confirmed")
                                             for p in fact.get("ports", [])):
                    continue
                target = targets[0]
                existing_ids = self.memory.source_store.alias_lookup.get(norm(value), [])
                assertion = {"fact_id": fact["id"], "role": role, "surface": value,
                             "target_doc": target,
                             "entity_id": existing_ids[0] if len(existing_ids) == 1 else identity("entity", target)}
                proposals.append({"id": identity("repair_anchor", assertion), "kind": "anchor",
                                  "residual_requirement": step["goal"], "supported_answer": step["answer"],
                                  "sources": copy.deepcopy(fact["sources"]), "assertion": assertion})
        return proposals

    def admit_after_answer(self, proposals, *, answer_completed):
        if not answer_completed:
            raise ValueError("current answer must finish before admitting repairs")
        # A separate candidate memory prevents any proposal from affecting the
        # question that produced it, including failed admission checks.
        old = self.memory
        candidate = HierarchicalKnowledgeHypergraph(list(old.documents.values()), old.snapshot())
        ledger, seen = [], set()
        for proposal in proposals:
            if proposal["id"] in seen:
                continue
            seen.add(proposal["id"])
            entry = {**copy.deepcopy(proposal), "admitted": False}
            fact = proposal["assertion"]
            try:
                if proposal["kind"] == "anchor":
                    entry.update(self._admit_anchor(candidate, proposal))
                    ledger.append(entry)
                    continue
                candidate.validate_fact(fact)
                spans = [{**s, "text": old.documents[s["doc_idx"]]["text"][s["start"]:s["end"]]}
                         for s in fact["sources"]]
                verdict = self.llm.ask("repair_admission", ADMIT,
                                       {"assertion": fact, "source_spans": spans,
                                        "constituents": [candidate.facts[m] for m in fact.get("members", [])]},
                                       tokens=400)
                entry["verdict"] = verdict
                if any(verdict.get(k) is not True for k in
                       ("supported", "identities_consistent", "qualifiers_supported")):
                    raise ValueError("source verification rejected persistent update")
                candidate.add(fact)
                entry["admitted"] = True
            except (KeyError, TypeError, ValueError) as exc:
                entry["reason"] = str(exc)
            ledger.append(entry)
        if any(e["admitted"] for e in ledger):
            candidate.version = old.version + 1
            candidate._refresh_source_store()
        return candidate, ledger

    def _admit_anchor(self, memory, proposal):
        anchor = proposal["assertion"]
        fact = memory.facts[anchor["fact_id"]]
        target = memory.documents[anchor["target_doc"]]
        if (fact["roles"].get(anchor["role"]) != anchor["surface"]
                or norm(target["title"]) != norm(anchor["surface"])
                or not memory.schema_ids(fact)):
            raise ValueError("anchor does not match the occurrence role and receiving schema")
        memory.validate_sources(proposal["sources"])
        verdict = self.llm.ask("anchor_admission",
                               "Verify that this cited occurrence role denotes the entity described by the target "
                               "document. A matching name alone is insufficient; reject ambiguous identities. "
                               'Return JSON {"same_entity":true|false,"reason":"explanation"}.',
                               {"anchor": anchor, "fact": fact,
                                "sources": [{**s, "text": memory.documents[s["doc_idx"]]["text"][s["start"]:s["end"]]}
                                            for s in proposal["sources"]], "target": target}, tokens=350)
        if verdict.get("same_entity") is not True:
            raise ValueError("anchor identity verification rejected")
        port = {"role": anchor["role"], "surface": anchor["surface"], "identity_confirmed": True,
                "alternatives": [{"id": anchor["entity_id"]}]}
        if port not in fact.setdefault("ports", []):
            fact["ports"].append(port)
        ids = memory.aliases.setdefault(anchor["surface"], [])
        if anchor["entity_id"] not in ids:
            ids.append(anchor["entity_id"])
        memory.anchors.append({**copy.deepcopy(anchor), "sources": copy.deepcopy(proposal["sources"])})
        return {"admitted": True, "verdict": verdict}

"""Source-Verified Binding: Compat AND Verify, before extending Z."""
from __future__ import annotations

import copy

from components.source_qa.reasoning import SourceCatalog, contains, source_spelling
from .knowledge_hypergraph import compatible, identity


EXTRACT = """Propose one occurrence satisfying the residual evidence goal from
ONLY the supplied indexed candidates and original source blocks. Return JSON:
{"answer":"exact source value or empty", "candidate_id":"existing ID or null",
 "evidence":["block ID"], "predicate":"relation", "roles":{"role":"value"},
 "qualifiers":{"name":"value"}, "alignment":{"goal role":"candidate role"}}.
An existing candidate must retain its roles, predicate, qualifiers and sources.
If extraction missed a relation, encode one provisional occurrence with named
roles and exact citations. Do not join unrelated occurrences. Every goal input
and its requested output must map injectively to roles of ONE evidence object.
Preserve relation direction and qualifiers. Return an empty answer if unsupported.
"""

VERIFY = """Check the entire proposed relation against ONLY its cited original
source spans and the original/residual question. Check the predicate family,
role alignment, bound input identities, output, argument direction, and ALL
qualifiers. For a composite also check its constituent relations and joins.
A matching answer word or co-occurring names do not establish a relation.
Return JSON {"supported":true|false,"binding_consistent":true|false,
"predicate_compatible":true|false,"reason":"brief explanation"}.
"""


class SourceVerifiedBinding:
    def __init__(self, llm, memory):
        self.llm, self.memory = llm, memory

    def solve(self, question, goal, doc_ids, candidates, parents, budget):
        memory = self.memory
        catalog = SourceCatalog(memory.source_store, doc_ids, goal["text"],
                                blocks_per_doc=9, max_blocks=45)
        # Indexed evidence must be verified using its own provenance, including
        # spans outside the lexical window shortlist.
        for fact in candidates:
            for span in fact["sources"]:
                key = identity("span", [span[k] for k in ("doc_idx", "start", "end")])
                doc = memory.documents[span["doc_idx"]]
                catalog.blocks[key] = {**span, "id": key, "title": doc["title"],
                                       "text": doc["text"][span["start"]:span["end"]]}
        if not catalog.blocks:
            return {"accepted": False, "error": "no_source_blocks"}
        try:
            proposal = self.llm.ask("binding_proposal", EXTRACT,
                                    {"original_question": question, "goal": goal,
                                     "candidates": candidates, "parents": parents,
                                     "blocks": catalog.payload()}, tokens=1000)
            blocks = catalog.resolve(proposal["answer"], proposal["evidence"])
            answer = source_spelling(proposal["answer"], blocks)
            alignment = proposal["alignment"]
            by_id = {f["id"]: f for f in candidates}
            fid = proposal.get("candidate_id")
            if fid is not None:
                if fid not in by_id:
                    raise ValueError("proposal selected an unvisited occurrence")
                fact = copy.deepcopy(by_id[fid])
            else:
                if not budget.take("recovered_fact"):
                    raise ValueError("visit budget exhausted before recovered fact")
                fact = {"predicate": proposal["predicate"], "roles": proposal["roles"],
                        "qualifiers": proposal["qualifiers"], "depth": 0,
                        "sources": [{k: b[k] for k in ("doc_idx", "start", "end")} for b in blocks]}
                fact["id"] = identity("occurrence", fact)
            memory.validate_fact(fact, require_schema=False)
            if not compatible(fact, goal, answer, alignment, memory.same_value):
                raise ValueError("role, input, output or qualifier compatibility rejected")
            sources = []
            for span in fact["sources"]:
                doc = memory.documents[span["doc_idx"]]
                sources.append({**span, "title": doc["title"],
                                "text": doc["text"][span["start"]:span["end"]]})
            if not any(contains(b["text"], answer) for b in sources):
                raise ValueError("answer absent from the candidate's own provenance")
            verdict = self.llm.ask("binding_verification", VERIFY,
                                   {"original_question": question, "goal": goal,
                                    "answer": answer, "candidate": fact, "alignment": alignment,
                                    "constituents": [memory.facts[m] for m in fact.get("members", [])],
                                    "cited_blocks": sources}, tokens=400)
            if any(verdict.get(k) is not True for k in
                   ("supported", "binding_consistent", "predicate_compatible")):
                raise ValueError("source verification rejected: " + str(verdict.get("reason", "")))
            return {"accepted": True, "question": goal["text"], "answer": answer,
                    "fact": fact, "alignment": alignment, "sources": sources,
                    "verdict": verdict, "goal": goal}
        except (KeyError, TypeError, ValueError) as exc:
            return {"accepted": False, "question": goal["text"], "error": str(exc)}

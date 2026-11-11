"""Recursive Evidence Hyperedges: dependency joins and provenance closure."""
from __future__ import annotations

import copy

from components.source_qa.store import norm
from .knowledge_hypergraph import identity


def compose_evidence(goal, accepted, predecessor_witnesses):
    if accepted.get("accepted") is not True:
        raise ValueError("only accepted realizations may become witnesses")
    fact = accepted["fact"]
    members = [fact] + list(predecessor_witnesses)
    joins = []
    for predecessor in predecessor_witnesses:
        output = predecessor["output_role"]
        value = predecessor["roles"][output]
        matching = [r for r, v in goal["inputs"].items() if norm(v) == norm(value)]
        if not matching:
            raise ValueError("predecessor output is not an input of the residual goal")
        for role in matching:
            joins.append({"left": predecessor["id"], "left_role": output,
                          "right": fact["id"], "right_role": accepted["alignment"][role],
                          "value": value})
    spans = {(s["doc_idx"], s["start"], s["end"]):
             {k: s[k] for k in ("doc_idx", "start", "end")}
             for member in members for s in member["sources"]}
    witness = {"predicate": goal["predicate"],
               "roles": {**goal["inputs"], goal["output_role"]: accepted["answer"]},
               "qualifiers": copy.deepcopy(goal["qualifiers"]),
               "output_role": goal["output_role"],
               "members": list(dict.fromkeys(m["id"] for m in members)),
               "joins": joins, "depth": 1 + max(m["depth"] for m in members),
               "sources": [spans[key] for key in sorted(spans)],
               "goal_id": goal["id"], "accepted": True}
    witness["id"] = identity("witness", witness)
    return witness


def reader_packet(steps, witnesses, initial, top_k=5):
    """Keep a witness only when ALL its source documents fit the reader packet.

    Prefer complete deep compositions, then other accepted branches. Never
    truncate a witness's provenance to make it fit the document budget.
    """
    selected = []
    for witness in sorted(witnesses, key=lambda w: -w["depth"]):
        docs = list(dict.fromkeys(s["doc_idx"] for s in witness["sources"]))
        union = list(dict.fromkeys(selected + docs))
        if len(union) <= top_k:
            selected = union
    selected = list(dict.fromkeys(selected + initial))[:top_k]
    included = {w["id"] for w in witnesses
                if {s["doc_idx"] for s in w["sources"]} <= set(selected)}
    chosen = [w for w in witnesses if w["id"] in included]
    chain = [s for s in steps if s.get("accepted") and s["witness_id"] in included]
    return selected, chosen, chain

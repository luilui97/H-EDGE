"""Assemble extracted role hyperedges into the Hierarchical Knowledge Hypergraph.

Extraction is performed by components.hyperedge_extraction. This export step
uses supplied relation-schema/domain-concept definitions and never calls a
model, invents a semantic hierarchy or drops unmatched occurrences.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sqlite3

from components.hyperedge_extraction import digest, load_documents, validate_facts, write_json
from components.source_qa.store import norm
from .knowledge_hypergraph import HierarchicalKnowledgeHypergraph


def build_index(documents, state, hierarchy, *, allow_incomplete=False):
    state = Path(state).resolve()
    if not state.is_file():
        raise FileNotFoundError("extraction state does not exist: " + str(state))
    db = sqlite3.connect(state.as_uri() + "?mode=ro", uri=True)
    try:
        row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row is None:
            raise ValueError("extraction state lacks its input identity")
        provenance = json.loads(row[0])
        if (provenance.get("format") != "hedge-extraction-state-1"
                or provenance.get("corpus_sha256") != digest(documents)):
            raise ValueError("extraction state does not match the source corpus")
        statuses = dict(db.execute("SELECT doc_idx, status FROM passages"))
        docs = {d["doc_idx"]: d for d in documents}
        if not set(statuses) <= docs.keys() or any(s not in {"done", "partial", "failed"} for s in statuses.values()):
            raise ValueError("invalid extraction progress records")
        complete = len(statuses) == len(docs) and all(s == "done" for s in statuses.values())
        if not complete and not allow_incomplete:
            raise ValueError("extraction is incomplete; resume it before exporting, or explicitly use --allow-incomplete")
        records = list(db.execute("SELECT id, doc_idx, record_json FROM occurrences ORDER BY doc_idx, id"))
    finally:
        db.close()
    # Validate the supplied hierarchy even when the extraction is empty.
    base = {"occurrences": [], "schemas": copy.deepcopy(hierarchy["schemas"]),
            "concepts": copy.deepcopy(hierarchy["concepts"]),
            "aliases": copy.deepcopy(hierarchy.get("aliases", {})),
            "anchors": copy.deepcopy(hierarchy.get("anchors", []))}
    memory = HierarchicalKnowledgeHypergraph(documents, base)
    unmapped = []
    for fid, did, raw in records:
        fact = json.loads(raw)
        if (did not in docs or fact.get("id") != fid or fact.get("depth") != 0
                or fact.get("semantic_verified") is not False or len(fact.get("sources", [])) != 1
                or fact["sources"][0].get("doc_idx") != did
                or fact.get("extraction", {}).get("model_digest") != provenance["model_digest"]):
            raise ValueError("invalid extracted occurrence identity: " + fid)
        source = fact["sources"][0]
        accepted, errors = validate_facts({"facts": [{"predicate": fact["predicate"], "roles": fact["roles"],
                                                      "qualifiers": fact["qualifiers"], "quote": fact["quote"],
                                                      "start": source["start"]}]}, docs[did])
        if errors or accepted[0]["end"] != source["end"]:
            raise ValueError("extracted occurrence no longer matches its source: " + fid)
        # The source component emits subject/object roles. Explicit schema
        # alignments can map a supplied semantic role vocabulary onto them.
        alignments = {schema["id"]: copy.deepcopy(schema["role_alignment"])
                      for schema in hierarchy["schemas"] if "role_alignment" in schema
                      and norm(fact["predicate"]) in {norm(p) for p in schema["predicates"]}}
        if alignments:
            fact["schema_alignments"] = alignments
        if not memory.schema_ids(fact):
            unmapped.append({"id": fid, "predicate": fact["predicate"], "roles": sorted(fact["roles"]),
                             "qualifiers": fact["qualifiers"]})
            continue
        memory.add(fact)
    if unmapped:
        raise ValueError(f"{len(unmapped)} occurrences lack a compatible supplied schema; "
                         "nothing was exported. Example: " + json.dumps(unmapped[0], ensure_ascii=False))
    memory._refresh_source_store()
    counts = {name: sum(s == name for s in statuses.values()) for name in ("done", "partial", "failed")}
    counts["pending"] = len(docs) - len(statuses)
    return {**memory.snapshot(), "extraction": {**provenance, "status": "complete" if complete else "incomplete",
                                                "counts": counts, "corpus_documents": len(docs),
                                                "occurrences": len(records), "semantic_truth_verified": False},
            "hierarchy_sha256": digest(hierarchy)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="SQLite state from components.hyperedge_extraction")
    parser.add_argument("--hierarchy", type=Path, required=True, help="JSON with supplied schemas, concepts and optional aliases")
    parser.add_argument("--output", type=Path, required=True, help="hypergraph JSON for python -m hedge --index")
    parser.add_argument("--allow-incomplete", action="store_true", help="export available candidates with explicit incomplete metadata")
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.documents, args.state, args.hierarchy)}:
        parser.error("index output must not overwrite an input")
    documents = load_documents(args.documents)
    hierarchy = json.loads(args.hierarchy.read_text(encoding="utf-8"))
    result = build_index(documents, args.state, hierarchy, allow_incomplete=args.allow_incomplete)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "occurrences": len(result["occurrences"]),
                      "extraction_status": result["extraction"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

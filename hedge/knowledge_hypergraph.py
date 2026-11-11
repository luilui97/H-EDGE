"""Hierarchical Knowledge Hypergraph (paper Section III).

L0 holds documents/blocks, L1 occurrences and compositions, L2 relation
schemas, and L3 domain concepts. Semantic level is independent of depth.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from components.source_qa.store import EvidenceStore, norm


def identity(kind, value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return kind + ":" + hashlib.sha256(payload).hexdigest()[:24]


def entails(actual, required):
    return all(key in actual and norm(str(actual[key])) == norm(str(value))
               for key, value in required.items())


def compatible(fact, goal, answer, alignment, same_value=None):
    """Eq. (role match): injective role alignment, bound values and qualifiers.

    Predicate-family semantics are checked separately by the source verifier;
    the declared role correspondence must still pass these concrete checks.
    """
    roles = list(goal["inputs"]) + [goal["output_role"]]
    if set(alignment) != set(roles) or len(set(alignment.values())) != len(roles):
        return False
    values = fact["roles"]
    same_value = same_value or (lambda a, b: norm(a) == norm(b))
    wanted = {**goal["inputs"], goal["output_role"]: answer}
    return (all(alignment[r] in values and same_value(values[alignment[r]], v)
                for r, v in wanted.items())
            and entails(fact.get("qualifiers", {}), goal["qualifiers"]))


class HierarchicalKnowledgeHypergraph:
    def __init__(self, documents, index):
        self.documents = {int(d.get("doc_idx", i)): copy.deepcopy(d)
                          for i, d in enumerate(documents)}
        if len(self.documents) != len(documents):
            raise ValueError("duplicate document IDs")
        self.version = int(index.get("version", 0))
        self.schemas = self._keyed(index["schemas"], "schema")
        self.schemas_by_predicate = defaultdict(list)
        for sid, schema in self.schemas.items():
            for predicate in schema["predicates"]:
                self.schemas_by_predicate[norm(predicate)].append(sid)
        self.concepts = self._keyed(index["concepts"], "concept")
        self.aliases = copy.deepcopy(index.get("aliases", {}))
        self._alias_lookup = defaultdict(set)
        for alias, ids in self.aliases.items():
            self._alias_lookup[norm(alias)].update(ids)
        self.title_documents = defaultdict(list)
        for did, doc in self.documents.items():
            self.title_documents[norm(doc["title"])].append(did)
        self.anchors = copy.deepcopy(index.get("anchors", []))
        self.facts = {}
        self.memberships = {sid: [] for sid in self.schemas}
        self.blocks = copy.deepcopy(index.get("blocks", []))
        for concept in self.concepts.values():
            if not concept["schemas"] or not set(concept["schemas"]) <= self.schemas.keys():
                raise ValueError("concept must reference existing relation schemas")
        covered = {sid for c in self.concepts.values() for sid in c["schemas"]}
        if covered != set(self.schemas):
            raise ValueError("each schema must belong to a domain concept")
        for schema in self.schemas.values():
            if (not schema["predicates"] or not schema["roles"]
                    or len(schema["roles"]) != len(set(schema["roles"]))):
                raise ValueError("schema requires predicates and distinct named roles")
        # Constituents must precede composites in the serialized index.
        for source in index["occurrences"]:
            fact = copy.deepcopy(source)
            if "sources" not in fact:
                fact["sources"] = [{k: fact[k] for k in ("doc_idx", "start", "end")}]
            fact.setdefault("depth", 0)
            fact.setdefault("qualifiers", {})
            self.add(fact)
        self._refresh_source_store()

    @staticmethod
    def _keyed(rows, kind):
        result = {r["id"]: copy.deepcopy(r) for r in rows}
        if len(result) != len(rows) or any(not key for key in result):
            raise ValueError(f"{kind} IDs must be unique and nonempty")
        return result

    def validate_sources(self, sources):
        if not sources:
            raise ValueError("evidence requires nonempty provenance")
        for span in sources:
            d, start, end = (span[k] for k in ("doc_idx", "start", "end"))
            if (d not in self.documents or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(self.documents[d]["text"])):
                raise ValueError("source offsets outside original document")
            text = self.documents[d]["text"][start:end]
            if not text.strip() or span.get("text", text) != text:
                raise ValueError("source quote differs from original text")

    def schema_compatible(self, fact, sid):
        schema = self.schemas[sid]
        role_map = fact.get("schema_alignments", {}).get(sid, {r: r for r in schema["roles"]})
        return (norm(fact["predicate"]) in {norm(p) for p in schema["predicates"]}
                and set(role_map) == set(schema["roles"])
                and len(set(role_map.values())) == len(role_map)
                and all(r in fact["roles"] for r in role_map.values())
                and entails(fact["qualifiers"], schema.get("qualifiers", {}))
                and all(not allowed or fact["roles"][role_map[r]] in allowed
                        for r, allowed in schema.get("role_values", {}).items()))

    def schema_ids(self, fact):
        return [sid for sid in self.schemas_by_predicate.get(norm(fact["predicate"]), [])
                if self.schema_compatible(fact, sid)]

    def validate_fact(self, fact, *, require_schema=True):
        if (not fact.get("id") or not isinstance(fact.get("predicate"), str)
                or not fact["predicate"].strip() or not fact.get("roles")
                or any(not isinstance(v, str) or not v.strip() for v in fact["roles"].values())):
            raise ValueError("occurrence requires identity, predicate and named role values")
        self.validate_sources(fact["sources"])
        if fact.get("members"):
            if fact["id"] in fact["members"] or any(m not in self.facts for m in fact["members"]):
                raise ValueError("composite has missing/cyclic constituents")
            members = [self.facts[m] for m in fact["members"]]
            if fact["depth"] != 1 + max(m["depth"] for m in members):
                raise ValueError("composition depth disagrees with constituents")
            expected = {(s["doc_idx"], s["start"], s["end"]) for m in members for s in m["sources"]}
            actual = {(s["doc_idx"], s["start"], s["end"]) for s in fact["sources"]}
            if expected != actual:
                raise ValueError("composite provenance is not closed")
            for join in fact.get("joins", []):
                left, right = (self.facts[join[k]] for k in ("left", "right"))
                if (left["id"] not in fact["members"] or right["id"] not in fact["members"]
                        or not self.same_value(left["roles"][join["left_role"]], join["value"])
                        or not self.same_value(right["roles"][join["right_role"]], join["value"])):
                    raise ValueError("composite join does not preserve shared arguments")
        elif fact.get("depth", 0) != 0:
            raise ValueError("atomic occurrence must have depth zero")
        if require_schema and not self.schema_ids(fact):
            raise ValueError("occurrence has no compatible receiving schema")

    def add(self, fact):
        self.validate_fact(fact)
        if fact["id"] in self.facts:
            if self.facts[fact["id"]] != fact:
                raise ValueError("occurrence identity collision")
            return
        self.facts[fact["id"]] = copy.deepcopy(fact)
        for sid in self.schema_ids(fact):
            self.memberships[sid].append(fact["id"])

    def _refresh_source_store(self):
        # Bridge and source-window components operate on atomic occurrences.
        atomic = []
        blocks = [{k: b[k] for k in ("doc_idx", "start", "end")} for b in self.blocks]
        by_document = defaultdict(list)
        for block in blocks:
            self.validate_sources([block])
            block["leaves"] = []
            by_document[block["doc_idx"]].append(block)
        for fact in self.facts.values():
            if fact["depth"]:
                continue
            for span in fact["sources"]:
                fid = len(atomic)
                atomic.append({**fact, **{k: span[k] for k in ("doc_idx", "start", "end")}})
                containing = [b for b in by_document[span["doc_idx"]]
                              if b["start"] <= span["start"] < span["end"] <= b["end"]]
                if not containing:
                    block = {**{k: span[k] for k in ("doc_idx", "start", "end")}, "leaves": []}
                    blocks.append(block)
                    by_document[span["doc_idx"]].append(block)
                    containing = [block]
                for block in containing:
                    block["leaves"].append(fid)
        # Every occurrence has a containing source block (M1), even for indices
        # whose extractor did not provide a coarser local block partition.
        for bid, block in enumerate(blocks):
            block["id"] = bid
        self.blocks = blocks
        self.source_store = EvidenceStore(list(self.documents.values()),
                                         {"occurrences": atomic, "blocks": blocks,
                                          "aliases": self.aliases})
        self._alias_lookup = self.source_store.alias_lookup
        self.argument_members = defaultdict(set)
        for fid, fact in self.facts.items():
            for value in fact["roles"].values():
                self.argument_members[self.value_key(value)].add(fid)

    def value_key(self, value):
        ids = self._alias_lookup.get(norm(value), [])
        return ("entity", next(iter(ids))) if len(ids) == 1 else ("surface", norm(value))

    def same_value(self, left, right):
        return norm(left) == norm(right) or self.value_key(left) == self.value_key(right)

    def anchored_occurrences(self, inputs):
        sets = [self.argument_members.get(self.value_key(v), set()) for v in inputs.values()]
        return set.intersection(*sets) if sets else set(self.facts)

    def snapshot(self):
        return copy.deepcopy({"format": "hedge-hypergraph-1", "version": self.version,
                              "occurrences": list(self.facts.values()),
                              "schemas": list(self.schemas.values()),
                              "concepts": list(self.concepts.values()), "blocks": self.blocks,
                              "aliases": self.aliases, "anchors": self.anchors})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)


class VisitBudget:
    """Shared per-question B; a node inspection is charged before it occurs."""
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("visit budget must be a positive integer")
        self.limit, self.used = limit, 0
        self.counts = Counter()

    def take(self, kind):
        if self.used >= self.limit:
            return False
        self.used += 1
        self.counts[kind] += 1
        return True

    @property
    def remaining(self):
        return self.limit - self.used

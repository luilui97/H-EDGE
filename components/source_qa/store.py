"""Source-corpus adapter used by bridge routing and source-window lookup.

Stored facts are candidates. Their source spans are checked here, but
semantic correctness is deliberately not inferred from a valid span.
"""

from __future__ import annotations

from collections import defaultdict
import re



def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


class EvidenceStore:
    def __init__(self, documents: list[dict], index: dict):
        self.documents = {int(d["doc_idx"]): d for d in documents}
        self.index = index
        self.facts = index["occurrences"]
        self.blocks = index["blocks"]
        self.by_doc: dict[int, list[int]] = defaultdict(list)
        self.by_entity: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.aliases = index.get("aliases", {})
        alias_lookup = defaultdict(set)
        for alias, ids in self.aliases.items():
            if norm(alias):
                alias_lookup[norm(alias)].update(ids)
        self.alias_lookup = {a: sorted(ids) for a, ids in alias_lookup.items()}
        self.max_alias_words = max((len(a.split()) for a in self.alias_lookup), default=1)
        for block in self.blocks:
            self.by_doc[int(block["doc_idx"])].append(int(block["id"]))
        for eid, postings in index.get("entity_blocks", {}).items():
            self.by_entity[eid] = [(int(b), float(w)) for b, w in postings]
        # A confirmed relation port can name an entity whose own document has
        # no confirmed port in its extracted facts. A unique title alias gives
        # that entity a source-document landing page without asserting any new
        # relation between the two documents.
        for doc_idx, doc in self.documents.items():
            ids = self.alias_lookup.get(norm(doc["title"]), [])
            if len(ids) != 1:
                continue
            eid = ids[0]
            existing = {bid for bid, _ in self.by_entity[eid]}
            for bid in self.by_doc.get(doc_idx, []):
                if bid not in existing:
                    self.by_entity[eid].append((bid, 1.0))
        self._valid_spans: dict[int, bool] = {}
        self._block_terms: dict[int, set[str]] = {}
        self._fact_terms: dict[int, set[str]] = {}


    def quote(self, fid: int) -> str:
        f = self.facts[fid]
        d = self.documents[int(f["doc_idx"])]
        return d["text"][int(f["start"]):int(f["end"])]

    def valid_span(self, fid: int) -> bool:
        if fid not in self._valid_spans:
            f = self.facts[fid]
            quote = self.quote(fid)
            self._valid_spans[fid] = bool(quote.strip()) and 0 <= f["start"] < f["end"] <= len(self.documents[int(f["doc_idx"])]["text"])
        return self._valid_spans[fid]

    def query_entities(self, query: str) -> list[str]:
        words = norm(query).split()
        matches = []
        for i in range(len(words)):
            for length in range(1, min(self.max_alias_words, len(words) - i) + 1):
                phrase = " ".join(words[i:i + length])
                if len(phrase) >= 3:
                    matches.extend(self.alias_lookup.get(phrase, []))
        return sorted(set(matches))

    def block_terms(self, bid: int) -> set[str]:
        if bid not in self._block_terms:
            b = self.blocks[bid]
            d = self.documents[int(b["doc_idx"])]
            self._block_terms[bid] = terms(d["text"][b["start"]:b["end"]])
        return self._block_terms[bid]

    def fact_terms(self, fid: int) -> set[str]:
        if fid not in self._fact_terms:
            f = self.facts[fid]
            self._fact_terms[fid] = terms(self.quote(fid) + " " + f.get("predicate", ""))
        return self._fact_terms[fid]

    def entity_names(self, fact: dict) -> set[str]:
        return {norm(p["surface"]) for p in fact.get("ports", []) if p.get("surface")}

    def confirmed_entity_ids(self, fact: dict) -> set[str]:
        return {a["id"] for p in fact.get("ports", []) if p.get("identity_confirmed")
                for a in p.get("alternatives", []) if a["id"].startswith("entity:")}

"""Source-bound title landing and relation-aware bridge selection.

Unique entity-title aliases, confirmed fact ports, and exact source mentions
create candidate routes. A route is not a verified answer.
"""

from __future__ import annotations

from collections import defaultdict
import re

from .store import EvidenceStore, norm, terms


STOP = {"by", "of", "the", "a", "an", "is", "was", "were", "has", "had", "in", "to", "and"}
ROOTS = {
    "director": "direct", "directors": "direct", "directed": "direct", "directing": "direct",
    "composer": "music", "composers": "music", "composed": "music", "composition": "music",
    "song": "music", "songs": "music", "musical": "music",
    "author": "write", "authors": "write", "writer": "write", "writers": "write",
    "wrote": "write", "written": "write", "writing": "write",
    "husband": "spouse", "wife": "spouse", "married": "spouse", "marriage": "spouse",
    "wed": "spouse", "spouses": "spouse",
    "father": "parent", "mother": "parent", "parents": "parent",
    "son": "parent", "daughter": "parent", "children": "parent", "child": "parent",
    "paternal": "parent", "maternal": "parent", "grandmother": "parent", "grandfather": "parent",
    "uncle": "parent", "aunt": "parent", "nephew": "parent", "niece": "parent",
    "birth": "born", "birthplace": "born", "death": "died", "dies": "died",
    "nationality": "country", "nationalities": "country", "countries": "country",
    "performer": "perform", "performed": "perform", "singer": "perform",
    "singers": "perform", "artist": "perform", "artists": "perform",
    "recorded": "perform", "recording": "perform", "actor": "perform",
    "actress": "perform", "appeared": "perform", "starred": "perform",
    "burial": "buried", "interred": "buried",
    "producer": "produce", "producers": "produce", "produced": "produce",
    "published": "publish", "publisher": "publish", "publishers": "publish",
    "studied": "study", "studying": "study", "education": "study",
    "degree": "study", "degrees": "study", "major": "study", "majored": "study",
    "founder": "found", "founded": "found", "founding": "found",
    "cofounder": "found", "cofounded": "found",
}
RELATIONS = {"direct", "music", "write", "spouse", "parent", "born", "died",
             "country", "perform", "buried", "founded", "founder", "located",
             "produce", "publish", "study", "found"}


def relation_terms(text: str) -> set[str]:
    return {root for t in terms(text) if t not in STOP and len(t) > 2
            if (root := ROOTS.get(t, t)) in RELATIONS}


class BridgeSelector:
    def __init__(self, store: EvidenceStore):
        self.store = store
        self.titles = {i: norm(d["title"]) for i, d in store.documents.items()}
        landing: dict[str, set[int]] = defaultdict(set)
        title_docs: dict[str, set[int]] = defaultdict(set)
        for doc, title in self.titles.items():
            title_docs[title].add(doc)
            ids = store.alias_lookup.get(title, [])
            if len(ids) == 1:
                landing[ids[0]].add(doc)
        self.alias_to_docs: dict[str, set[int]] = {}
        for alias, ids in store.alias_lookup.items():
            if len(ids) == 1 and len(alias) >= 2 and ids[0] in landing:
                plausible = {doc for doc in landing[ids[0]] if
                             len((set(alias.split()) - STOP) &
                                 (set(self.titles[doc].split()) - STOP)) /
                             max(1, len(set(self.titles[doc].split()) - STOP)) >= 0.5}
                if len(plausible) == 1:
                    self.alias_to_docs[alias] = plausible
        for title, docs in title_docs.items():
            if len(docs) == 1 and len(title) >= 2:
                self.alias_to_docs.setdefault(title, docs)
        self.unique_titles = {t for t, docs in title_docs.items() if len(docs) == 1}
        self.max_alias_words = max((len(a.split()) for a in self.alias_to_docs), default=2)
        self.raw_cache: dict[int, list[dict]] = {}
        self.links: dict[int, list[dict]] = defaultdict(list)
        for fid, fact in enumerate(store.facts):
            if not fact.get("roles") or not store.valid_span(fid):
                continue
            source = int(fact["doc_idx"])
            relation = relation_terms(str(fact.get("predicate", "")))
            if not relation:
                relation = relation_terms(store.quote(fid))
            if not relation:
                continue
            for port in fact.get("ports", []):
                targets = set()
                if port.get("identity_confirmed"):
                    for alt in port.get("alternatives", []):
                        if str(alt.get("id", "")).startswith("entity:"):
                            targets.update(landing.get(alt["id"], ()))
                else:
                    surface = norm(str(port.get("surface", "")))
                    quote = norm(store.quote(fid))
                    if surface and f" {surface} " in f" {quote} ":
                        words = surface.split()
                        for start in range(len(words)):
                            for length in range(1, min(self.max_alias_words, len(words) - start) + 1):
                                targets.update(self.alias_to_docs.get(" ".join(words[start:start + length]), ()))
                for target in targets:
                    if target != source:
                        self.links[source].append({"target": target, "fid": fid,
                                                   "relation": relation,
                                                   "port_role": port.get("role", ""),
                                                   "identity_route": "confirmed" if port.get("identity_confirmed")
                                                   else "source_surface_alias"})

    def source_mentions(self, source: int) -> list[dict]:
        """Route through title aliases appearing near a relation in original text."""
        if source in self.raw_cache:
            return self.raw_cache[source]
        text = self.store.documents[source]["text"]
        # Offsets refer to ORIGINAL text: case folding can change its length.
        spans = list(re.finditer(r"\w+", text))
        words = [m.group().casefold() for m in spans]
        found: dict[tuple[int, int], dict] = {}
        for start in range(len(words)):
            for length in range(1, min(self.max_alias_words, len(words) - start) + 1):
                alias = " ".join(words[start:start + length])
                if alias not in self.alias_to_docs:
                    continue
                context = " ".join(words[max(0, start - 10):min(len(words), start + length + 10)])
                relation = relation_terms(context)
                if not relation:
                    continue
                for target in self.alias_to_docs[alias]:
                    if target == source:
                        continue
                    key = (target, start)
                    context_start = max(0, start - 10)
                    context_end = min(len(words), start + length + 10)
                    found[key] = {"target": target, "fid": None,
                                  "relation": relation, "port_role": "mention",
                                  "identity_route": "source_mention",
                                  "start": spans[context_start].start(),
                                  "end": spans[context_end - 1].end(),
                                  "mention_start": spans[start].start(),
                                  "mention_end": spans[start + length - 1].end()}
        self.raw_cache[source] = list(found.values())
        return self.raw_cache[source]

    def select(self, question: str, ranked_docs: list[int], *, top_k: int = 5,
               max_anchors: int = 3, implicit_anchors: bool = True,
               fact_ports: bool = True, source_mentions: bool = True,
               budget=None) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        position = {doc: i for i, doc in enumerate(ranked_docs)}
        qnorm = f" {norm(question)} "
        qrel = relation_terms(question)
        if " in law " in qnorm:
            qrel.add("spouse")
        anchors = [doc for doc, title in self.titles.items()
                   if title in self.unique_titles and len(title) >= 2 and
                   position.get(doc, len(ranked_docs)) < 30 and f" {title} " in qnorm]
        anchors.sort(key=lambda d: (position.get(d, len(ranked_docs)), -len(self.titles[d]), d))
        # Several pages can share a case-folded title. Keep the strongest dense
        # page so a duplicate title cannot consume another Reader slot.
        unique = []
        seen_titles = set()
        for doc in anchors:
            title = self.titles[doc]
            if title not in seen_titles:
                unique.append(doc)
                seen_titles.add(title)
        anchors = [doc for doc in unique if not any(
            doc != other and self.titles[doc] != self.titles[other] and
            f" {self.titles[doc]} " in f" {self.titles[other]} "
            for other in unique)][:min(max_anchors, top_k)]
        if not anchors and implicit_anchors and qrel:
            anchors = ranked_docs[:min(2, top_k)]
        if not anchors:
            return {"indices": ranked_docs[:top_k], "anchors": [],
                    "bridges": [], "candidate_ids": ranked_docs[:30],
                    "mode": "source_bridge", "semantic_verified": False,
                    "work": {"blocks": 0, "facts": 0, "links": 0,
                             "source_docs_scanned": 0}}

        # The established top two dense documents are a conservative rail for
        # tasks where exact title mentions are contextual, not support pages.
        selected = list(ranked_docs[:min(2, top_k)])
        for doc in anchors:
            if doc not in selected and len(selected) < top_k:
                selected.append(doc)
        bridge_options: list[tuple[float, int, int, dict]] = []
        fact_links_examined = 0
        raw_links_examined = 0
        for source in anchors:
            fact_links = self.links.get(source, []) if fact_ports else []
            raw_links = self.source_mentions(source) if source_mentions else []
            best: dict[int, tuple[float, dict]] = {}
            for link in fact_links + raw_links:
                if budget is not None and not budget.take("fact_port" if link["fid"] is not None else "source_mention"):
                    break
                fact_links_examined += int(link["fid"] is not None)
                raw_links_examined += int(link["fid"] is None)
                matched = len(qrel & link["relation"])
                if not matched:
                    continue
                target = link["target"]
                quality = {"confirmed": 0.15, "source_surface_alias": 0.10,
                           "source_mention": 0.0}[link["identity_route"]]
                score = matched + quality + 0.05 / (1 + position.get(target, len(ranked_docs)))
                if target not in best or score > best[target][0]:
                    best[target] = (score, link)
            for target, (score, link) in best.items():
                bridge_options.append((score, source, target, link))
        # One bridge per anchor first: comparison questions need both branches.
        used_sources = set()
        chosen = []
        for score, source, target, link in sorted(bridge_options,
                                                 key=lambda x: (-x[0], position.get(x[1], len(ranked_docs)),
                                                                position.get(x[2], len(ranked_docs)), x[2])):
            if source not in used_sources and target not in selected and len(selected) < top_k:
                selected.append(target)
                chosen.append({"source": source, "target": target,
                               "fid": link["fid"], "relation": sorted(link["relation"]),
                               "identity_route": link["identity_route"],
                               "source_start": link.get("start"), "source_end": link.get("end"),
                               "mention_start": link.get("mention_start"),
                               "mention_end": link.get("mention_end"),
                               "relation_score": score})
                used_sources.add(source)
        # A single anchor may need two distinct relation-bearing documents.
        for score, source, target, link in sorted(bridge_options,
                                                 key=lambda x: (-x[0], position.get(x[2], len(ranked_docs)), x[2])):
            if len(selected) >= top_k:
                break
            if target not in selected:
                selected.append(target)
                chosen.append({"source": source, "target": target,
                               "fid": link["fid"], "relation": sorted(link["relation"]),
                               "identity_route": link["identity_route"],
                               "source_start": link.get("start"), "source_end": link.get("end"),
                               "mention_start": link.get("mention_start"),
                               "mention_end": link.get("mention_end"),
                               "relation_score": score})
        for doc in ranked_docs:
            if len(selected) >= top_k:
                break
            if doc not in selected:
                selected.append(doc)
        # When the first dense page is an explicitly named anchor, a candidate
        # source route can occupy the second slot while retaining the dense
        # runner-up in the same five-document packet.
        if len(selected) > 1 and selected[0] in anchors and selected[1] not in anchors:
            primary = next((x for x in chosen if x["source"] == selected[0]), None)
            if primary and primary["target"] in selected:
                selected.remove(primary["target"])
                selected.insert(1, primary["target"])
        pool = set(ranked_docs[:30]) | set(anchors) | {x[2] for x in bridge_options}
        return {"indices": selected, "anchors": anchors, "bridges": chosen,
                "candidate_ids": sorted(pool), "mode": "source_bridge",
                "semantic_verified": False,
                "work": {"blocks": 0, "facts": fact_links_examined,
                         "links": len(bridge_options),
                         "source_docs_scanned": len(anchors),
                         "raw_links_examined": raw_links_examined}}

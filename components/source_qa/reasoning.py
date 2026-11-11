"""Original-source windows and witness-conditioned answer generation."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from .store import EvidenceStore, terms



READER = '''Answer the original question using ONLY the supplied documents and source-verified steps.
Use the steps to resolve named intermediates, then answer the original relation. A complete chain provides a candidate answer for an ordinary answer question, but check that its final step actually answers the ORIGINAL question. A partial chain can still identify an intermediate; continue from the documents. For compare or intersect questions, combine the relevant branches instead of copying the last step's answer. Do not reject an answer merely because the documents express the relation indirectly or use a paraphrase. If the required relation or answer is unsupported, answer Not specified.
Return JSON {"answer":"concise answer or Not specified","reasoning":"brief evidence-grounded explanation"}. Keep the answer field concise and do not add citations there.'''


def contains(haystack: str, needle: str) -> bool:
    def folded(value: str) -> str:
        value = unicodedata.normalize("NFKD", value.casefold())
        value = "".join(c for c in value if not unicodedata.combining(c))
        return " ".join(re.findall(r"\w+", value))

    needle_folded = folded(needle)
    return bool(needle_folded) and f" {needle_folded} " in f" {folded(haystack)} "


def source_spelling(answer: str, blocks: list[dict]) -> str:
    """Restore a source's spelling for an answer differing only in case/accents."""
    def fold_token(value: str) -> str:
        value = unicodedata.normalize("NFKD", value.casefold())
        return "".join(c for c in value if not unicodedata.combining(c))

    wanted = [fold_token(m.group()) for m in re.finditer(r"\w+", answer)]
    for block in blocks:
        text = block["text"]
        matches = list(re.finditer(r"\w+", text))
        for i in range(len(matches) - len(wanted) + 1):
            window = matches[i:i + len(wanted)]
            if [fold_token(m.group()) for m in window] != wanted:
                continue
            surface = text[window[0].start():window[-1].end()]
            if (len(surface) == len(answer) and
                re.sub(r"\w", "", surface) == re.sub(r"\w", "", answer)):
                return surface
    return answer


class SourceCatalog:
    def __init__(self, store: EvidenceStore, doc_ids: list[int], question: str,
                 *, blocks_per_doc: int = 6, max_blocks: int = 30):
        self.store = store
        self.blocks: dict[str, dict] = {}
        qterms = terms(question)
        for doc_idx in dict.fromkeys(doc_ids):
            doc = store.documents[doc_idx]
            text = doc["text"]
            words = list(re.finditer(r"\S+", text))
            options = []
            for offset in range(0, len(words), 48):
                group = words[offset:offset + 60]
                if not group:
                    continue
                a, b = group[0].start(), group[-1].end()
                quote = text[a:b]
                score = len(qterms & terms(quote + " " + doc["title"])) / max(1, len(qterms))
                options.append((score, a, b, quote))
            for _, a, b, quote in sorted(options, key=lambda x: (-x[0], x[1]))[:blocks_per_doc]:
                key = f"d{doc_idx}:{a}:{b}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
                self.blocks[key] = {"id": key, "doc_idx": doc_idx, "title": doc["title"],
                                    "start": a, "end": b, "text": quote}
        if len(self.blocks) > max_blocks:
            self.blocks = dict(list(self.blocks.items())[:max_blocks])

    def payload(self) -> list[dict]:
        return [{k: b[k] for k in ("id", "doc_idx", "title", "text")}
                for b in self.blocks.values()]

    def resolve(self, answer: str, evidence: list[str]) -> list[dict]:
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 200:
            raise ValueError("no bounded answer")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4 or len(evidence) != len(set(evidence)):
            raise ValueError("invalid evidence IDs")
        if any(i not in self.blocks for i in evidence):
            raise ValueError("unknown evidence ID")
        selected = [self.blocks[i] for i in evidence]
        # A source-local answer can have a harmless case/punctuation difference.
        # The exact evidence window remains the source of truth; a citation to
        # the wrong window can only be repaired within the blocks actually shown
        # to the extractor, and the semantic verifier still checks the relation.
        if not any(contains(b["text"], answer) for b in selected):
            match = next((b for b in self.blocks.values()
                          if contains(b["text"], answer)), None)
            if match is None or len(selected) >= 4:
                raise ValueError("answer absent from cited source")
            selected.append(match)
        for b in selected:
            if self.store.documents[b["doc_idx"]]["text"][b["start"]:b["end"]] != b["text"]:
                raise ValueError("source span changed")
        return selected




def read_answer(llm, store: EvidenceStore, question: str, doc_ids: list[int],
                chain: list[dict], *, chain_closed: bool = False,
                operation: str = "answer", witnesses: list[dict] | None = None,
                occurrences: list[dict] | None = None) -> dict:
    content = {"question": question,
               "operation": operation, "chain_closed": chain_closed,
               "documents": [{"doc_idx": i, "title": store.documents[i]["title"],
                              "text": store.documents[i]["text"]} for i in doc_ids],
               "verified_chain": [{"question": s["question"], "answer": s["answer"],
                                   "sources": s["sources"]} for s in chain]}
    if witnesses is not None:
        content["evidence_witnesses"] = witnesses
        content["evidence_occurrences"] = occurrences or []
    try:
        guidance = ""
        if re.match(r"^\s*(?:are|is|was|were|do|does|did|has|have|had|can|could|would|will)\b",
                    question, re.I):
            guidance = "\nThis is a yes/no question. Return Yes or No when the evidence resolves it; do not return the shared entity."
        elif re.search(r"\blived longer\b", question, re.I):
            guidance = "\nCompare lifespans using birth and death dates or stated ages; later death alone is insufficient."
        value = llm.ask("reader", READER + guidance, content, tokens=800)
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty Reader answer")
        surface_sources = [b for step in chain for b in step["sources"]]
        surface_sources.extend({"text": store.documents[i]["text"]} for i in doc_ids)
        answer = source_spelling(answer.strip(), surface_sources)
        return {"answer": answer.strip(), "reasoning": value.get("reasoning", ""),
                "error": None, "indices": doc_ids}
    except Exception as exc:
        return {"answer": "", "reasoning": "", "error": str(exc), "indices": doc_ids}

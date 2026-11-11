"""Offline checks of supplied extraction logic, resume state and index export."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from components.hyperedge_extraction import extract, load_documents, run_extraction, validate_facts
from hedge.build_index import build_index
from hedge import HierarchicalKnowledgeHypergraph


DOC = {"doc_idx": 0, "title": "Ada", "text": "Ada studied physics at North University in 2010."}
FACT = {"predicate": "studied", "roles": {"subject": "Ada", "object": "physics", "institution": "North University"},
        "qualifiers": {"time": "2010"}, "quote": DOC["text"]}
HIERARCHY = {"schemas": [{"id": "study", "predicates": ["studied"], "roles": ["person", "field"],
                          "role_alignment": {"person": "subject", "field": "object"}}],
             "concepts": [{"id": "education", "schemas": ["study"]}]}


def response(facts):
    return {"done": True, "done_reason": "stop", "eval_count": 20,
            "message": {"content": json.dumps({"facts": facts}, ensure_ascii=False)}}


class Gateway:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, base, endpoint, payload=None):
        self.calls.append(copy.deepcopy(payload))
        if not self.replies:
            raise AssertionError("unexpected model request")
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class HyperedgeExtraction(unittest.TestCase):
    def test_exact_spans_roles_and_qualifiers(self):
        bad = copy.deepcopy(FACT)
        bad["qualifiers"] = {"time": "2011"}
        accepted, errors = validate_facts({"facts": [FACT, bad]}, DOC)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["start"], 0)
        self.assertEqual(DOC["text"][accepted[0]["start"]:accepted[0]["end"]], FACT["quote"])
        self.assertEqual(errors[0]["fact_index"], 1)
        bad["qualifiers"] = {}
        bad["quote"] = "Ada studied astronomy."
        self.assertEqual(validate_facts({"facts": [bad]}, DOC)[0], [])

    def test_repeated_quote_requires_occurrence_offset(self):
        doc = {"doc_idx": 0, "title": "Ada", "text": "Ada knows Bo. Ada knows Bo."}
        fact = {"predicate": "knows", "roles": {"subject": "Ada", "object": "Bo"}, "quote": "Ada knows Bo."}
        self.assertEqual(validate_facts({"facts": [fact]}, doc)[0], [])
        facts, errors = validate_facts({"facts": [{**fact, "start": 14}]}, doc)
        self.assertEqual(errors, [])
        self.assertEqual(facts[0]["start"], 14)

    def test_truncation_retry_and_identity_checked_cache(self):
        gateway = Gateway([{"done": True, "done_reason": "length"}, response([FACT])])
        with tempfile.TemporaryDirectory() as tmp:
            first = extract("unused", "model", "digest-A", DOC, tmp, requester=gateway)
            self.assertEqual([p["options"]["num_predict"] for p in gateway.calls], [2400, 4000])
            again = extract("unused", "model", "digest-A", DOC, tmp, requester=gateway)
            self.assertEqual(first, again)
            self.assertEqual(len(gateway.calls), 2)
            other = Gateway([response([FACT])])
            changed = extract("unused", "model", "digest-B", DOC, tmp, requester=other)
            self.assertNotEqual(changed[2], first[2])

    def test_incomplete_or_oversized_model_response_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = Gateway([{"done": False}, response([FACT] * 9)])
            with self.assertRaisesRegex(ValueError, "malformed"):
                extract("unused", "model", "digest", DOC, tmp, requester=gateway)

    def test_resume_partial_retries_and_replaces_candidates(self):
        bad = copy.deepcopy(FACT)
        bad["roles"]["object"] = "not in the source"
        with tempfile.TemporaryDirectory() as tmp:
            state, cache = Path(tmp) / "state.sqlite", Path(tmp) / "cache"
            partial = run_extraction([DOC], state, cache, "unused", "model", "digest",
                                     requester=Gateway([response([FACT, bad])]))
            self.assertEqual(partial["counts"]["partial"], 1)
            self.assertEqual(partial["status"], "incomplete")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                build_index([DOC], state, HIERARCHY)
            interim = build_index([DOC], state, HIERARCHY, allow_incomplete=True)
            self.assertEqual(interim["extraction"]["status"], "incomplete")
            retry = Gateway([response([FACT, FACT])])
            completed = run_extraction([DOC], state, cache, "unused", "model", "digest", requester=retry)
            self.assertEqual(len(retry.calls), 1)  # rejected cached response was refreshed
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["occurrences"], 1)
            no_calls = Gateway([])
            resumed = run_extraction([DOC], state, cache, "unused", "model", "digest", requester=no_calls)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(no_calls.calls, [])

    def test_failure_remains_retryable_and_empty_list_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, cache = Path(tmp) / "state.sqlite", Path(tmp) / "cache"
            failed = run_extraction([DOC], state, cache, "unused", "model", "digest",
                                    requester=Gateway([TimeoutError("test timeout")]))
            self.assertEqual(failed["counts"]["failed"], 1)
            done = run_extraction([DOC], state, cache, "unused", "model", "digest", requester=Gateway([response([])]))
            self.assertEqual(done["status"], "complete")
            self.assertEqual(done["occurrences"], 0)
            self.assertFalse(done["semantic_truth_verified"])

    def test_export_loads_with_schema_alignment_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, cache = Path(tmp) / "state.sqlite", Path(tmp) / "cache"
            run_extraction([DOC], state, cache, "unused", "model", "digest", requester=Gateway([response([FACT])]))
            index = build_index([DOC], state, HIERARCHY)
            memory = HierarchicalKnowledgeHypergraph([DOC], index)
            self.assertEqual(len(memory.facts), 1)
            fact = next(iter(memory.facts.values()))
            self.assertEqual(memory.memberships["study"], [fact["id"]])
            self.assertEqual(fact["schema_alignments"]["study"], {"person": "subject", "field": "object"})
            self.assertEqual(fact["qualifiers"], {"time": "2010"})
            self.assertFalse(fact["semantic_verified"])
            self.assertTrue(all(not p["identity_confirmed"] for p in fact["ports"]))
            for mention in fact["mentions"]:
                self.assertEqual(DOC[mention["field"]][mention["start"]:mention["end"]], mention["surface"])
            with self.assertRaisesRegex(ValueError, "compatible supplied schema"):
                build_index([DOC], state, {"schemas": [], "concepts": []})

    def test_title_role_and_unicode_offsets(self):
        doc = {"doc_idx": 0, "title": "李明", "text": "于2010年就读于北京大学。"}
        fact = {"predicate": "就读", "roles": {"subject": "李明", "object": "北京大学"}, "quote": doc["text"]}
        hierarchy = {"schemas": [{"id": "study", "predicates": ["就读"], "roles": ["subject", "object"]}],
                     "concepts": [{"id": "education", "schemas": ["study"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "中文.sqlite"
            run_extraction([doc], state, Path(tmp) / "cache", "unused", "model", "digest", requester=Gateway([response([fact])]))
            exported = build_index([doc], state, hierarchy)["occurrences"][0]
            self.assertEqual(exported["mentions"][0]["field"], "title")
            self.assertEqual(exported["sources"][0]["end"], len(doc["text"]))

    def test_resume_identity_and_corpus_export_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, cache = Path(tmp) / "state.sqlite", Path(tmp) / "cache"
            run_extraction([DOC], state, cache, "unused", "model", "digest", requester=Gateway([response([FACT])]))
            with self.assertRaisesRegex(ValueError, "another corpus, model"):
                run_extraction([DOC], state, cache, "unused", "model", "different", requester=Gateway([]))
            changed = {**DOC, "text": DOC["text"] + " More text."}
            with self.assertRaisesRegex(ValueError, "source corpus"):
                build_index([changed], state, HIERARCHY)

    def test_loader_whitelists_source_fields_and_limit_keeps_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(json.dumps([{**DOC, "answer": "hidden label", "support": [5]},
                                        {**DOC, "doc_idx": 1}]), encoding="utf-8")
            docs = load_documents(path)
            self.assertEqual(set(docs[0]), {"doc_idx", "title", "text"})
            summary = run_extraction(docs, Path(tmp) / "state.sqlite", Path(tmp) / "cache",
                                     "unused", "model", "digest", limit=1, requester=Gateway([response([FACT])]))
            self.assertEqual(summary["counts"]["pending"], 1)


if __name__ == "__main__":
    unittest.main()

"""Offline engineering checks with synthetic data; not benchmark evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hedge import HEDGE, Settings, HierarchicalKnowledgeHypergraph, SourceBridge
from hedge.evidence_witnesses import reader_packet
from hedge.knowledge_hypergraph import VisitBudget, compatible
from hedge.hierarchical_retrieval import HierarchicalRetrieval
from components.source_qa.store import EvidenceStore
from hedge.embeddings import NVEmbedRetrieval, texts_hash
import numpy as np


QUESTION = "What was the master's-degree field of Google's co-founder other than Larry Page?"
PLAN = {"requirements": [
    {"id": "r0", "text": "Who co-founded Google with Larry Page?", "depends_on": []},
    {"id": "r1", "text": "What was {{r0.answer}}'s master's-degree field?", "depends_on": ["r0"]}],
    "operation": "answer"}
SPECS = [
    {"predicate": "cofound", "domain": "employment", "inputs": {"organization": "Google", "partner": "Larry Page"},
     "output_role": "person", "qualifiers": {}},
    {"predicate": "major_in", "domain": "education", "inputs": {"person": "{{r0.answer}}"},
     "output_role": "field", "qualifiers": {"degree": "master's"}}]


def fixture(*, missing=False):
    docs = [{"doc_idx": 0, "title": "Google", "text": "Sergey Brin co-founded Google with Larry Page."},
            {"doc_idx": 1, "title": "Sergey Brin", "text": "Sergey Brin earned a master's degree in Computer Science at Stanford University."}]
    docs += [{"doc_idx": i, "title": f"Source {i}", "text": f"Unrelated material {i}."} for i in range(2, 7)]
    facts = [
        {"id": "f0", "predicate": "cofound", "roles": {"person": "Sergey Brin", "organization": "Google", "partner": "Larry Page"},
         "qualifiers": {}, "doc_idx": 0, "start": 0, "end": len(docs[0]["text"])},
        {"id": "f1", "predicate": "study_at", "roles": {"person": "Sergey Brin", "field": "Computer Science", "institution": "Stanford University"},
         "qualifiers": {"degree": "master's"}, "doc_idx": 1, "start": 0, "end": len(docs[1]["text"])}]
    index = {"occurrences": facts[:1] if missing else facts,
             "schemas": [{"id": "founding", "predicates": ["cofound"], "roles": ["organization", "partner", "person"]},
                         {"id": "degree", "predicates": ["study_at", "major_in"], "roles": ["person", "field"]}],
             "concepts": [{"id": "employment", "schemas": ["founding"]},
                          {"id": "education", "schemas": ["degree"]}]}
    return HierarchicalKnowledgeHypergraph(docs, index)


class Dense:
    def __init__(self, memory):
        self.memory = memory
        self.queries = []

    def scores(self, text):
        self.queries.append(text)
        return {i: 10.0 - i for i in self.memory.documents}


class ScriptedLLM:
    """Deterministic protocol double: assertions target execution, not accuracy."""
    def __init__(self, memory, reject=False, wrong_roles=False):
        self.memory, self.reject, self.wrong_roles = memory, reject, wrong_roles
        self.calls = []
        self.reader = None
        self.binding_count = 0

    def ask(self, kind, system, content, **kwargs):
        self.calls.append(kind)
        if kind in {"plan", "plan_review", "plan_repair"}:
            return copy.deepcopy(PLAN)
        if kind == "evidence_slot":
            return copy.deepcopy(SPECS[int(content["goal"]["id"][1:])])
        if kind == "binding_proposal":
            self.binding_count += 1
            goal = content["goal"]
            i = int(goal["id"][1:])
            answer = ["Sergey Brin", "Computer Science"][i]
            candidate = next((f for f in content["candidates"] if f["id"] == f"f{i}"), None)
            block = next(b for b in content["blocks"] if b["doc_idx"] == i and answer in b["text"])
            roles = {**goal["inputs"], goal["output_role"]: answer}
            alignment = {r: r for r in roles}
            if self.wrong_roles and i == 0:
                alignment["person"], alignment["partner"] = "partner", "person"
            return {"answer": answer, "candidate_id": candidate["id"] if candidate else None,
                    "evidence": [block["id"]], "predicate": goal["predicate"], "roles": roles,
                    "qualifiers": goal["qualifiers"], "alignment": alignment}
        if kind == "binding_verification":
            return {"supported": not self.reject, "binding_consistent": True, "predicate_compatible": True}
        if kind == "reader":
            assert self.memory.version == 0, "current question saw updated memory"
            assert "repair_admission" not in self.calls, "repair ran before answering"
            self.reader = copy.deepcopy(content)
            return {"answer": "Computer Science", "reasoning": "synthetic test response"}
        if kind == "repair_admission":
            assert self.reader is not None
            return {"supported": True, "identities_consistent": True, "qualifiers_supported": True}
        if kind == "anchor_admission":
            return {"same_entity": True}
        raise AssertionError(f"unexpected model operation: {kind}")


def fact_scores(query, facts):
    return {f["id"]: 1.0 for f in facts}


class MethodContract(unittest.TestCase):
    def pipeline(self, *, missing=False, reject=False, wrong_roles=False, budget=256):
        memory = fixture(missing=missing)
        llm = ScriptedLLM(memory, reject=reject, wrong_roles=wrong_roles)
        dense = Dense(memory)
        pipeline = HEDGE(memory, llm, dense, fact_scores, Settings(visit_budget=budget))
        return pipeline, llm, dense, pipeline.run(QUESTION)

    def test_verified_dependency_and_recursive_provenance(self):
        pipeline, llm, dense, result = self.pipeline()
        self.assertTrue(result["chain_closed"])
        self.assertIn("Sergey Brin", dense.queries[-1])
        self.assertNotIn("{{", dense.queries[-1])
        self.assertEqual([w["depth"] for w in result["witnesses"]], [1, 2])
        final = result["witnesses"][-1]
        self.assertEqual({s["doc_idx"] for s in final["sources"]}, {0, 1})
        self.assertEqual(final["joins"][0]["value"], "Sergey Brin")
        self.assertEqual(result["memory_version_answered"], 0)
        self.assertEqual(result["memory_version_after"], 1)
        self.assertEqual(len(llm.reader["documents"]), 5)
        ids = {f["id"] for f in llm.reader["evidence_occurrences"]}
        self.assertTrue(all(set(w["members"]) <= ids for w in result["witnesses"]))
        self.assertLessEqual(result["visits"]["used"], result["visits"]["limit"])
        self.assertIn(final["id"], pipeline.memory.memberships["degree"])
        goal = result["steps"][-1]["goal"]
        candidates = HierarchicalRetrieval(pipeline.memory, fact_scores).retrieve(goal, VisitBudget(100))["candidates"]
        self.assertIn(final["id"], [f["id"] for f in candidates])

    def test_rejected_binding_does_not_activate_successor_or_repair(self):
        _, llm, dense, result = self.pipeline(reject=True)
        self.assertEqual(llm.binding_count, 1)
        self.assertEqual(result["steps"][1]["error"], "unclosed_parent")
        self.assertEqual(result["witnesses"], [])
        self.assertEqual(result["repair_proposals"], [])
        self.assertEqual(result["memory_version_after"], 0)

    def test_role_direction_is_checked_before_model_acceptance(self):
        _, llm, _, result = self.pipeline(wrong_roles=True)
        self.assertFalse(result["steps"][0]["accepted"])
        self.assertNotIn("binding_verification", llm.calls)

    def test_missing_fact_is_source_checked_and_retrievable_after_update(self):
        pipeline, _, _, result = self.pipeline(missing=True)
        self.assertTrue(result["chain_closed"])
        step = result["steps"][1]
        self.assertEqual(step["fact"]["predicate"], "major_in")
        self.assertIn(step["fact"]["id"], pipeline.memory.memberships["degree"])
        self.assertTrue(any(e["kind"] == "fact" and e["admitted"] for e in result["repair_ledger"]))

    def test_budget_exhaustion_blocks_further_binding(self):
        _, llm, _, result = self.pipeline(budget=1)
        self.assertEqual(result["visits"]["used"], 1)
        self.assertEqual(llm.binding_count, 1)
        self.assertEqual(result["steps"][1]["error"], "visit_budget_exhausted")

    def test_no_partial_provenance_in_reader_packet(self):
        sources = [{"doc_idx": i, "start": 0, "end": 1} for i in range(6)]
        witness = {"id": "w", "depth": 2, "sources": sources}
        docs, witnesses, chain = reader_packet(
            [{"accepted": True, "witness_id": "w"}], [witness], list(range(7)))
        self.assertEqual(docs, list(range(5)))
        self.assertEqual(witnesses, [])
        self.assertEqual(chain, [])

    def test_qualifier_and_injective_alignment(self):
        goal = {**SPECS[1], "inputs": {"person": "Sergey Brin"}}
        fact = fixture().facts["f1"]
        self.assertTrue(compatible(fact, goal, "Computer Science", {"person": "person", "field": "field"}))
        goal["qualifiers"] = {"degree": "doctorate"}
        self.assertFalse(compatible(fact, goal, "Computer Science", {"person": "person", "field": "field"}))
        self.assertFalse(compatible(fact, goal, "Computer Science", {"person": "field", "field": "field"}))

    def test_source_and_composition_corruption_rejected(self):
        memory = fixture()
        snapshot = memory.snapshot()
        snapshot["occurrences"][0]["sources"][0]["end"] = 9999
        with self.assertRaises(ValueError):
            HierarchicalKnowledgeHypergraph(list(memory.documents.values()), snapshot)
        pipeline, _, _, result = self.pipeline()
        damaged = copy.deepcopy(result["witnesses"][-1])
        damaged["sources"] = damaged["sources"][:1]
        with self.assertRaises(ValueError):
            pipeline.memory.validate_fact(damaged)

    def test_roundtrip_and_original_source_immutable(self):
        pipeline, llm, _, result = self.pipeline()
        self.assertEqual(llm.memory.version, 0)
        self.assertEqual(set(llm.memory.facts), {"f0", "f1"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.json"
            pipeline.memory.save(path)
            restored = HierarchicalKnowledgeHypergraph(list(pipeline.memory.documents.values()), json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(restored.snapshot(), pipeline.memory.snapshot())

    def test_five_document_configuration(self):
        with self.assertRaises(ValueError):
            Settings(document_budget=7)

    def test_bridge_equation_and_role_metadata_control(self):
        memory = fixture()
        store = memory.source_store
        bridge = SourceBridge(store)
        # Inject a fixed extracted link to check the documented ranking formula.
        bridge.links[0] = [{"target": 3, "fid": 0, "relation": {"found"},
                            "port_role": "person", "identity_route": "confirmed"}]
        first = bridge.select("Who founded Google?", list(range(7)), source_mentions=False)
        bridge.links[0][0]["port_role"] = "scrambled"
        second = bridge.select("Who founded Google?", list(range(7)), source_mentions=False)
        self.assertEqual(first["indices"], second["indices"])
        self.assertAlmostEqual(first["bridges"][0]["relation_score"], 1 + .15 + .05 / 4)
        self.assertEqual(first["anchors"], [0])
        self.assertEqual(bridge.select("Who founded Google?", list(range(7)), fact_ports=False,
                                       source_mentions=False)["bridges"], [])

    def test_source_mentions_offsets_use_original_unicode_text(self):
        docs = [{"doc_idx": 0, "title": "Alpha", "text": "Straße: Beta University was founded here."},
                {"doc_idx": 1, "title": "Beta University", "text": "An institution."}]
        bridge = SourceBridge(EvidenceStore(docs, {"occurrences": [], "blocks": []}))
        link = bridge.source_mentions(0)[0]
        self.assertEqual(docs[0]["text"][link["mention_start"]:link["mention_end"]], "Beta University")

    def test_many_to_many_schema_membership_and_entity_anchors(self):
        original = fixture()
        index = original.snapshot()
        index["schemas"].append({"id": "degree_alt", "predicates": ["study_at"], "roles": ["person", "field"]})
        index["concepts"][1]["schemas"].append("degree_alt")
        index["aliases"] = {"Sergey Brin": ["entity:brin"], "Brin": ["entity:brin"]}
        index["occurrences"][1]["roles"]["person"] = "Brin"
        memory = HierarchicalKnowledgeHypergraph(list(original.documents.values()), index)
        self.assertEqual(len(memory.facts), 2)
        self.assertIn("f1", memory.memberships["degree_alt"])
        goal = {**SPECS[1], "text": "What did Sergey Brin study?", "inputs": {"person": "Sergey Brin"}}
        candidates = HierarchicalRetrieval(memory, fact_scores).retrieve(goal, VisitBudget(30))["candidates"]
        self.assertEqual([f["id"] for f in candidates], ["f1"])
        self.assertTrue(compatible(memory.facts["f1"], goal, "Computer Science",
                                   {"person": "person", "field": "field"}, memory.same_value))
        index["aliases"]["Brin"].append("entity:other")
        ambiguous = HierarchicalKnowledgeHypergraph(list(original.documents.values()), index)
        self.assertFalse(ambiguous.same_value("Brin", "Sergey Brin"))

    def test_embedding_identity_order_and_similarity(self):
        class Encoder:
            model_id = "nvidia/NV-Embed-v2"
            def encode_query(self, text):
                return [1., 0.]
            def encode_passages(self, texts):
                return [[1., 0.] for _ in texts]

        memory = fixture()
        texts = [d["title"] + "\n" + d["text"] for d in memory.documents.values()]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.save(path / "passages.npy", [[1., 0.], [0., 1.]] + [[-1., 0.]] * 5)
            manifest = {"configuration": {"encoder": Encoder.model_id, "texts_sha256": texts_hash(texts)}}
            (path / "passages.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            dense = NVEmbedRetrieval(memory, path, Encoder())
            self.assertEqual(dense.scores("query")[0], 1.0)
            self.assertEqual(dense.scores("query")[1], 0.0)
            self.assertEqual(dense.score_facts("query", list(memory.facts.values())), {"f0": 1., "f1": 1.})
            memory.documents[0]["text"] += " changed"
            with self.assertRaises(ValueError):
                NVEmbedRetrieval(memory, path, Encoder())


if __name__ == "__main__":
    unittest.main()

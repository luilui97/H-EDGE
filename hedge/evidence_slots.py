"""Dependency-Aware Evidence Slots (paper Section IV-A)."""
from __future__ import annotations

import copy

from components.source_qa.planning import plan, instantiate, PLACEHOLDER
from components.source_qa.plan_refinement import refine_plan_faithful


SPECIFY = """Specify the relational requirement for this evidence question, using ONLY
the original question and its question-only dependency plan. Do not answer it.
Return JSON {"predicate":"relation", "domain":"concept ID or empty",
"inputs":{"named input role":"literal or {{r0.answer}} placeholder"},
"output_role":"requested role", "qualifiers":{"qualifier name":"required value"}}.
Preserve every known input, relation direction, degree/date/negation qualifier.
Use the supplied schema vocabulary where appropriate, without weakening the
question to fit a schema. The output role must be distinct from every input role.
Every declared dependency must occur in an input value with its exact placeholder.
"""


class DependencyAwareEvidenceSlots:
    def __init__(self, llm):
        self.llm = llm

    def construct(self, question, memory):
        planned = refine_plan_faithful(self.llm, question, plan(self.llm, question))
        goals = []
        for original in planned["requirements"]:
            spec = self.llm.ask("evidence_slot", SPECIFY,
                                {"original_question": question, "plan": planned,
                                 "goal": original, "schemas": list(memory.schemas.values()),
                                 "concepts": list(memory.concepts.values())}, tokens=600)
            goal = {**original, **{k: spec[k] for k in
                                  ("predicate", "domain", "inputs", "output_role", "qualifiers")}}
            self.validate(goal)
            goals.append(goal)
        return {**planned, "requirements": goals}

    @staticmethod
    def validate(goal):
        if (not isinstance(goal["inputs"], dict) or not isinstance(goal["qualifiers"], dict)
                or not isinstance(goal["domain"], str)
                or not isinstance(goal["predicate"], str) or not goal["predicate"].strip()
                or not isinstance(goal["output_role"], str) or not goal["output_role"].strip()
                or goal["output_role"] in goal["inputs"]
                or any(not isinstance(v, str) or not v.strip() for v in goal["inputs"].values())):
            raise ValueError("invalid typed evidence slot")
        refs = {ref for value in goal["inputs"].values() for ref in PLACEHOLDER.findall(value)}
        if refs != set(goal["depends_on"]):
            raise ValueError("input roles must expose exactly the declared dependencies")

    @staticmethod
    def ready(goal, accepted):
        return goal["id"] not in accepted and all(d in accepted for d in goal["depends_on"])

    @staticmethod
    def instantiate(goal, accepted):
        result = copy.deepcopy(goal)
        values = {key: value["answer"] for key, value in accepted.items()}
        result["text"] = instantiate(goal["text"], values)
        result["inputs"] = {r: instantiate(v, values) for r, v in goal["inputs"].items()}
        result["qualifiers"] = {r: instantiate(v, values) if isinstance(v, str) else v
                                for r, v in goal["qualifiers"].items()}
        return result

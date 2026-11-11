"""Question-only DAG plan with strict binding syntax."""

from __future__ import annotations

import re


PLACEHOLDER = re.compile(r"\{\{(r[0-3])\.answer\}\}")
COMPARATORS = re.compile(r"\b(first|earlier|later|older|younger|larger|smaller|greater|less|more|longer|shorter|same|different)\b", re.I)
LIFESPAN_PAIR = re.compile(
    r"^(?:who|which [^,?]+) lived longer,\s*(.+?)\s+or\s+(.+?)\s*\?$", re.I)

SYSTEM = '''Return JSON {"requirements":[{"id":"r0","text":"atomic evidence question","depends_on":[]}],"operation":"answer|compare|intersect"}.
Break the supplied question into 1..4 necessary atomic evidence questions. Resolve unknown intermediates before asking about their attributes. A dependent question MUST use {{r0.answer}} etc. and declare that ID in depends_on. Keep relations, comparison branches, dates, negation and qualifiers. For "who lived longer", retrieve each person's lifespan or both birth and death dates; comparing death dates alone is invalid. Do not answer any goal or invent entity values. If an atomic plan is uncertain, return the original question as one goal. IDs must be r0,r1,r2,r3 in order.'''

REPAIR = '''Repair the proposed evidence plan against the ORIGINAL question. Return the same JSON structure.
Every goal must be a complete grammatical evidence question. Every dependency MUST appear verbatim as {{r0.answer}}, {{r1.answer}}, etc. in that goal text; listing a dependency without its placeholder is invalid. Do not substitute a guessed answer or hide an unresolved relation inside a noun phrase. Preserve all entities, qualifiers and comparison branches from the original question. A "lived longer" comparison needs each lifespan or both birth and death dates, not just death dates. If uncertain, output one goal whose text is the exact original question and has no dependencies.'''

REVIEW = '''Review and correct the proposed plan using ONLY the original question. Return the plan JSON, with no explanation.
Resolve every hidden intermediate: "When did X's mother die?" requires first finding X's mother, then her death date; "what is a star of Film known for?" requires identifying an individual star, then finding that person's notable role. A comparison requires the attributes needed for the stated comparison. "Who lived longer?" needs lifespans or birth and death dates of both people; later death alone does not imply longer life. Every goal must be a complete evidence QUESTION ending in ?. For a dependent goal write the literal {{r0.answer}} or {{r1.answer}} token in its text and declare exactly those dependencies. Never guess the intermediate answer. Keep the original relation, direction, names, dates and qualifiers. If the proposed plan changes the question, repair it. If uncertain use one goal with the exact original question.'''

GOAL_SCHEMA = {"type": "object", "properties": {
    "id": {"type": "string", "enum": ["r0", "r1", "r2", "r3"]},
    "text": {"type": "string", "maxLength": 600},
    "depends_on": {"type": "array", "items": {"type": "string", "enum": ["r0", "r1", "r2", "r3"]}}},
    "required": ["id", "text", "depends_on"], "additionalProperties": False}
PLAN_SCHEMA = {"type": "object", "properties": {
    "requirements": {"type": "array", "minItems": 1, "maxItems": 4, "items": GOAL_SCHEMA},
    "operation": {"type": "string", "enum": ["answer", "compare", "intersect"]}},
    "required": ["requirements", "operation"], "additionalProperties": False}


def validate(value: dict, question: str) -> dict:
    goals = value.get("requirements")
    if not isinstance(goals, list) or not 1 <= len(goals) <= 4:
        raise ValueError("plan requires 1..4 goals")
    # Comparison is a computation over source-supported attributes, not a
    # third evidence lookup. Models sometimes append a synthetic final goal.
    if (len(goals) >= 3 and COMPARATORS.search(question) and
        isinstance(goals[-1], dict) and COMPARATORS.search(str(goals[-1].get("text", ""))) and
        set(goals[-1].get("depends_on", [])) == {g.get("id") for g in goals[:-1]}):
        goals = goals[:-1]
        value = {**value, "requirements": goals, "operation": "compare"}
    for i, goal in enumerate(goals):
        if not isinstance(goal, dict) or goal.get("id") != f"r{i}":
            raise ValueError("plan IDs/order invalid")
        text = goal.get("text")
        deps = goal.get("depends_on")
        if not isinstance(text, str) or not 1 <= len(text) <= 600:
            raise ValueError("plan goal text invalid")
        if not text.rstrip().endswith("?"):
            raise ValueError("plan goal must be a complete question")
        if not isinstance(deps, list) or len(deps) != len(set(deps)) or any(
            d not in {f"r{j}" for j in range(i)} for d in deps):
            raise ValueError("plan DAG invalid")
        if set(PLACEHOLDER.findall(text)) != set(deps):
            raise ValueError("dependency placeholder mismatch")
        if "{{" in PLACEHOLDER.sub("", text) or "}}" in PLACEHOLDER.sub("", text):
            raise ValueError("unknown placeholder")
    if value.get("operation") not in {"answer", "compare", "intersect"}:
        raise ValueError("plan operation invalid")
    if re.search(r"\blived longer\b", question, re.I) and len(goals) >= 2:
        texts = [g["text"].casefold() for g in goals]
        if all(re.search(r"\b(die|died|death)\b", t) for t in texts):
            raise ValueError("lifespan comparison cannot use death dates alone")
    return {"requirements": goals, "operation": value["operation"], "fallback": False}


def fallback(question: str, error: str = "") -> dict:
    return {"requirements": [{"id": "r0", "text": question, "depends_on": []}],
            "operation": "answer", "fallback": True, "error": error}


def instantiate(text: str, answers: dict[str, str]) -> str:
    result = PLACEHOLDER.sub(lambda m: answers[m.group(1)], text)
    if PLACEHOLDER.search(result):
        raise ValueError("unbound placeholder")
    return result


def plan(llm, question: str) -> dict:
    pair = LIFESPAN_PAIR.match(question.strip())
    if pair:
        left, right = (name.strip() for name in pair.groups())
        goals = [{"id": f"r{i}", "text": text, "depends_on": []}
                 for i, text in enumerate((f"When was {left} born?",
                                           f"When did {left} die?",
                                           f"When was {right} born?",
                                           f"When did {right} die?"))]
        return validate({"requirements": goals, "operation": "compare"}, question)
    try:
        proposal = llm.ask("plan", SYSTEM, {"question": question}, tokens=850,
                           schema=PLAN_SCHEMA)
        try:
            validate(proposal, question)
            issue = "check hidden intermediates and semantic fidelity"
        except (ValueError, KeyError, TypeError) as first:
            issue = str(first)
        reviewed = llm.ask("plan_review", REVIEW,
                           {"original_question": question, "proposed_plan": proposal,
                            "validation_issue": issue}, tokens=900, schema=PLAN_SCHEMA)
        try:
            return validate(reviewed, question)
        except (ValueError, KeyError, TypeError) as second:
            repair = llm.ask("plan_repair", REPAIR,
                             {"original_question": question, "proposed_plan": reviewed,
                              "validation_error": str(second)}, tokens=900,
                             schema=PLAN_SCHEMA)
            return validate(repair, question)
    except Exception as exc:
        return fallback(question, str(exc))

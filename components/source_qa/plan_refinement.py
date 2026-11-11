"""Question-only plan refinement, canonical dependency IDs and qualifier preservation."""
from __future__ import annotations

import re
from .planning import PLAN_SCHEMA, validate

PLAN_REPAIR = '''Return JSON {"requirements":[{"id":"r0","text":"atomic evidence question","depends_on":[]}],"operation":"answer|compare|intersect"}.
Inspect the ORIGINAL question for an unknown person, place, work, organization or date hidden inside a noun phrase. Resolve that intermediate as its own evidence question before asking for the requested attribute. Use 2..4 ordered goals when the question needs multiple relations; otherwise use one goal. Every dependent goal must contain a literal {{r0.answer}} or similar placeholder and list the same ID in depends_on. Do not fill placeholders with a guessed answer. Preserve relation direction and all named entities. Examples:
"Who is the child of the performer of song Silver Grove?" -> "Who performed the song Silver Grove?" then "Who is the child of {{r0.answer}}?"
"What is the place of birth of Mira Vale's husband?" -> "Who is Mira Vale's husband?" then "Where was {{r0.answer}} born?"
"What company succeeded the owner of Cedar Sports Network?" -> "What company owned Cedar Sports Network?" then "What company succeeded {{r0.answer}}?"
For comparisons, retrieve each required attribute before comparing. If a split cannot preserve the question, return the original one-goal plan. Return only JSON.'''

BOUND = re.compile(r"\{\{(r[0-3])\.answer\}\}")

BARE = re.compile(r"(?<!\{)\b(r[0-3])\.answer\b(?!\})")

def canonicalize(proposal: dict, question: str) -> dict:
    """Order goals as emitted, renumber IDs, and remove repeated original goal.

    Llama occasionally emits r1,r2,r3,r0 or appends the original question as
    a duplicate final goal. IDs are formatting; the ordered dependency graph
    and source question determine the executable plan.
    """
    raw = proposal.get("requirements")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
        raise ValueError("invalid proposed goals")
    goals = [g for g in raw if isinstance(g, dict)]
    if len(goals) != len(raw):
        raise ValueError("invalid goal object")
    if len(goals) > 1:
        duplicates = [j for j, g in enumerate(goals)
                      if str(g.get("text", "")).strip().casefold() == question.strip().casefold()]
        if duplicates:
            goals = [g for j, g in enumerate(goals) if j not in duplicates]
    old_ids = [g.get("id") for g in goals]
    if len(old_ids) != len(set(old_ids)) or any(not isinstance(v, str) for v in old_ids):
        raise ValueError("duplicate or invalid goal IDs")
    mapping = {old: f"r{i}" for i, old in enumerate(old_ids)}
    revised = []
    for i, goal in enumerate(goals):
        text = str(goal.get("text", ""))
        text = BARE.sub(lambda m: "{{" + m.group(1) + ".answer}}", text)
        def replace(match: re.Match[str]) -> str:
            old = match.group(1)
            if old not in mapping:
                raise ValueError("unknown plan dependency")
            return "{{" + mapping[old] + ".answer}}"
        text = BOUND.sub(replace, text)
        deps = goal.get("depends_on")
        if not isinstance(deps, list) or any(d not in mapping for d in deps):
            raise ValueError("unknown plan dependency")
        revised.append({"id": f"r{i}", "text": text,
                        "depends_on": [mapping[d] for d in deps]})
    return {"requirements": revised, "operation": proposal.get("operation", "answer")}

def refine_plan_normalized(llm, question: str, original: dict) -> dict:
    if len(original.get("requirements", [])) != 1:
        return original
    try:
        proposal = llm.ask("plan_refinement", PLAN_REPAIR,
                           {"question": question, "proposed_plan": original},
                           tokens=850, schema=PLAN_SCHEMA)
        revised = validate(canonicalize(proposal, question), question)
        return revised if len(revised["requirements"]) > 1 else original
    except (KeyError, TypeError, ValueError):
        return original

NUMBERS = re.compile(r"(?<![A-Za-z0-9])\d+")

def refine_plan_faithful(llm, question: str, original: dict) -> dict:
    proposed = refine_plan_normalized(llm, question, original)
    source_numbers = set(NUMBERS.findall(question))
    plan_numbers = set(NUMBERS.findall(" ".join(
        goal["text"] for goal in proposed["requirements"])))
    if source_numbers != plan_numbers:
        return original
    return proposed

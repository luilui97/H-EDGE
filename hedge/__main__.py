"""Run H-EDGE on a question using a supplied extracted index and model services."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from components.source_qa.llm import LocalLLM
from . import HEDGE, Settings, HierarchicalKnowledgeHypergraph
from .embeddings import NVEmbedRetrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--encoder", required=True, help="importable module:factory for the NV-Embed-v2 adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-output", type=Path, help="save admitted H_(t+1) for subsequent questions")
    parser.add_argument("--cache", type=Path, default=Path("cache/hedge"))
    parser.add_argument("--model", default="llama3.1:70b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11438/api/")
    parser.add_argument("--model-digest", help="optional expected service model digest")
    parser.add_argument("--visit-budget", type=int, default=256)
    parser.add_argument("--document-budget", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()
    # Output paths cannot overwrite the corpus or the input memory.
    protected = {args.documents.resolve(), args.index.resolve(),
                 (args.embeddings / "passages.npy").resolve(),
                 (args.embeddings / "passages.manifest.json").resolve()}
    outputs = [args.output] + ([args.memory_output] if args.memory_output else [])
    if any(p.resolve() in protected for p in outputs) or len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("output paths must be distinct from inputs and each other")
    documents = json.loads(args.documents.read_text(encoding="utf-8"))
    memory = HierarchicalKnowledgeHypergraph(documents, json.loads(args.index.read_text(encoding="utf-8")))
    module, factory = args.encoder.split(":", 1)
    encoder = getattr(importlib.import_module(module), factory)()
    dense = NVEmbedRetrieval(memory, args.embeddings, encoder)
    llm = LocalLLM(args.cache, model=args.model, base_url=args.base_url,
                   expected_digest=args.model_digest)
    pipeline = HEDGE(memory, llm, dense, dense.score_facts,
                     Settings(visit_budget=args.visit_budget, document_budget=args.document_budget,
                              persist_repairs=not args.no_repair))
    result = pipeline.run(args.question)
    result["models"] = {"language_model": llm.model, "language_model_digest": llm.model_digest,
                         "embedding_model": encoder.model_id}
    result["llm_calls"] = llm.calls
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.memory_output:
        pipeline.memory.save(args.memory_output)
    print(result["output"]["answer"])


if __name__ == "__main__":
    main()

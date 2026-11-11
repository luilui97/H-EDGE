# H-EDGE

H-EDGE constructs source-supported evidence for multi-hop question answering.
The public implementation is `hedge/`; `python -m hedge` runs its three stages.
The supplied paper is the specification for the module names below.


## Run

Python 3.10+ is required. From this directory:

```bash
python -m pip install -r requirements.txt
python -B verify_package.py
python -B -m unittest discover -s tests -v
python -B -m hedge --help
```

Full inference requires an extracted hypergraph index, NV-Embed-v2 passage
embeddings and online encoder, and an Ollama-compatible model service. These
model/index artifacts are **not bundled**. No inference command uses the
reference answers or supporting-document labels in `questions.json`.

```bash
python -B -m hedge \
  --question "Your multi-hop question?" \
  --documents data/musique/corpus.json \
  --index /path/to/hypergraph.json \
  --embeddings /path/to/nv/musique \
  --encoder your_nv_adapter:build \
  --model llama3.1:70b \
  --base-url http://127.0.0.1:11438/api/ \
  --output runs/question.json \
  --memory-output runs/next-memory.json
```

The encoder factory returns an object with `model_id = "nvidia/NV-Embed-v2"`,
`encode_query(text)` and `encode_passages(texts)`. It must use the same model,
retrieval instruction and embedding configuration as the passage matrices.
`hedge/embeddings.py` validates model identity, source order, matrix dimensions
and finite values, and computes cosine similarities. Occurrence embeddings are
cached by their predicate/role/qualifier representation.

The embeddings directory contains `passages.npy` and
`passages.manifest.json`. The manifest has
`configuration.encoder = "nvidia/NV-Embed-v2"` and
`configuration.texts_sha256 = texts_hash(texts)` as defined in
`hedge.embeddings`, where `texts` is the ordered list of `title + "\n" + text`.
Documents without an explicit `doc_idx` use their zero-based corpus positions.

The same `--model` serves planning, binding verification, repair verification
and answer generation. The default is Llama 3.1 70B; changing it changes the
whole reasoning configuration. The output records its actual service digest.
Use `--model-digest` to require a specific digest. `--visit-budget` defaults to
256; it controls candidate exploration independently of the document budget.
`--document-budget` defaults to 5 and cannot exceed 5. `--no-repair` preserves
proposals without admitting them. Use the saved `--memory-output` as `--index`
for a subsequent process; a reused `HEDGE` instance keeps admitted memory in
process. Input files are never overwritten by the CLI.

## Offline hyperedge extraction and index construction

`components/hyperedge_extraction.py` adapts the supplied corpus-wide extraction
code to this repository. Its prompt preserves joint-event participants,
direction, negation and temporal scope. Every returned role and qualifier must
occur verbatim in the cited quote or document title. Quotes must be contiguous
original-text spans; repeated quotes require an explicit character offset.

Run extraction on a corpus file, using a separate state file for each dataset:

```bash
python -B -m components.hyperedge_extraction \
  --documents data/musique/corpus.json \
  --state runs/extraction/musique.sqlite \
  --cache cache/hyperedge-extraction \
  --model llama3.1:70b \
  --base-url http://127.0.0.1:11438/api/
```

The component replaces the supplied script's external `.build`/`.storage`
dependencies with a self-contained SQLite progress store. It records the
corpus identity, served model digest, extractor version, occurrence records,
rejected-fact reasons and passage status. Repeating the command skips completed
passages and retries failed or partially validated passages with refreshed
responses. `--limit N` bounds pending documents for one invocation without
marking the remaining corpus complete. An incomplete run exits with code 2;
its state is retained for resumption. A changed corpus, model or extractor
requires a separate state file. Responses are cached by model digest and exact
payload; truncated or malformed JSON gets a second attempt with a larger
output allowance.

Extraction creates atomic **candidates** with named roles, qualifiers, source
offsets and role mentions in the text/title. It does not mark semantic truth or
entity identities as verified. Fact-port surfaces are retained for Source
Bridge; entity-linking aliases can be supplied with the hierarchy. Query-time
acceptance still goes through `hedge/source_verified_binding.py`.

Export the extracted occurrences into the index consumed by `python -m hedge`:

```bash
python -B -m hedge.build_index \
  --documents data/musique/corpus.json \
  --state runs/extraction/musique.sqlite \
  --hierarchy /path/to/relation_schemas.json \
  --output runs/indices/musique.json
```

The hierarchy file supplies `schemas`, `concepts` and optional `aliases` and
`anchors`, following the contract below. The exporter does not generate domain
concepts from predicate names. When semantic schema roles differ from the
extractor's `subject/object` roles, a schema can provide an explicit
`role_alignment`, for example:

```json
{
  "schemas": [{
    "id": "education:study",
    "predicates": ["studied"],
    "roles": ["person", "field"],
    "role_alignment": {"person": "subject", "field": "object"}
  }],
  "concepts": [{"id": "education", "schemas": ["education:study"]}]
}
```

This example illustrates the file format; the supplied definitions must cover
the actual corpus's predicate families, roles and qualifier constraints.
Export checks source identity, grounding, schema membership and the resulting
H-EDGE memory. Unmatched occurrences cause an error rather than being silently
dropped. By default, incomplete extraction cannot be exported; the explicit
`--allow-incomplete` option retains completion counts and incomplete status in
the output metadata. Export makes no model calls and does not modify the input
corpus, hierarchy or extraction state. Use its output as `--index` for inference.

## Extracted-index contract

`hypergraph.json` contains `occurrences`, `schemas`, `concepts`, and optionally
`blocks`, `aliases`, `anchors` and `version`. It is an extracted artifact, not
benchmark labels. The extraction component is included; schema/concept
definitions, entity-linking resources and precomputed indices are external
artifacts. The loader does not infer domain
concepts or silently replace missing semantic structure with a flat index.

- An atomic occurrence has a unique `id`, `predicate`, named string-valued
  `roles`, `qualifiers`, and nonempty `sources`. A source is
  `{"doc_idx": 0, "start": 0, "end": 42}` using Python character offsets into
  the original document text. Single-span `doc_idx/start/end` fields are accepted
  as a single source. Occurrence identity is preserved across schema routes.
- A schema has `id`, a `predicates` family, required named `roles`, and optional
  `qualifiers` and `role_values` (allowed values for each schema role; an empty
  list is unconstrained). Extra occurrence roles are permitted. Optional
  `schema_alignments[schema_id]` on a fact maps schema roles injectively to its
  occurrence roles; otherwise exact role names are used.
- A concept has `id` and a nonempty list of schema IDs in `schemas`. Every
  schema must belong to a concept. Compatible schemas receive the same
  occurrence ID rather than duplicate copies.
- Optional local `blocks` retain `doc_idx/start/end`; occurrence memberships
  are derived by source-span containment. An uncovered atomic span gets its
  own evidence block. Optional `aliases` maps surfaces to entity IDs; fact
  `ports` use `role`, `surface`, `identity_confirmed` and `alternatives`.
- A composite additionally has `members`, `joins`, `depth`, and exposed
  `predicate/roles/qualifiers`. Constituents precede composites in the file.
  Its sources must equal the union of constituent sources and its depth must
  be one plus their maximum. Persisted compositions use the same retrieval
  and per-goal verification interface as atomic occurrences.

See `tests/test_method_contract.py` for a small synthetic index and injected
service examples. These fixtures test the software contract; they are not a
replacement dataset, accuracy evaluation or reproduction of paper scores.

## Supporting components

`hedge/` defines the paper modules. Offline extraction lives in
`components/hyperedge_extraction.py`; `components/source_qa/` contains the
implementation utilities used during inference:

| Component | Responsibility |
|:--|:--|
| `planning.py`, `plan_refinement.py` | Construct and validate the question-only dependency plan; normalize references and preserve question qualifiers. |
| `store.py` | Access original documents, occurrence spans and entity aliases. |
| `bridge.py` | Implement the Source Bridge ranking and anchor expansion used by `hedge/source_bridge.py`. |
| `reasoning.py` | Build original-source windows and generate answers from the document/witness packet. Binding acceptance is implemented in `hedge/source_verified_binding.py`. |
| `llm.py` | Use one configured language-model service for planning, verification, repair checks and answer generation. |

`python -m hedge` is the single inference entry point. Model weights and the
embedding encoder are supplied through the runtime interfaces above; this
source package has no fitted answer-selection model or separate training stage.

## Included data and validation scope

| Dataset | Questions | Corpus documents |
|:--|--:|--:|
| MuSiQue | 1,000 | 11,656 |
| 2WikiMultiHopQA | 1,000 | 6,119 |
| HotpotQA | 1,000 | 9,811 |

## Acknowledgment

This repository is an initial release and will be updated progressively.

# Convergent Knowledge Engine (CKE)

Convergent Knowledge Engine (CKE) is a research infrastructure project for graph-based reasoning over knowledge.

## Architecture

CKE is organized with clear separation between core system stages:

- **Ingestion** (`extractor`): semantic extraction pipeline.
- **Memory** (`graph`): contextual knowledge graph storage and memory.
- **Retrieval** (`retrieval`): sparse retrieval and baseline RAG components.
- **Reasoning** (`reasoning`): reasoning engine over retrieved evidence.

Additional modules support datasets, routing, evaluation, experiments, and shared utilities.

## Repository Structure

```text
cke/
  datasets/
  extractor/
  graph/
  retrieval/
  reasoning/
  router/
  evaluation/
  experiments/
  utils/
  __init__.py
  version.py

configs/
  config.yaml

tests/
  __init__.py
```

## Conversational Memory + RAG

CKE now includes a conversation-first pipeline for live natural-language sessions:

- `cke.conversation.memory.ConversationalMemoryStore` stores every turn with raw text, role, turn order, timestamp, extracted entities, and extracted facts.
- `cke.conversation.retriever.ConversationalRetriever` performs dense retrieval over prior turns before enriching results with extracted facts, graph neighbors, and lightweight candidate paths.
- `cke.conversation.reference_resolution.ConversationalReferenceResolver` expands follow-up references such as `that company`, `that role`, and `when was that again?` from recent and retrieved context.
- `cke.conversation.answering.GroundedAnswerComposer` turns retrieved evidence into grounded natural-language answers, using operators only as an augmentation layer for cases like counting and existence checks.
- `cke.pipeline.conversational_orchestrator.ConversationalOrchestrator` wires ingestion, semantic retrieval, reference resolution, and grounded answering into a modular API for realistic conversational memory and RAG evaluation.

The new conversation-first evaluation scenarios live in `cke.evaluation.conversation_cases` and the end-to-end regression tests live in `tests/test_conversational_orchestrator.py`.

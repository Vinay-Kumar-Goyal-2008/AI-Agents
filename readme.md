# Scientific Agent RAG Benchmark

## Overview

This repository contains a series of experiments conducted to improve scientific question answering over a research corpus focused on AI agents.

The project began with a simple baseline Retrieval-Augmented Generation (RAG) system and gradually evolved into more advanced agentic architectures through multiple iterations and ablation studies.

The primary objective was to investigate whether retrieval planning, reflection, gap detection, and question-type-aware retrieval could improve answer quality, evidence coverage, and grounding.

---

## Research Journey

The development process followed multiple stages.

### Stage 1: Baseline RAG

The initial system used a traditional RAG pipeline.

Architecture:

Question
↓
FAISS Retrieval
↓
Top-k Chunks
↓
Gemini
↓
Answer

Characteristics:

* Dense retrieval only
* Fixed retrieval budget
* No query decomposition
* No reranking
* No reflection
* No verification

This system served as the primary baseline for all later experiments.

---

### Stage 2: Baseline Improvements

Several modifications were explored while keeping the overall architecture unchanged.

Experiments included:

#### LLM Variants

Different generation models were tested.

Examples:

* Gemini `Flash 2.5`
* `Qwen 7B`

Evaluation focused on:

* Answer quality
* Grounding quality
* Cost
* Latency

#### Prompt Engineering

Different system prompts were tested.

Examples:

* Direct answering
* Evidence-grounded answering
* Scientific summarization
* Citation-aware generation


The best prompt template out of those was 


```text
You are a strict information extractor.

Rules:
-Double check facts being asked in the question if any and if it is answered or not
- Only use facts explicitly present in evidence
- Do NOT repeat template text
- Each line must be a real fact from evidence
- Each line must end with [DOC X]
- Do NOT write generic placeholders like "Sentence -> [DOC X]"
- Take all evidence given and strictly according to the question give an paragraph answer of about {type_ans}.
- Dont try to connect two different facts if there is any evidence of there connection then only connect


Question:
{state['query']}

Evidence:
{evidence_text}

Answer:
```

#### Retrieval Configuration

Experiments included:

* Different Top-k values
* Different chunk sizes
* Different chunk overlaps
* Different embedding models

Examples:

```text
sentence-transformers/all-MiniLM-L6-v2
BAAI/bge-large-en-v1.5
BAAI/bge-base-en-v1.5
```

Result- BAAI was unnecessarily large model without much changes in the output and out of remaining two all-MiniLM-L6-v2 turned out the best.

---

### Stage 3: Agentic RAG

After establishing the baseline, a more advanced agent-based pipeline was developed using LangGraph.

Architecture:

Question

↓

Planner

↓

Query Decomposition

↓

Hybrid Retrieval (Retrieval using similarity and character matching)

↓

Cross-Encoder Reranking

↓

Reflection (checking if answer is actually supported by evidence or not)

↓

Answer Synthesis

↓

Verification

Major additions:

#### Query Planning

The system generates multiple search queries from the original question.

Example:

Question:

```text
What are the limitations of memory systems in autonomous agents?
```

Generated Queries:

```text
memory systems in autonomous agents
limitations of agent memory
long-term memory challenges
agent memory evaluation
```

#### Hybrid Retrieval

Combined:

* Dense Retrieval (FAISS)
* BM25 Retrieval

Scores were fused before reranking.

#### Cross-Encoder Reranking

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Purpose:

* Improve retrieval precision
* Reduce noisy evidence
* Removed hallucinations of cosine similarity retrieval

#### Reflection

The system attempts to determine whether the retrieved evidence is sufficient to answer the question.

#### Verification

Generated answers are checked against retrieved evidence.

---

### Stage 4: Gap-Aware and Type-Aware Agentic RAG

To address limitations observed in the original agentic pipeline, an adaptive architecture was developed.

Architecture:

Question

↓
Planner

↓
Query Decomposition

↓
Type-Aware Retrieval (Retrieve a fixed amount of evidences according to the type of question)

↓
Hybrid Retrieval

↓

Cross-Encoder Reranking

↓

Gap Detection 
↓

Targeted Retrieval (Detect the gap between query and retrieved evidence using llm and then retrieve those evidences (gap))


↓

Answer Synthesis

↓
Verification

This architecture extends the original agentic system rather than replacing it.

---

## Proposed Improvements

### Gap-Aware Retrieval

Observation:

The original agent often retrieved evidence that covered only part of a complex question.

Solution:

After retrieval, the system identifies missing information required to answer the question completely.

Example:

Question:

```text
Compare CLIP and VideoCLIP in terms of architecture, training, benchmarks, and performance.
```

Retrieved evidence may only cover:

```text
architecture
training
```

Gap Detection identifies:

```text
benchmarks
performance
```

Additional retrieval is then performed for the missing aspects.

---

### Question-Type-Aware Retrieval

Observation:

Different question types require different retrieval strategies.

Question Categories:

* Factoid
* Comparative
* Survey

Examples:

Factoid:

```text
What is Mem0?
```

Comparative:

```text
Compare SWE-Agent and OpenHands.
```

Survey:

```text
Summarize recent developments in computer use agents.
```

Retrieval budgets are adjusted based on question type.

---

## Dataset

The corpus consists of research papers collected from arXiv.

Collection Period:

```text
January 2024 - April 2026
```

Target Corpus Size:

```text
Approximately 700 papers
```

Topics include:

* Agent memory
* Long-term memory
* Computer use agents
* GUI agents
* SWE agents
* Research agents
* Agentic RAG
* Multi-agent systems
* Agent benchmarks
* Tool use
* Reflection
* Agent protocols

Collection Script:

```text
data_collection.py
```

The collection pipeline:

1. Queries arXiv using targeted search terms.
2. Filters papers using keyword scoring.
3. Removes irrelevant domains.
4. Stores metadata and abstracts.
5. Builds a research corpus for retrieval.

Output:

```text
papers.json
```

---

## Embeddings and Retrieval

Embedding Model:

```text
sentence_transformers/all-MiniLM-L6-v2
```

Vector Store:

```text
FAISS
```

Retrieval Methods:

* Dense Retrieval
* BM25 Retrieval
* Score Fusion

---

## Generation

Generation Model:

```text
Gemini
```

System Prompt:

```text
You are a strict information extractor.

Rules:
-Double check facts being asked in the question if any and if it is answered or not
- Only use facts explicitly present in evidence
- Do NOT repeat template text
- Each line must be a real fact from evidence
- Each line must end with [DOC X]
- Do NOT write generic placeholders like "Sentence -> [DOC X]"
- Take all evidence given and strictly according to the question give an paragraph answer of about {type_ans}.
- Dont try to connect two different facts if there is any evidence of there connection then only connect


Question:
{state['query']}

Evidence:
{evidence_text}

Answer:
```

Temperature:

```text
0.3
```

---

## Evaluation Strategy

Because no human-written ground truth answers are available, evaluation focuses on relative comparison between systems.

Metrics include:

### Win Rate

Pairwise comparison between systems using an LLM judge.

### Faithfulness

Percentage of answer content supported by retrieved evidence.

### Aspect Coverage

Measures how many aspects of a multi-part question are addressed.

### Evidence Diversity

Measures the diversity of retrieved sources.

### Citation Coverage

Measures the relationship between generated answers and supporting papers.

---

## Experimental Configurations

### Configuration A

Baseline RAG

### Configuration B

Prompt tuned baseline RAG

### Configuration C

Baseline RAG + Retrieval Tuning

### Configuration D

Agentic RAG

### Configuration E

Gap-Aware and Type-Aware Agentic RAG

---

## Results

| Metric             | Baseline | Prompt Tuned | Agentic RAG | Proposed |
| ------------------ | -------- | ------------ | ----------- | -------- |
| Win Rate           | 72%      | 72%          | 78%         | 79.4%    |
| Faithfulness       | 61%      | 58%          | 72%         | 73.3%    |
| Aspect Coverage    | 85%      | 86%          | 82%         | 82.9%    |
| Evidence Diversity | 60%      | 63%          | 68%         | 87.3%    |

(Since I dont have Ground truth, these are the metrics given by chatgpt by feeding it the answers of each model.)

---

## Repository Structure

```text
project/
│
|- data_collection.py
├- papers.json
│
├- baseline.py
│
├- agent.py
├- agent1.py
│-answers.json (final gap aware type aware agentic rag answers)
|- answers1.json (agentic rag answers)
|- answers2.json (Prompt tuned baseline model)
|- answers3.json (Baseline model)
|
└── README.md
```


## Order of running

```
data_collection.py
main1.py
baseline.py
agent.py (Agentic RAG)
agent1.py (gap aware, type aware agentic rag)
```
---

## Future Work

Potential future directions include:

* Evidence graph construction
* Multi-hop retrieval
* Adaptive retrieval stopping
* Scientific claim verification
* Aspect-aware reranking

---

## Acknowledgements

This project explores scientific question answering over recent AI agent literature using retrieval-augmented generation, retrieval planning, reflection, and adaptive retrieval strategies.

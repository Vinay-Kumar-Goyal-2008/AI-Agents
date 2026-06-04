# Scientific Agent RAG Benchmark

## Overview

This repository contains a series of experiments exploring how retrieval-augmented generation (RAG) systems can be improved for scientific question answering over a corpus of AI-agent research papers.

The project began with a simple baseline RAG pipeline and progressively evolved into more advanced agentic retrieval architectures. Multiple ablation studies were conducted to evaluate the impact of prompt engineering, retrieval tuning, query planning, reranking, reflection, verification, gap-aware retrieval, and question-type-aware retrieval.

The primary objective was to investigate whether adaptive retrieval strategies can improve answer quality, evidence coverage, faithfulness, and grounding when answering research questions about AI agents.

---

## Main Contributions

This project makes the following contributions:

1. Built a scientific question-answering benchmark over recent AI-agent literature.
2. Developed and compared five progressively more advanced RAG architectures.
3. Evaluated the impact of prompt engineering, retrieval tuning, and agentic retrieval strategies.
4. Introduced a Gap-Aware Retrieval mechanism that identifies missing aspects of a question and performs targeted evidence collection.
5. Introduced Question-Type-Aware Retrieval that adapts retrieval behavior for factoid, comparative, and survey-style questions.
6. Conducted comparative evaluation across multiple retrieval architectures using a benchmark of scientific research questions.

---

## Research Journey

The development process followed five experimental configurations.

### Configuration A – Baseline RAG

The initial system used a traditional retrieval-augmented generation pipeline.

Architecture:

```text
Question
↓
FAISS Retrieval
↓
Top-k Chunks
↓
Gemini
↓
Answer
```

Characteristics:

* Dense retrieval only
* Fixed retrieval budget
* No query decomposition
* No reranking
* No reflection
* No verification

This system served as the primary baseline for all later experiments.

---

### Configuration B – Prompt-Tuned Baseline RAG

The retrieval pipeline remained unchanged while prompt engineering techniques were explored.

Experiments included:

* Direct answering
* Evidence-grounded answering
* Scientific summarization prompts
* Citation-aware generation

Best-performing prompt:

```text
You are a strict information extractor.

Rules:
- Double check facts being asked in the question if any and if it is answered or not
- Only use facts explicitly present in evidence
- Do NOT repeat template text
- Each line must be a real fact from evidence
- Each line must end with [DOC X]
- Do NOT write generic placeholders like "Sentence -> [DOC X]"
- Take all evidence given and strictly according to the question give a paragraph answer
- Do not connect facts unless evidence explicitly supports the connection
```

Goals:

* Improve grounding
* Improve factual consistency
* Improve citation quality

---

### Configuration C – Retrieval-Tuned RAG

This configuration is an architectural change in base agentic rag in which for each query an llm checks whether that retrieval supports the given query or not

Architecture:

```text
Question
↓
Planner
↓
Query Decomposition
↓
Hybrid Retrieval
↓
Check if the retrieval supports query or not
↓ yes
Cross-Encoder Reranking
↓
Reflection
↓
Answer Synthesis
↓
Verification
```

#### Retrieval Parameters

* Top-k retrieval budget
* Chunk size
* Chunk overlap
* Embedding model selection

#### Retrieval Variants

* Dense Retrieval
* BM25 Retrieval
* Hybrid Retrieval


---

### Configuration D – Agentic RAG

After establishing the baseline systems, an agent-based retrieval pipeline was developed using LangGraph.

Architecture:

```text
Question
↓
Planner
↓
Query Decomposition
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking
↓
Reflection
↓
Answer Synthesis
↓
Verification
```

Major additions:

#### Query Planning

The system generates multiple retrieval queries from the original question.

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

#### Cross-Encoder Reranking

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Purpose:

* Improve retrieval precision
* Reduce noisy evidence
* Improve ranking quality

#### Reflection

The agent evaluates whether retrieved evidence is sufficient to answer the question.

#### Verification

Generated answers are checked against retrieved evidence to identify unsupported claims.

---

### Configuration E – Gap-Aware + Type-Aware Agentic RAG

To address limitations observed in the original agentic pipeline, an adaptive retrieval architecture was developed.

Architecture:

```text
Question
↓
Planner
↓
Query Decomposition
↓
Type-Aware Retrieval
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking
↓
Gap Detection
↓
Targeted Retrieval
↓
Answer Synthesis
↓
Verification
```

This architecture extends the original Agentic RAG system rather than replacing it.

---

## Proposed Improvements

### Gap-Aware Retrieval

#### Observation

Complex scientific questions often contain multiple independent aspects.

Traditional retrieval frequently covers only a subset of the required information.

#### Solution

After retrieval, the system identifies missing aspects required to fully answer the question.

Example:

Question:

```text
Compare CLIP and VideoCLIP in terms of architecture, training, benchmarks, and performance.
```

Retrieved evidence may cover:

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

Benefits:

* Improved aspect coverage
* Better evidence diversity
* More complete answers

---

### Question-Type-Aware Retrieval

#### Observation

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
Summarize recent developments in computer-use agents.
```

The system adapts retrieval budgets and evidence requirements according to question complexity.

Benefits:

* Improved coverage on multi-aspect questions
* Better retrieval allocation
* Improved answer completeness

---

## Dataset

The corpus consists of research papers collected from arXiv.

Collection Period:

```text
January 2024 – April 2026
```

Corpus Size:

```text
Approximately 700 papers
```

Topics include:

* Agent memory
* Long-term memory
* Computer-use agents
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

Pipeline:

1. Query arXiv using targeted search terms.
2. Filter papers using keyword scoring.
3. Remove irrelevant domains.
4. Store metadata and abstracts.
5. Build a scientific retrieval corpus.

Output:

```text
papers.json
```

---

## Embeddings and Retrieval

Embedding Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Vector Store:

```text
FAISS
```

Retrieval Methods:

* Dense Retrieval
* BM25 Retrieval
* Hybrid Retrieval
* Score Fusion
* Cross-Encoder Reranking

---

## Generation

Generation Model:

```text
Gemini
```

Temperature:

```text
0.3
```

---

## Evaluation Methodology

The benchmark consists of 30 scientific questions spanning multiple categories.

Question Types:

* Factoid Questions
* Comparative Questions
* Survey Questions

Since no human-written reference answers were available, evaluation focused on relative comparison between systems.

Metrics:

### Win Rate

Pairwise comparison using an LLM evaluator.

### Faithfulness

Percentage of answer content supported by retrieved evidence.

### Aspect Coverage

Measures how many required aspects of a question are addressed.

### Evidence Diversity

Measures diversity of supporting sources and cited papers.

### Citation Coverage

Measures alignment between generated content and supporting evidence.

### Composite Score

```text
Composite Score =
(Win Rate + Faithfulness + Aspect Coverage + Evidence Diversity) / 4
```

---

## Experimental Configurations

| Configuration | Description                        | File          |
| ------------- | ---------------------------------- | ------------- |
| A             | Baseline RAG                       | answers3.json |
| B             | Prompt-Tuned Baseline RAG          | answers2.json |
| C             | Retrieval-Tuned RAG                | answers4.json |
| D             | Agentic RAG                        | answers1.json |
| E             | Gap-Aware + Type-Aware Agentic RAG | answers.json  |

---

## Results

### Overall Results

| Configuration                          | Win Rate | Faithfulness | Aspect Coverage | Evidence Diversity | Composite Score |
| -------------------------------------- | -------- | ------------ | --------------- | ------------------ | --------------- |
| A – Baseline RAG                       | 87.0%    | 90.0%        | 84.0%           | 80.0%              | 85.3%           |
| B – Prompt-Tuned Baseline              | 79.3%    | 83.7%        | 78.3%           | 78.0%              | 79.8%           |
| C – Retrieval-Tuned RAG                | 72.3%    | 81.3%        | 71.7%           | 87.7%              | 78.3%           |
| D – Agentic RAG                        | 86.3%    | 87.3%        | 87.0%           | 89.0%              | 87.4%           |
| E – Gap-Aware + Type-Aware Agentic RAG | 91.7%    | 90.3%        | 94.0%           | 92.3%              | 92.1%           |

### Type-Wise Win Rate

| Model | Factoid | Comparative | Survey |
| ----- | ------- | ----------- | ------ |
| A     | 95%     | 86%         | 80%    |
| B     | 85%     | 78%         | 75%    |
| C     | 82%     | 70%         | 65%    |
| D     | 88%     | 87%         | 84%    |
| E     | 86%     | 93%         | 96%    |

### Type-Wise Faithfulness

| Model | Factoid | Comparative | Survey |
| ----- | ------- | ----------- | ------ |
| A     | 94%     | 90%         | 86%    |
| B     | 85%     | 84%         | 82%    |
| C     | 83%     | 81%         | 80%    |
| D     | 89%     | 88%         | 85%    |
| E     | 88%     | 91%         | 92%    |

### Type-Wise Aspect Coverage

| Model | Factoid | Comparative | Survey |
| ----- | ------- | ----------- | ------ |
| A     | 98%     | 82%         | 72%    |
| B     | 90%     | 75%         | 70%    |
| C     | 85%     | 68%         | 62%    |
| D     | 92%     | 86%         | 83%    |
| E     | 90%     | 95%         | 97%    |

### Type-Wise Evidence Diversity

| Model | Factoid | Comparative | Survey |
| ----- | ------- | ----------- | ------ |
| A     | 78%     | 80%         | 82%    |
| B     | 76%     | 78%         | 80%    |
| C     | 85%     | 88%         | 90%    |
| D     | 87%     | 89%         | 91%    |
| E     | 90%     | 92%         | 95%    |

## Abalation Table 

| Model | Planner   | Hybrid Retrieval | Reranker | Reflection | Retrieval QA Filter | Gap-Aware | Type-Aware | Win Rate | Faithfulness | Aspect Coverage | Evidence Diversity | Composite |
| ----- | --------- | ---------------- | -------- | ---------- | ------------------- | --------- | ---------- | -------- | ------------ | --------------- | ------------------ | --------- |
| **A** | ✗         | Dense only       | ✗        | ✗          | ✗                   | ✗         | ✗          | 87.0     | 90.0         | 84.0            | 80.0               | 85.3      |
| **B** | ✗         | Dense only       | ✗        | ✗          | ✗                   | ✗         | ✗          | 79.3     | 83.7         | 78.3            | 78.0               | 79.8      |
| **C** | ✔ (light) | ✔                | ✔        | ✗          | ✔                   | ✗         | ✗          | 72.3     | 81.3         | 71.7            | 87.7               | 78.3      |
| **D** | ✔         | ✔                | ✔        | ✔          | ✗                   | ✗         | ✗          | 86.3     | 87.3         | 87.0            | 89.0               | 87.4      |
| **E** | ✔         | ✔                | ✔        | ✔          | ✔                   | ✔         | ✔          | 91.7     | 90.3         | 94.0            | 92.3               | 92.1      |

---

## Key Findings

* The Gap-Aware and Type-Aware Agentic RAG configuration achieved the strongest overall performance under the chosen evaluation protocol.
* Baseline RAG remained highly competitive on factoid questions where information requirements were narrow and well-defined.
* Agentic retrieval strategies produced the largest gains on comparative and survey questions requiring synthesis across multiple papers.
* Gap-aware retrieval improved aspect coverage and evidence diversity by identifying missing information after retrieval and performing targeted evidence collection.
* Question-type-aware retrieval significantly improved answer completeness on complex scientific questions.
* Retrieval tuning alone improved evidence diversity but did not consistently improve answer quality or coverage.
* Adaptive retrieval strategies contributed more to scientific QA performance than prompt engineering alone.

---

## Limitations

* Evaluation relies on an LLM-based judge rather than human annotators.
* The benchmark contains only 30 scientific questions.
* The corpus focuses exclusively on AI-agent literature.
* Performance may vary with different retrieval models, embedding models, and corpus sizes.
* Results should be interpreted as relative comparisons between architectures rather than definitive benchmark measurements.

---

## Repository Structure

```text
project/
│
├── data_collection.py
├── papers.json
│
├── baseline.py
├── agent.py
├── agent1.py
|__agent2.py
│
├── answers.json
├── answers1.json
├── answers2.json
├── answers3.json
├── answers4.json
│
└── README.md
```

---

## Running the Project

```text
data_collection.py
main1.py
baseline.py
agent.py
agent1.py
agent2.py
```

---

## Future Work

Potential future directions include:

* Evidence graph construction
* Multi-hop retrieval
* Adaptive retrieval stopping
* Retrieval confidence estimation
* Aspect-aware reranking
* Scientific claim verification
* Retrieval quality validation
* Memory-augmented retrieval agents
* Multi-agent retrieval systems
* Agent self-correction loops

---

## Acknowledgements

This project explores scientific question answering over recent AI-agent literature using retrieval-augmented generation, retrieval planning, query decomposition, hybrid retrieval, reranking, reflection, verification, gap-aware retrieval, and adaptive retrieval strategies.

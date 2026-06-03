import os
import json
import torch
from typing import TypedDict, List, Any, Optional, Dict
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from transformers import pipeline
from langchain_classic.llms.base import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
os.environ["GOOGLE_API_KEY"] = "API_KEY"
with open("papers.json", "r", encoding="utf-8") as f:
    papers=json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)


class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    docs: List[Any]
    evidence: List[dict]
    answer: str
    iteration: int
    done: bool
    critique: str
    type_of_ans:str
    cited:List[str]

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

class LocalLLM(LLM):
    @property
    def _llm_type(self):
        return "local"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        out = pipe(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            return_full_text=False
        )
        return out[0]["generated_text"]

llm = LocalLLM()
llmfin = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": 'cpu'},
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
all_docs = list(vector_store.docstore._dict.values())

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)
paper_lookup = {
    p["title"]: p
    for p in papers
}
def tokenize(text):
    return [w.lower().strip(".,!?()[]{}:;\"'") for w in text.split() if w.strip()]

bm25 = BM25Okapi(
    [tokenize(d.page_content) for d in all_docs]
)
def is_valid_doc(e):
    if e["paper_id"] == "UNKNOWN":
        return False
    if "Paper Title" in e["title"]:
        return False
    if len(e["text"].strip()) < 200:
        return False
    return True


def build_evidence_text(evidence):
    evidence = [e for e in evidence if is_valid_doc(e)]

    text = ""
    for i, e in enumerate(evidence):
        text += f"""
[DOC {i}]
TITLE: {e['title']}
TEXT: {e['text'][:2000]}
"""
    return text, evidence

def planner(state):
    prompt = f"""
Read the question.

Generate 5 search queries that restate or decompose the question.

Rules:
- Do not add new facts.
- Do not infer answers.
- One query per line.
- Output only queries.

Question:
{state['query']}
"""

    res = llm.invoke(prompt)

    queries = []

    for line in res.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = line.lstrip("0123456789.-) ")

        if len(line) > 3:
            queries.append(line)

    if not queries:
        queries = [state["query"]]

    state["sub_queries"] = queries[:5]
    state["iteration"] = 0
    state["done"] = False

    return state

def retrieve(state):
    candidates = {}
    cited = set(state.get("cited", []))

    for q in state["sub_queries"]:

        dense_docs = retriever.invoke(q)

        for rank, doc in enumerate(dense_docs):
            key = doc.page_content

            candidates.setdefault(key, {
                "doc": doc,
                "score": 0
            })

            candidates[key]["score"] += 1.0 / (rank + 1)

        bm_scores = bm25.get_scores(tokenize(q))

        top_idx = sorted(
            range(len(bm_scores)),
            key=lambda i: bm_scores[i],
            reverse=True
        )[:10]

        for rank, idx in enumerate(top_idx):
            doc = all_docs[idx]

            key = doc.page_content

            candidates.setdefault(key, {
                "doc": doc,
                "score": 0
            })

            candidates[key]["score"] += 1.0 / (rank + 1)

    docs = [
        x["doc"]
        for x in sorted(
            candidates.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:20]
    ]

    pairs = [
        (
            state["query"],
            d.page_content[:1500]
        )
        for d in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    evidence = []
    seen = set()

    for doc, score in ranked:

        title = doc.metadata.get("title", "").strip()

        key = (
            title,
            doc.page_content[:300]
        )

        if key in seen:
            continue

        seen.add(key)

        evidence.append({
            "paper_id": "OK",
            "title": title,
            "text": doc.page_content
        })

        paper = paper_lookup.get(title)

        if paper:
            arxiv_id = paper.get("arxiv_id", "")

            if arxiv_id:
                cited.add(
                    arxiv_id.split("/")[-1]
                )

    state["evidence"] = evidence
    state["cited"] = list(cited)

    return state


def reflect(state):
    state["iteration"] += 1

    evidence_text, filtered = build_evidence_text(
        state["evidence"]
    )

    prompt = f"""
Answer ONLY YES or NO.

Question:
{state['query']}

Does the evidence explicitly contain the answer?

Evidence:
{evidence_text}
"""

    res = llm.invoke(prompt).strip().upper()

    if res.startswith("YES"):
        state["done"] = True

    elif state["iteration"] >= 2:
        state["done"] = True

    else:
        state["done"] = False
        prompt = f"""
Read the question.

Generate 1 search query that restate or decompose the question.

Rules:
- Do not add new facts.
- Do not infer answers.
- One query per line.
- Output only queries.

Question:
{state['query']}

do not include these:
{state['sub_queries']}
"""

        res = llm.invoke(prompt)
        queries = []

        for line in res.split("\n"):
            line = line.strip()
    
            if not line:
                continue
    
            line = line.lstrip("0123456789.-) ")
    
            if len(line) > 3:
                queries.append(line)
    
        if not queries:
            queries = []
        state['sub_queries'].extend(queries)

    state["evidence"] = filtered

    return state


def synthesize(state):
    evidence_text, _ = build_evidence_text(state["evidence"])
    type_ans=state['type_of_ans']
    if (type_ans=='factoid'):
        type_ans='50-70 words'
    elif (type_ans=='comparative'):
        type_ans='100-300 words'
    elif (type_ans=='survey'):
        type_ans='250-600 words'
    prompt = f"""
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
"""

    state["answer"] = llmfin.invoke(prompt).content
    return state


def verify(state):
    evidence_text, _ = build_evidence_text(state["evidence"])

    state["critique"] = llm.invoke(f"""
Check if every sentence is supported by DOCs.

List ONLY unsupported sentences.

Answer:
{state['answer']}

Evidence:
{evidence_text}
""")

    return state

def route(state):
    print(
        f"Iteration={state['iteration']} "
        f"Evidence={len(state['evidence'])} "
        f"Done={state['done']}"
    )

    if state["done"]:
        return "synthesize"

    return "retrieve"



graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("retrieve", retrieve)
graph.add_node("reflect", reflect)
graph.add_node("synthesize", synthesize)
graph.add_node("verify", verify)

graph.set_entry_point("planner")

graph.add_edge("planner", "retrieve")
graph.add_edge("retrieve", "reflect")

graph.add_conditional_edges(
    "reflect",
    route,
    {
        "retrieve": "retrieve",
        "synthesize": "synthesize"
    }
)

graph.add_edge("synthesize", "verify")
graph.add_edge("verify", END)

app = graph.compile()



with open("questions.jsonl", "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

ans=[]
for i in questions:
    try:
        with open('answers1.json','r',encoding='utf-8') as f:
            done=json.load(f)
    except:
        done=[]
    if done!=[]:
        done_ids = {x["id"] for x in done}
        if i["id"] in done_ids:
            print('done',i['id'])
            continue
    result=app.invoke({
    "query": i['question'],
    "sub_queries": [],
    "docs": [],
    "evidence": [],
    "answer": "",
    "iteration": 0,
    "done": False,
    "critique": "",
    "type_of_ans":i['type'],
    "cited":[]
})
    print(i['question'])
    print(result['answer'])
    print(result['cited'])
    try:
        with open("answers1.json", "r", encoding="utf-8") as f:
            answers = json.load(f)
    except:
        answers = []
    
    answers.append({
        "id": i["id"],
        "answer": result["answer"],
        "cited_papers": result["cited"]
    })
    
    with open("answers1.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
            
    
print(ans)
# print("\nVERIFICATION\n")
# print(result["critique"])
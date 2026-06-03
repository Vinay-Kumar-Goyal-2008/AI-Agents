import json
import torch
from typing import Optional, List
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "API_KEY"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)
with open("papers.json", "r", encoding="utf-8") as f:
    papers=json.load(f)
type_ans_mapping = {
    "factoid": "50-70 words",
    "comparative": "100-300 words",
    "survey": "250-600 words"
}
paper_lookup = {
    p["title"]: p
    for p in papers
}
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

def retrieve(query):
    docs = retriever.invoke(query)
    return docs


def llm_call(query, docs, type_ans):

    word_limit = type_ans_mapping[type_ans]

    evidence_text = ""
    cited_papers = []

    for i, d in enumerate(docs):
        title = d.metadata.get("title", "UNKNOWN")


        paper = paper_lookup.get(title)

        if paper:
            arxiv_id = paper.get("arxiv_id", "")

        evidence_text += f"""
[DOC {i}]
TITLE: {title}
TEXT: {d.page_content[:1500]}
"""

        if arxiv_id:
            cited_papers.append(arxiv_id.split("/")[-1])

    prompt = f"""
Answer the given question in the given length using the given evidence
- Length: {word_limit}

Question:
{query}

Evidence:
{evidence_text}

Answer:
"""

    response = llm.invoke(prompt).content.strip()

    return {
        "answer": response,
        "cited": list(set(cited_papers))
    }


def final_pipeline(query, type_ans):
    docs = retrieve(query)
    return llm_call(query, docs, type_ans)
with open("questions.jsonl", "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

for q in questions:

    try:
        with open("answers3.json", "r", encoding="utf-8") as f:
            done = json.load(f)
    except:
        done = []

    done_ids = {x["id"] for x in done} if done else set()

    if q["id"] in done_ids:
        print("done", q["id"])
        continue

    result = final_pipeline(q["question"], q["type"])

    print(q["question"])
    print(result["answer"])
    print(result["cited"])
    print("*" * 40)

    try:
        with open("answers3.json", "r", encoding="utf-8") as f:
            answers = json.load(f)
    except:
        answers = []

    answers.append({
        "id": q["id"],
        "answer": result["answer"],
        "cited_papers": result["cited"]
    })

    with open("answers3.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)
import urllib.parse
import urllib.request
import feedparser
import json
import time
from datetime import datetime

TARGET_PAPERS = 700
BATCH_SIZE = 200
SLEEP_TIME = 2

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 4, 30)

QUERIES = [
    'all:"Mem0"',
    'all:"agent memory"',
    'all:"memory augmented agent"',
    'all:"long-term memory" AND all:"agent"',
    'all:"computer use agent"',
    'all:"computer-use agent"',
    'all:"GUI agent"',
    'all:"desktop agent"',
    'all:"web agent"',
    'all:"OSWorld"',
    'all:"AppWorld"',
    'all:"UI-TARS"',
    'all:"SWE-agent"',
    'all:"OpenHands"',
    'all:"software engineering agent"',
    'all:"code agent"',
    'all:"autonomous software engineering"',
    'all:"agentic RAG"',
    'all:"retrieval augmented agent"',
    'all:"RAG agent"',
    'all:"multi-agent"',
    'all:"agent orchestration"',
    'all:"agent debate"',
    'all:"collaborative agents"',
    'all:"deep research"',
    'all:"research agent"',
    'all:"autonomous research"',
    'all:"tau-bench"',
    'all:"agent benchmark"',
    'all:"agent evaluation"',
    'all:"tool use"',
    'all:"tool-using agent"',
    'all:"function calling"',
    'all:"reflection"',
    'all:"self-reflection"',
    'all:"reflexion"',
    'all:"MCP"',
    'all:"A2A"',
    'all:"agent interoperability"',
    'all:"agent protocol"'
]

HIGH_VALUE = [
    "mem0",
    "memory",
    "long-term memory",
    "a-mem",
    "computer use",
    "computer-use",
    "gui agent",
    "desktop agent",
    "web agent",
    "osworld",
    "appworld",
    "ui-tars",
    "swe-agent",
    "openhands",
    "software engineering",
    "code agent",
    "agentic rag",
    "retrieval augmented",
    "tau-bench",
    "benchmark",
    "evaluation",
    "deep research",
    "research agent",
    "multi-agent",
    "orchestration",
    "debate",
    "reflection",
    "reflexion",
    "mcp",
    "a2a",
    "acp",
    "anp",
    "protocol",
    "tool use",
    "function calling"
]

MEDIUM_VALUE = [
    "agent",
    "autonomous",
    "planning",
    "reasoning",
    "tool",
    "retrieval",
    "memory system"
]

NOISE = [
    "protein",
    "genome",
    "medical imaging",
    "segmentation",
    "object detection",
    "speech enhancement",
    "wireless communication",
    "recommendation system",
    "robot control",
    "traffic prediction"
]

def compute_score(text):
    text = text.lower()

    score = 0

    for term in HIGH_VALUE:
        if term in text:
            score += 5

    for term in MEDIUM_VALUE:
        if term in text:
            score += 1

    for term in NOISE:
        if term in text:
            score -= 4

    return score

def is_valid_paper(title, abstract):
    text = (title + " " + abstract).lower()

    has_agent_signal = any(
        term in text for term in [
            "agent",
            "agentic",
            "autonomous",
            "multi-agent",
            "tool use",
            "tool-using",
            "memory",
            "rag",
            "benchmark",
            "evaluation",
            "research agent",
            "software engineering"
        ]
    )

    return has_agent_signal and compute_score(text) >= 5

papers = []
seen = set()

for query in QUERIES:

    start = 0

    while len(papers) < TARGET_PAPERS:

        print(f"Collected: {len(papers)}")

        encoded = urllib.parse.quote(query)

        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={encoded}"
            f"&start={start}"
            f"&max_results={BATCH_SIZE}"
            f"&sortBy=submittedDate"
            f"&sortOrder=descending"
        )

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()

            feed = feedparser.parse(data)

        except Exception:
            time.sleep(5)
            continue

        if not feed.entries:
            break

        for e in feed.entries:

            try:
                pub = datetime.strptime(
                    e.published,
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            except:
                continue

            if pub < START_DATE or pub > END_DATE:
                continue

            if e.id in seen:
                continue

            title = e.title.strip()
            abstract = e.summary.strip()

            if not is_valid_paper(title, abstract):
                continue

            seen.add(e.id)

            pdf = ""

            for l in e.links:
                if hasattr(l, "type") and l.type == "application/pdf":
                    pdf = l.href
                    break

            if not pdf:
                pdf = e.id.replace("abs", "pdf") + ".pdf"

            text = title + "\n" + abstract

            papers.append({
                "title": title,
                "abstract": abstract,
                "text": text,
                "published": pub.strftime("%Y-%m-%d"),
                "arxiv_id": e.id,
                "pdf_url": pdf,
                "score": compute_score(text)
            })

            if len(papers) >= TARGET_PAPERS:
                break

        start += BATCH_SIZE
        time.sleep(SLEEP_TIME)

with open("papers.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"\nFinal corpus size: {len(papers)}")
import asyncio
import os
import json
from typing import Any
from datetime import datetime  # <-- ADDED

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field

if load_dotenv():
    print("Loaded .env file")
else:
    print("No .env file found")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env or environment variables.")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


UNIQUE_LABELS: list[str] = []


def _extract_labels_from_read_result(result: Any) -> list[str]:
    """Normalize different MCP result shapes and extract the `labels` list."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return []

    if isinstance(result, list):
        if result and isinstance(result[0], dict) and "labels" in result[0]:
            labels = result[0].get("labels", [])
            return sorted({str(x) for x in labels if x is not None})
        if result and all(isinstance(x, (str, int, float)) for x in result):
            return sorted({str(x) for x in result})
        for item in result:
            labels = _extract_labels_from_read_result(item)
            if labels:
                return labels
        return []

    if isinstance(result, dict):
        if "labels" in result and isinstance(result["labels"], list):
            return sorted({str(x) for x in result["labels"] if x is not None})

        for key in ("records", "data", "results", "result"):
            if key in result:
                labels = _extract_labels_from_read_result(result[key])
                if labels:
                    return labels

        for k, v in result.items():
            if k == "labels" and isinstance(v, list):
                return sorted({str(x) for x in v if x is not None})
            labels = _extract_labels_from_read_result(v)
            if labels:
                return labels

    return []


async def fetch_unique_labels_via_mcp(read_tool) -> list[str]:
    res = await read_tool.ainvoke(
        {"query": "CALL db.labels() YIELD label RETURN collect(label) AS labels"}
    )
    return _extract_labels_from_read_result(res)

MAX_TARGET_LABELS = int(os.getenv("MAX_TARGET_LABELS", "12"))

def _parse_csv_labels(text: str) -> list[str]:
    """
    Parse comma-separated labels from LLM output.
    Handles: "A, B, C" or "[A, B]" or JSON-like accidentally.
    """
    t = (text or "").strip()

    if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
                return [str(x).strip() for x in obj["labels"] if str(x).strip()]
        except Exception:
            pass

    t = t.strip("[](){}")
    t = t.replace('"', "").replace("'", "")

    parts = [p.strip() for p in t.split(",")]
    return [p for p in parts if p]


async def select_target_labels(llm: ChatOpenAI, labels: list[str], question: str) -> list[str]:
    """
    First pass: send labels + question to LLM, return filtered labels (must exist in labels list).
    Uses your prompt format: return comma-separated labels only.
    """
    if not labels:
        return []

    labels_string = ", ".join(labels)

    prompt = (
        "You are a database schema expert. A user has a question about a cable harness database. "
        "Based on their question, identify the most relevant database labels (nodes) which can be used "
        "from the following list. Return only the list of labels comma-separated, "
        "with no other text.\n\n"
        f"Question: '{question}'\n\n"
        f"Available Labels: {labels_string}"
    )

    resp = await llm.ainvoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    candidates = _parse_csv_labels(text)

    label_set = set(labels)
    lower_map = {l.lower(): l for l in labels}

    out: list[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        if c in label_set:
            chosen = c
        else:
            chosen = lower_map.get(c.lower())
        if chosen and chosen not in out:
            out.append(chosen)
        if len(out) >= MAX_TARGET_LABELS:
            break

    return out

def _escape_backticks(name: str) -> str:
    return name.replace("`", "``")

def _normalize_mcp_rows(raw: Any) -> list[dict[str, Any]]:
    """Convert MCP tool output into list[dict]."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []

    if isinstance(raw, dict):
        for k in ("data", "records", "result", "results"):
            if k in raw and isinstance(raw[k], list):
                raw = raw[k]
                break

    return raw if isinstance(raw, list) else []

async def _read_rows(read_tool, query: str) -> list[dict[str, Any]]:
    raw = await read_tool.ainvoke({"query": query})
    return _normalize_mcp_rows(raw)

async def fetch_subschema_for_label(read_tool, label: str) -> dict[str, Any]:
    """
    Sub-schema for one label:
      - properties: list[str]
      - relationships: list[{relationshipType: str, connectedNodeLabels: list[str], direction: str}]
    """
    lbl = _escape_backticks(label)

    q_props = f"""
MATCH (n:`{lbl}`)
UNWIND keys(n) AS propertyName
RETURN DISTINCT propertyName
ORDER BY propertyName
""".strip()

    q_rels = f"""
MATCH (n:`{lbl}`)-[r]->(other)
RETURN DISTINCT
  type(r) AS relationshipType,
  labels(other) AS connectedNodeLabels,
  "out" AS direction
UNION
MATCH (n:`{lbl}`)<-[r]-(other)
RETURN DISTINCT
  type(r) AS relationshipType,
  labels(other) AS connectedNodeLabels,
  "in" AS direction
ORDER BY relationshipType, direction
""".strip()

    prop_rows = await _read_rows(read_tool, q_props)
    rel_rows = await _read_rows(read_tool, q_rels)

    properties = [r["propertyName"] for r in prop_rows if isinstance(r, dict) and "propertyName" in r]
    relationships = []
    for r in rel_rows:
        if not isinstance(r, dict):
            continue
        rt = r.get("relationshipType")
        cnl = r.get("connectedNodeLabels")
        if rt:
            direction = r.get("direction")
            relationships.append({
                "relationshipType": rt,
                "connectedNodeLabels": cnl if isinstance(cnl, list) else [],
                "direction": direction,
            })

    return {
        "label": label,
        "properties": properties,
        "relationships": relationships,
    }

def print_subschema_summary(subschema: dict[str, Any]) -> None:
    label = subschema.get("label")
    props = subschema.get("properties") or []
    rels = subschema.get("relationships") or []
    rel_types = sorted({r.get("relationshipType") for r in rels if isinstance(r, dict) and r.get("relationshipType")})

    print(f"[Step-3] Sub-schema for label: {label}")
    print(f"  Properties: {len(props)}")
    if props:
        print("   - preview:", props[:25])
    print(f"  Relationship patterns: {len(rels)}")
    if rel_types:
        print("- rel type preview:", rel_types[:25])


CONFIG = {"configurable": {"thread_id": "1"}}


SYSTEM_PROMPT = """You are a Neo4j expert that knows how to write Cypher queries to address wire harness manufacturing questions."""


def pre_model_hook(state: AgentState) -> dict[str, list[AnyMessage]]:
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=30_000,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    return {"llm_input_messages": trimmed_messages}


async def print_astream(async_stream, output_messages_key: str = "llm_input_messages") -> None:
    async for chunk in async_stream:
        for node, update in chunk.items():
            print(f"Update from node: {node}")
            messages_key = output_messages_key if node == "pre_model_hook" else "messages"
            for message in update[messages_key]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        print("\n\n")


async def main():
    llm = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0,
    )

    async with stdio_client(neo4j_cypher_mcp) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await load_mcp_tools(session)

            global UNIQUE_LABELS
            read_tool = next((t for t in mcp_tools if t.name == "read_neo4j_cypher"), None)
            if read_tool is None:
                print("WARNING: MCP tool read_neo4j_cypher not found; UNIQUE_LABELS will be empty")
                UNIQUE_LABELS = []
            else:
                UNIQUE_LABELS = await fetch_unique_labels_via_mcp(read_tool)
                print(f"Loaded {len(UNIQUE_LABELS)} unique labels from Neo4j.")
                print("Label preview:", UNIQUE_LABELS[:25])

            allowed_tools = [
                tool for tool in mcp_tools if tool.name in {""}
            ]

            if os.getenv("", "0") == "1":
                allowed_tools.append()

            agent = create_react_agent(
                llm,
                allowed_tools,
                pre_model_hook=pre_model_hook,
                checkpointer=InMemorySaver(),
                prompt=SYSTEM_PROMPT,
            )

            print("\n===================================== Chat =====================================\n")

            while True:
                user_input = input("> ")
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                target_labels = await select_target_labels(llm, UNIQUE_LABELS, user_input)
                print(f"\n[Step-2] Target labels ({len(target_labels)}): {target_labels}\n")

                subschemas: dict[str, Any] = {}
                if read_tool is not None and target_labels:
                    for lbl in target_labels:
                        subschema = await fetch_subschema_for_label(read_tool, lbl)
                        subschemas[lbl] = subschema
                        print_subschema_summary(subschema)
                        print()

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = f"subschema_step3_{ts}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "question": user_input,
                                "target_labels": target_labels,
                                "subschemas": subschemas,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    print(f"[Step-3] Saved combined sub-schema to: {out_path}\n")
                else:
                    print("[Step-3] Skipped (no target labels or no read tool).\n")

                routing_ctx = f"Thesis routing: Relevant Neo4j labels for this question are: {target_labels}."
                await print_astream(
                    agent.astream(
                        {"messages": [("system", routing_ctx), ("human", user_input)]},
                        config=CONFIG,
                        stream_mode="updates",
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())

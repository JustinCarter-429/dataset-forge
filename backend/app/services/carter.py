"""Small, application-owned Carter 1.0 knowledge and inference boundary."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..domain.extraction_models import CanonicalExtractedDocument
from ..providers.contracts import ProviderError
from ..providers.runpod import RunPodProvider

MAX_DOCUMENTS = 3
MAX_RESULTS = 10
MAX_TOOL_ROUNDS = 3
CARTER_SYSTEM_PROMPT = """You are Carter 1.0. Answer questions about local documents only from tool results. Use source references returned by tools; never invent references or reveal hidden reasoning. Be concise."""


@dataclass(frozen=True)
class CarterInferenceRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int = 700


@dataclass(frozen=True)
class CarterInferenceResponse:
    content: str
    tool_calls: list[dict[str, Any]]


class CarterProvider(Protocol):
    runtime: str
    def available(self) -> dict[str, Any]: ...
    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse: ...


class LMStudioCarterProvider:
    runtime = "local"
    def __init__(self, base_url: str, model: str, timeout: float, enabled: bool):
        self.base_url, self.model, self.timeout, self.enabled = base_url.rstrip("/"), model, timeout, enabled

    def available(self) -> dict[str, Any]:
        if not self.enabled:
            return {"configured": False, "available": False, "model": self.model}
        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=min(self.timeout, 3))
            models = response.json().get("data", []) if response.is_success else []
            loaded = any(item.get("id") == self.model for item in models if isinstance(item, dict))
            return {"configured": True, "available": loaded, "model": self.model}
        except (httpx.HTTPError, ValueError):
            return {"configured": True, "available": False, "model": self.model}

    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse:
        state = self.available()
        if not state["configured"]:
            raise ProviderError("LM_STUDIO_UNAVAILABLE", "Carter 1.0 local runtime is unavailable.")
        if not state["available"]:
            raise ProviderError("LM_STUDIO_MODEL_NOT_LOADED", "Local Carter 1.0 model not available.")
        payload = {"model": self.model, "messages": request.messages, "tools": request.tools, "tool_choice": "auto", "max_tokens": request.max_tokens, "temperature": 0.1, "stream": False}
        try:
            response = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
            response.raise_for_status(); message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("LM_STUDIO_UNAVAILABLE", "Carter 1.0 local runtime is unavailable.") from exc
        return CarterInferenceResponse(str(message.get("content") or ""), list(message.get("tool_calls") or []))


class RunPodCarterProvider:
    """Adapter deliberately reusing the V1 native RunPod transport."""
    runtime = "cloud"
    def __init__(self, provider: RunPodProvider): self.provider = provider
    def available(self) -> dict[str, Any]: return {"configured": True, "available": True, "model": self.provider.config.model}
    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse:
        schema = {"type":"object", "properties":{"answer":{"type":"string"}}, "required":["answer"], "additionalProperties":False}
        job = self.provider.generate(messages=request.messages, schema=schema, max_tokens=request.max_tokens)
        output = job.output[0] if isinstance(job.output, list) and job.output else job.output
        try:
            content = output["choices"][0]["message"]["content"] if isinstance(output, dict) else ""
            parsed = json.loads(content) if isinstance(content, str) else content
            return CarterInferenceResponse(str(parsed.get("answer", "")), [])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("CARTER_PROVIDER_INVALID_RESPONSE", "Carter cloud runtime returned an invalid response.") from exc


class KnowledgeStore:
    def __init__(self, path: Path):
        self.path = path; path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, name TEXT NOT NULL, file_type TEXT NOT NULL, extractor TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS source_units (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, section TEXT, page INTEGER, unit_type TEXT, text TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS source_units_fts USING fts5(id UNINDEXED, text, section);""")
    def _connect(self): return sqlite3.connect(self.path)
    def reset(self) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM source_units_fts")
            db.execute("DELETE FROM source_units")
            db.execute("DELETE FROM documents")
    def ingest(self, document: CanonicalExtractedDocument) -> None:
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM documents WHERE id=?", (document.document_id,)).fetchone()
            if not exists and db.execute("SELECT count(*) FROM documents").fetchone()[0] >= MAX_DOCUMENTS:
                raise ValueError("Carter local knowledge supports at most 3 documents.")
            db.execute("INSERT OR REPLACE INTO documents(id,name,file_type,extractor) VALUES(?,?,?,?)", (document.document_id, document.source_filename, document.source_filename.rsplit('.', 1)[-1].lower(), document.extractor))
            old_ids = [row[0] for row in db.execute("SELECT id FROM source_units WHERE document_id=?", (document.document_id,))]
            if old_ids: db.execute("DELETE FROM source_units_fts WHERE id IN (" + ",".join("?" for _ in old_ids) + ")", old_ids)
            db.execute("DELETE FROM source_units WHERE document_id=?", (document.document_id,))
            for element in document.elements:
                if not element.text.strip(): continue
                section = " / ".join(element.section_path)
                db.execute("INSERT OR REPLACE INTO source_units VALUES(?,?,?,?,?,?)", (element.element_id, document.document_id, section, element.page_number, element.type.value, element.text))
                db.execute("INSERT INTO source_units_fts VALUES(?,?,?)", (element.element_id, element.text, section))
    def documents(self) -> list[dict[str, str]]:
        with self._connect() as db: return [{"documentId": r[0], "name": r[1], "fileType": r[2]} for r in db.execute("SELECT id,name,file_type FROM documents ORDER BY name")]
    def search(self, query: str, document_ids: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip(): raise ValueError("Search query is required.")
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query)
        if not terms: return []
        fts_query = " AND ".join(f'"{term}"' for term in terms[:12])
        limit = max(1, min(limit, MAX_RESULTS)); ids = document_ids or []
        where, params = "", [fts_query]
        if ids: where = " AND u.document_id IN (" + ",".join("?" for _ in ids) + ")"; params.extend(ids)
        params.append(limit)
        sql = "SELECT u.id,u.document_id,d.name,u.section,u.page,u.text FROM source_units_fts f JOIN source_units u ON u.id=f.id JOIN documents d ON d.id=u.document_id WHERE source_units_fts MATCH ?" + where + " ORDER BY bm25(source_units_fts) LIMIT ?"
        with self._connect() as db: rows = db.execute(sql, params).fetchall()
        return [{"sourceRef": r[0], "documentId": r[1], "documentName": r[2], "section": r[3], "page": r[4], "text": r[5][:1200]} for r in rows]
    def source_units(self, refs: list[str]) -> list[dict[str, Any]]:
        refs = refs[:MAX_RESULTS]
        if not refs: return []
        with self._connect() as db: rows = db.execute("SELECT u.id,u.document_id,d.name,u.section,u.page,u.text FROM source_units u JOIN documents d ON d.id=u.document_id WHERE u.id IN (" + ",".join("?" for _ in refs) + ")", refs).fetchall()
        if len(rows) != len(set(refs)): raise ValueError("One or more Carter source references were not found.")
        return [{"sourceRef": r[0], "documentId": r[1], "documentName": r[2], "section": r[3], "page": r[4], "text": r[5][:1200]} for r in rows]


TOOL_SCHEMAS = [{"type":"function","function":{"name":"list_documents","description":"List available local knowledge documents.","parameters":{"type":"object","properties":{},"additionalProperties":False}}}, {"type":"function","function":{"name":"search_local_knowledge","description":"Search local document evidence.","parameters":{"type":"object","properties":{"query":{"type":"string"},"documentIds":{"type":"array","items":{"type":"string"}},"limit":{"type":"integer","minimum":1,"maximum":10}},"required":["query"],"additionalProperties":False}}}, {"type":"function","function":{"name":"get_source_units","description":"Get exact document evidence by source references.","parameters":{"type":"object","properties":{"sourceRefs":{"type":"array","items":{"type":"string"},"maxItems":10}},"required":["sourceRefs"],"additionalProperties":False}}}]

class CarterAskService:
    def __init__(self, store: KnowledgeStore, provider: CarterProvider): self.store, self.provider = store, provider

    def _tool_result(self, name: str, raw_arguments: Any) -> list[dict[str, Any]]:
        """Execute the intentionally small tool surface using identifiers only.

        Tool arguments are JSON data, never paths.  The store only resolves IDs
        from its SQLite tables, so an attempted path can neither escape the
        knowledge database nor cause a filesystem read.
        """
        if not isinstance(raw_arguments, dict):
            raise ValueError("Carter tool arguments must be an object.")
        if name == "list_documents":
            if raw_arguments:
                raise ValueError("list_documents does not accept arguments.")
            return self.store.documents()
        if name == "search_local_knowledge":
            if set(raw_arguments) - {"query", "documentIds", "limit"}:
                raise ValueError("search_local_knowledge received unknown arguments.")
            query = raw_arguments.get("query")
            ids = raw_arguments.get("documentIds")
            limit = raw_arguments.get("limit", 5)
            if not isinstance(query, str) or not query.strip() or (ids is not None and (not isinstance(ids, list) or not all(isinstance(value, str) for value in ids))) or not isinstance(limit, int):
                raise ValueError("search_local_knowledge arguments are invalid.")
            self._validate_document_ids(ids or [])
            return self.store.search(query, ids, limit)
        if name == "get_source_units":
            if set(raw_arguments) != {"sourceRefs"} or not isinstance(raw_arguments.get("sourceRefs"), list) or not all(isinstance(value, str) for value in raw_arguments["sourceRefs"]):
                raise ValueError("get_source_units arguments are invalid.")
            return self.store.source_units(raw_arguments["sourceRefs"])
        raise ValueError("Unknown Carter tool requested.")

    def _validate_document_ids(self, ids: list[str]) -> None:
        known = {item["documentId"] for item in self.store.documents()}
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate Carter document IDs are not allowed.")
        if any(identifier not in known for identifier in ids):
            raise ValueError("One or more Carter document IDs were not found.")

    @staticmethod
    def _call_parts(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        function = call.get("function", call)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Carter provider returned a malformed tool call.")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try: arguments = json.loads(arguments)
            except json.JSONDecodeError as exc: raise ValueError("Carter provider returned malformed tool arguments.") from exc
        return function["name"], arguments, str(call.get("id", function["name"]))

    def ask(self, question: str, document_ids: list[str] | None = None) -> dict[str, Any]:
        if not question.strip(): raise ValueError("A Carter question is required.")
        document_ids = document_ids or []
        self._validate_document_ids(document_ids)
        results = self.store.search(question, document_ids, 5)
        if not results and not (getattr(self.provider, "allow_empty_retrieval", False) and "TEST_ASK_FAILURE" in question): return {"answer":"I could not find relevant information in the selected local documents.", "sources":[], "assistant":"Carter 1.0", "runtime":self.provider.runtime}
        messages = [{"role":"system","content":CARTER_SYSTEM_PROMPT}, {"role":"user","content":question}, {"role":"system","content":"Retrieved evidence (cite only these sourceRef values): " + json.dumps(results)}]
        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            response = self.provider.infer(CarterInferenceRequest(messages, TOOL_SCHEMAS))
            if not response.tool_calls:
                return {"answer": response.content or "I found relevant source evidence.", "sources":[{k:v for k,v in item.items() if k != "text"} for item in results], "assistant":"Carter 1.0", "runtime":self.provider.runtime, "toolRounds":tool_round}
            if tool_round >= MAX_TOOL_ROUNDS:
                raise ProviderError("CARTER_TOOL_ROUND_LIMIT", "Carter requested more than three tool rounds.")
            messages.append({"role":"assistant", "content": response.content or "", "tool_calls": response.tool_calls})
            for call in response.tool_calls:
                name, arguments, call_id = self._call_parts(call)
                result = self._tool_result(name, arguments)
                messages.append({"role":"tool", "tool_call_id":call_id, "name":name, "content":json.dumps(result)})
        raise AssertionError("unreachable")

"""Small, application-owned Carter 1.0 knowledge and inference boundary."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..domain.extraction_models import CanonicalExtractedDocument
from ..providers.contracts import ProviderError
from ..providers.runpod import RunPodProvider

MAX_DOCUMENTS = 3
MAX_RESULTS = 10
MAX_TOOL_ROUNDS = 3
SEARCH_STOPWORDS = {
    "a", "about", "an", "and", "are", "document", "documents", "do", "does", "for", "from",
    "how", "is", "of", "on", "say", "the", "these", "this", "to",
    "what", "which", "with",
}
CARTER_SYSTEM_PROMPT = """You are Carter 1.0. For questions grounded in selected local documents, retrieve evidence with the registered local knowledge tools before answering. Do not answer from memory, do not invent references, and do not reveal hidden reasoning. After evidence is available, return only JSON shaped exactly as {\"answer\": string, \"citations\": [{\"sourceRef\": string}]}; every citation must be a sourceRef returned by a tool."""


def exact_output_contract_instruction(schema: dict[str, Any]) -> str:
    """Return a provider-neutral, schema-derived output instruction.

    The frozen Carter prompts establish the semantic contract; this envelope
    gives the active model call the exact compiled schema and cardinality.
    """
    records = None
    for branch in schema.get("allOf", []):
        candidate = branch.get("then", {}).get("properties", {}).get("records")
        if isinstance(candidate, dict):
            records = candidate; break
    record_schema = schema.get("$defs", {}).get("dynamic_record_template", {})
    properties = record_schema.get("properties", {}) if isinstance(record_schema, dict) else {}
    dynamic_fields = sorted(name for name in properties if name != "evidence")
    required_fields = list(record_schema.get("required", [])) if isinstance(record_schema, dict) else []
    contract = {
        "instruction": "Return exactly one JSON value conforming to output_schema. No markdown, code fences, commentary, explanation, schema description, summary, reasoning, provider metadata, tool metadata, or fields outside output_schema. Every required field must be present with its exact declared JSON type; never use null unless output_schema permits it. Evidence must match its declared structure exactly.",
        "batch_record_count": {"min": records.get("minItems"), "max": records.get("maxItems")} if isinstance(records, dict) else None,
        "dynamic_fields": dynamic_fields,
        "required_record_fields": required_fields,
        "output_schema": schema,
    }
    return "AUTHORITATIVE_OUTPUT_CONTRACT=" + json.dumps(contract, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CarterInferenceRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int = 4096
    tool_choice: str = "auto"
    response_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class CarterInferenceResponse:
    content: str
    tool_calls: list[dict[str, Any]]


class CarterCitation(BaseModel):
    source_ref: str = Field(alias="sourceRef", min_length=1, max_length=200)
    model_config = {"populate_by_name": True, "extra": "forbid"}


class CarterFinalResponse(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)
    citations: list[CarterCitation] = Field(min_length=1, max_length=10)
    model_config = {"extra": "forbid"}


class CarterProvider(Protocol):
    runtime: str
    def available(self) -> dict[str, Any]: ...
    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse: ...


class LMStudioCarterProvider:
    runtime = "local_lm_studio"
    def __init__(self, base_url: str, model: str, timeout: float, enabled: bool, max_tokens: int = 4096):
        self.base_url, self.model, self.timeout, self.enabled, self.max_tokens = base_url.rstrip("/"), model, timeout, enabled, max_tokens
        self.invocations = 0

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
        messages = list(request.messages)
        if request.response_schema:
            messages.insert(-1 if messages else 0, {"role": "system", "content": exact_output_contract_instruction(request.response_schema)})
        payload = {"model": self.model, "messages": messages, "tools": request.tools, "tool_choice": request.tool_choice, "max_tokens": min(request.max_tokens, self.max_tokens), "temperature": 0.1, "stream": False}
        try:
            self.invocations += 1
            response = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
            response.raise_for_status(); message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("LM_STUDIO_UNAVAILABLE", "Carter 1.0 local runtime is unavailable.") from exc
        return CarterInferenceResponse(str(message.get("content") or ""), list(message.get("tool_calls") or []))


class RunPodCarterProvider:
    """Adapter deliberately reusing the V1 native RunPod transport."""
    runtime = "runpod"
    def __init__(self, provider: RunPodProvider): self.provider = provider; self.invocations = 0
    def available(self) -> dict[str, Any]: return {"configured": True, "available": True, "model": self.provider.config.model}
    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse:
        self.invocations += 1
        messages, schema = list(request.messages), request.response_schema
        if schema:
            messages.insert(-1 if messages else 0, {"role": "system", "content": exact_output_contract_instruction(schema)})
        if schema and _requires_runpod_json_object_compat(schema):
            # The deployed vLLM worker accepts simple JSON-object constraints but
            # rejects Carter's conditional/$defs schemas.  Keep the authoritative
            # schema in the prompt and validate the returned JSON in application.
            schema = {"type": "object"}
            messages.insert(-1 if messages else 0, {"role": "system", "content": "Provider constraint is JSON object; the authoritative contract remains the supplied AUTHORITATIVE_OUTPUT_CONTRACT."})
        job = self.provider.chat(messages=messages, tools=request.tools, tool_choice=request.tool_choice, schema=schema, max_tokens=min(request.max_tokens, max(1024, self.provider.config.max_model_len // 4)))
        output = job.output[0] if isinstance(job.output, list) and job.output else job.output
        try:
            message = output["choices"][0]["message"] if isinstance(output, dict) else {}
            if not isinstance(message, dict): raise TypeError("missing message")
            content, tool_calls = message.get("content"), list(message.get("tool_calls") or [])
            if not isinstance(content, str) or not content.strip():
                if not tool_calls:
                    self.provider._publish_telemetry(json_parse="NOT_RUN", dynamic_schema_validation="NOT_RUN", safe_error_code="PROVIDER_NO_FINAL_CONTENT")
                    raise ProviderError("PROVIDER_NO_FINAL_CONTENT", "Carter cloud runtime returned no final content.")
                content = ""
            return CarterInferenceResponse(content, tool_calls)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("CARTER_PROVIDER_INVALID_RESPONSE", "Carter cloud runtime returned an invalid response.") from exc


def _requires_runpod_json_object_compat(schema: dict[str, Any]) -> bool:
    """Detect schemas beyond the deployed worker's verified native subset."""
    unsupported = {"$defs", "$ref", "oneOf", "contains", "minContains", "maxContains"}
    if isinstance(schema, dict):
        return bool(unsupported & set(schema)) or any(_requires_runpod_json_object_compat(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_requires_runpod_json_object_compat(value) for value in schema)
    return False


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
        terms = [term for term in re.findall(r"[A-Za-z0-9_]{2,}", query.lower()) if term not in SEARCH_STOPWORDS]
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
    def __init__(self, store: KnowledgeStore, provider: CarterProvider, runtime: str | None = None): self.store, self.provider, self.runtime, self._allowed_document_ids = store, provider, runtime or provider.runtime, []

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
            if ids: self._validate_document_ids(ids)
            if self._allowed_document_ids and ids and set(ids) - set(self._allowed_document_ids):
                raise ValueError("Carter tool requested a document outside the selected scope.")
            return self.store.search(query, ids or self._allowed_document_ids, limit)
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

    @staticmethod
    def _final_response(content: str) -> CarterFinalResponse:
        """Accept JSON emitted in a fenced compatibility wrapper, never prose."""
        candidate = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1)
        try:
            return CarterFinalResponse.model_validate_json(candidate)
        except ValidationError as exc:
            raise ProviderError("CARTER_STRUCTURED_OUTPUT_INVALID", "Carter returned an invalid structured response.") from exc

    def ask(self, question: str, document_ids: list[str] | None = None) -> dict[str, Any]:
        if not question.strip(): raise ValueError("A Carter question is required.")
        document_ids = document_ids or []
        self._validate_document_ids(document_ids)
        self._allowed_document_ids = document_ids
        selected = [item for item in self.store.documents() if item["documentId"] in document_ids]
        messages = [{"role":"system","content":CARTER_SYSTEM_PROMPT}, {"role":"system","content":"Selected document scope: " + json.dumps(selected)}, {"role":"user","content":question}]
        retrieved: dict[str, dict[str, Any]] = {}
        tools_requested: list[str] = []
        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            response = self.provider.infer(CarterInferenceRequest(messages, TOOL_SCHEMAS, tool_choice="required" if tool_round == 0 and document_ids else "auto", response_schema=CarterFinalResponse.model_json_schema() if tool_round else None))
            if not response.tool_calls:
                final = self._final_response(response.content)
                citation_refs = [citation.source_ref for citation in final.citations]
                if len(citation_refs) != len(set(citation_refs)) or any(reference not in retrieved for reference in citation_refs):
                    raise ProviderError("CARTER_CITATION_INVALID", "Carter returned citations not grounded in retrieved evidence.")
                sources = [{key: value for key, value in retrieved[reference].items() if key != "text"} for reference in citation_refs]
                return {"answer": final.answer, "sources":sources, "assistant":"Carter 1.0", "runtime":self.runtime, "logicalModel":"Carter 1.0", "technicalModel":getattr(self.provider, "model", getattr(getattr(self.provider, "provider", None), "config", None).model if getattr(getattr(self.provider, "provider", None), "config", None) else "openai/gpt-oss-20b"), "inferenceCount":getattr(self.provider, "invocations", 0), "toolRounds":tool_round, "toolsRequested":tools_requested, "retrievalResults":len(retrieved)}
            if tool_round >= MAX_TOOL_ROUNDS:
                raise ProviderError("CARTER_TOOL_ROUND_LIMIT", "Carter requested more than three tool rounds.")
            messages.append({"role":"assistant", "content": response.content or "", "tool_calls": response.tool_calls})
            for call in response.tool_calls:
                name, arguments, call_id = self._call_parts(call)
                result = self._tool_result(name, arguments)
                tools_requested.append(name)
                for item in result:
                    if isinstance(item, dict) and isinstance(item.get("sourceRef"), str): retrieved[item["sourceRef"]] = item
                messages.append({"role":"tool", "tool_call_id":call_id, "name":name, "content":json.dumps(result)})
        raise AssertionError("unreachable")

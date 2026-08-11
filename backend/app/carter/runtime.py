"""Fail-closed, provider-neutral Carter 1.0 prompt-package foundation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


PACKAGE_ERROR = "CARTER_PROMPT_PACKAGE_INVALID"
TOOLS = ("list_documents", "search_local_knowledge", "get_source_units")
DEFAULT_PACKAGE_PATH = Path(__file__).parent / "contracts" / "1.0"


class CarterPromptPackageError(RuntimeError):
    code = PACKAGE_ERROR

    def __init__(self, detail: str = "Carter prompt package is invalid."):
        super().__init__(PACKAGE_ERROR)
        self.detail = detail


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class CarterOperation:
    name: str
    system_prompt: dict[str, Any]
    task_prompt: dict[str, Any]
    output_schema: dict[str, Any]
    agent_action_schema: dict[str, Any] | None
    tools: tuple[dict[str, Any], ...]
    max_tool_rounds: int
    structured_output_required: bool


@dataclass(frozen=True)
class CarterInferenceRequest:
    """Logical request only; adapters own endpoint/auth/envelope serialization."""
    runtime: str
    operation: str
    messages: tuple[dict[str, str], ...]
    tools: tuple[dict[str, Any], ...]
    response_schema: dict[str, Any]
    package_fingerprint: str
    tool_choice: str = "auto"
    max_tokens: int = 4096


class CarterPromptPackage:
    def __init__(self, *, root: Path, manifest: dict[str, Any], contracts: dict[str, dict[str, Any]], fingerprints: dict[str, str], registry: Registry):
        self.root, self.manifest, self.contracts, self.contract_fingerprints, self.registry = root, manifest, contracts, fingerprints, registry
        self.package_version = str(manifest["package"]["package_version"])
        inventory = [{"file": name, "sha256": fingerprints[name]} for name in sorted(fingerprints)]
        self.package_fingerprint = sha256(_canonical(inventory)).hexdigest()
        self.prompts_by_id = {entry["id"]: contracts[entry["file"]] for entry in manifest["contracts"]["prompts"]}
        self.schemas_by_id = {entry["id"]: contracts[entry["file"]] for entry in manifest["contracts"]["schemas"]}
        self.tools_by_id = {entry["id"]: contracts[entry["file"]] for entry in manifest["contracts"]["tools"]}
        self.tools_by_name = {contract["name"]: contract for contract in self.tools_by_id.values()}

    @classmethod
    def load(cls, root: Path | None = None) -> "CarterPromptPackage":
        root = (root or DEFAULT_PACKAGE_PATH).resolve()
        manifest_path = root / "carter-prompt-manifest-1.0.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("id") != "carter-prompt-manifest-1.0" or manifest.get("version") != "1.0.0": raise ValueError("manifest identity")
            groups = manifest["contracts"]
            declarations = [item for group in ("prompts", "tools", "schemas") for item in groups[group]]
            names = [item["file"] for item in declarations] + [manifest_path.name]
            if len(names) != 14 or len(set(names)) != 14 or {path.name for path in root.iterdir() if path.is_file()} != set(names): raise ValueError("inventory")
            if manifest["contract_totals"] != {"prompts": 5, "tools": 3, "schemas": 5, "contracts_excluding_manifest": 13, "manifest": 1, "total_files": 14}: raise ValueError("totals")
            contracts = {manifest_path.name: manifest}
            fingerprints = {manifest_path.name: sha256(manifest_path.read_bytes()).hexdigest()}
            ids: set[str] = {manifest["id"]}
            for declaration in declarations:
                path = root / declaration["file"]
                raw = path.read_bytes(); value = json.loads(raw)
                actual_id = value.get("id")
                # JSON Schema resource IDs deliberately remain filename-based;
                # the manifest carries their logical Carter IDs.
                if path.name.endswith(".schema.json"):
                    if value.get("$id") != path.name: raise ValueError("schema resource identity")
                    actual_id = declaration["id"]
                if actual_id != declaration["id"] or value.get("version", declaration["version"]) != declaration["version"] or actual_id in ids: raise ValueError(f"contract identity: {path.name}")
                ids.add(actual_id); contracts[path.name] = value; fingerprints[path.name] = sha256(raw).hexdigest()
            tools = groups["tools"]
            if tuple(item["name"] for item in tools) != TOOLS or tuple(manifest["registered_tool_names"]) != TOOLS or len({item["name"] for item in tools}) != 3: raise ValueError("tool registry")
            prompt_ids = {item["id"] for item in groups["prompts"]}; schema_ids = {item["id"] for item in groups["schemas"]}
            if "carter-system-1.0" not in prompt_ids: raise ValueError("root")
            for prompt in groups["prompts"]:
                parent = contracts[prompt["file"]].get("parent_prompt_id")
                if prompt["id"] == "carter-system-1.0":
                    if parent is not None: raise ValueError("root parent")
                elif parent != "carter-system-1.0": raise ValueError("prompt parent")
            resources = []
            for name, value in contracts.items():
                if name.endswith(".schema.json") or name.endswith(".tool.json"):
                    resource = Resource.from_contents(value, default_specification=DRAFT202012)
                    resources.append((name, resource))
                    if "$id" in value: resources.append((value["$id"], resource))
            registry = Registry().with_resources(resources)
            for item in groups["schemas"]:
                schema = contracts[item["file"]]
                Draft202012Validator.check_schema(schema)
                Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
            for value in contracts.values():
                for reference in _references(value):
                    if reference.startswith("#"): continue
                    target = reference.split("#", 1)[0]
                    if target not in contracts: raise ValueError("external or unresolved schema reference")
            cls._validate_bindings(manifest, prompt_ids, schema_ids, {item["name"] for item in tools})
            return cls(root=root, manifest=manifest, contracts=contracts, fingerprints=fingerprints, registry=registry)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, SchemaError, ValidationError) as exc:
            raise CarterPromptPackageError(str(exc)) from exc

    @staticmethod
    def _validate_bindings(manifest: dict[str, Any], prompts: set[str], schemas: set[str], tools: set[str]) -> None:
        for binding in manifest["operation_bindings"].values():
            if binding["system_prompt"] not in prompts or binding["task_prompt"] not in prompts: raise ValueError("operation prompt")
            output = binding.get("output_schema") or binding.get("output_schema_template")
            if output not in schemas or binding.get("agent_action_schema") not in (None, "carter-tool-call-1.0"): raise ValueError("operation schema")
            if any(name not in tools for name in binding["tools"]): raise ValueError("operation tool")

    def validate(self, schema: dict[str, Any], instance: Any) -> None:
        try: Draft202012Validator(schema, registry=self.registry, format_checker=FormatChecker()).validate(instance)
        except ValidationError as exc: raise CarterPromptPackageError("Carter contract validation failed.") from exc

    def safe_validation_errors(self, schema: dict[str, Any], instance: Any, *, limit: int = 12) -> list[str]:
        """Return bounded structural diagnostics without retaining model values."""
        errors = sorted(Draft202012Validator(schema, registry=self.registry, format_checker=FormatChecker()).iter_errors(instance), key=lambda item: list(item.absolute_path))
        result: list[str] = []
        for error in errors[:limit]:
            path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
            if error.validator == "required":
                missing = next((name for name in error.validator_value if isinstance(name, str) and name not in error.instance), "required field")
                result.append(f"{path}: missing required field {missing}")
            elif error.validator == "additionalProperties":
                result.append(f"{path}: unexpected field")
            elif error.validator in {"minItems", "maxItems"}:
                result.append(f"{path}: expected {error.validator_value} {error.validator}; received {len(error.instance) if isinstance(error.instance, list) else type(error.instance).__name__}")
            else:
                result.append(f"{path}: expected {error.validator}; received {type(error.instance).__name__}")
        return result

    def resolve_operation(self, name: str, output_schema: dict[str, Any] | None = None) -> CarterOperation:
        try:
            binding = self.manifest["operation_bindings"][name]
            output = output_schema or self.schemas_by_id[binding.get("output_schema") or binding["output_schema_template"]]
            action_id = binding.get("agent_action_schema")
            action = self.schemas_by_id[action_id] if action_id else None
            return CarterOperation(name, self.prompts_by_id[binding["system_prompt"]], self.prompts_by_id[binding["task_prompt"]], output, action, tuple(self.tools_by_name[item] for item in binding["tools"]), int(binding.get("maximum_tool_rounds", 0)), bool(binding["structured_output_required"]))
        except (KeyError, TypeError, ValueError) as exc: raise CarterPromptPackageError() from exc

    def render(self, operation: CarterOperation, application_inputs: dict[str, Any], *, runtime: str) -> CarterInferenceRequest:
        if runtime not in {"cloud", "local"}: raise ValueError("runtime must be cloud or local")
        if operation.name == "dataset_generation" and operation.output_schema["$defs"]["dynamic_record_template"].get("x-dataset-forge-dynamic-record"):
            raise CarterPromptPackageError("An uncompiled generation template cannot reach a provider.")
        context = {"application_inputs": application_inputs, "operation": operation.name, "package_version": self.package_version}
        messages = ( {"role": "system", "content": _canonical(operation.system_prompt).decode()}, {"role": "system", "content": _canonical(operation.task_prompt).decode()}, {"role": "user", "content": _canonical(context).decode()} )
        tools = tuple({"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}} for tool in operation.tools)
        return CarterInferenceRequest(runtime, operation.name, messages, tools, deepcopy(operation.output_schema), self.package_fingerprint)

    def compile_generation_schema(self, spec: dict[str, Any], batch_count: int) -> dict[str, Any]:
        self.validate(self.schemas_by_id["carter-dataset-spec-1.0"], spec); validate_dataset_spec_semantics(spec)
        if spec["status"] != "ready" or not isinstance(batch_count, int) or not 1 <= batch_count <= 100: raise CarterPromptPackageError("Uncompiled generation request.")
        schema = deepcopy(self.schemas_by_id["carter-dataset-generation-result-1.0"])
        record = schema["$defs"]["dynamic_record_template"]
        record.pop("x-dataset-forge-dynamic-record", None)
        properties = record["properties"]; required = record["required"]
        mapping = {"string": {"type": "string", "minLength": 1}, "integer": {"type": "integer"}, "number": {"type": "number"}, "boolean": {"type": "boolean"}, "array_string": {"type": "array", "items": {"type": "string", "minLength": 1}}, "enum": {}}
        for field in spec["fields"]:
            compiled = deepcopy(mapping[field["type"]])
            if field["type"] == "enum": compiled["enum"] = field["enum_values"]
            for old, new in (("min_length", "minLength"), ("max_length", "maxLength"), ("minimum", "minimum"), ("maximum", "maximum"), ("min_items", "minItems"), ("max_items", "maxItems")):
                if old in field.get("constraints", {}): compiled[new] = field["constraints"][old]
            properties[field["name"]] = compiled
            if field["required"]: required.append(field["name"])
        exact_records = schema["allOf"][0]["then"]["properties"]["records"]
        exact_records["minItems"] = batch_count; exact_records["maxItems"] = batch_count
        Draft202012Validator.check_schema(schema)
        return schema


def validate_dataset_spec_semantics(spec: dict[str, Any], maximum_records: int = 100) -> None:
    if spec["status"] != "ready": return
    if spec["effective_record_count"] > maximum_records or len({field["name"] for field in spec["fields"]}) != len(spec["fields"]): raise CarterPromptPackageError("DatasetSpec semantic validation failed.")
    for field in spec["fields"]:
        constraints = field.get("constraints", {})
        if constraints.get("min_length", 0) > constraints.get("max_length", float("inf")) or constraints.get("minimum", float("-inf")) > constraints.get("maximum", float("inf")) or constraints.get("min_items", 0) > constraints.get("max_items", float("inf")): raise CarterPromptPackageError("DatasetSpec semantic validation failed.")
    if spec["dataset_type"] == "classification" and not any(field["type"] == "enum" for field in spec["fields"]): raise CarterPromptPackageError("DatasetSpec semantic validation failed.")


def _references(value: Any) -> list[str]:
    if isinstance(value, dict):
        return ([value["$ref"]] if isinstance(value.get("$ref"), str) else []) + [reference for child in value.values() for reference in _references(child)]
    if isinstance(value, list): return [reference for child in value for reference in _references(child)]
    return []


class CarterToolRegistry:
    """Canonical validation wrapper for the existing application-owned handlers."""
    def __init__(self, package: CarterPromptPackage, handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]):
        if set(handlers) != set(TOOLS) or set(package.tools_by_name) != set(TOOLS): raise CarterPromptPackageError("Invalid Carter tool binding.")
        self.package, self.handlers = package, handlers

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOLS: raise CarterPromptPackageError("Unknown Carter tool.")
        contract = self.package.tools_by_name[name]
        self.package.validate(contract["input_schema"], arguments)
        result = self.handlers[name](arguments)
        self.package.validate(contract["output_schema"], result)
        return result


def normalize_agent_action(package: CarterPromptPackage, native_calls: list[dict[str, Any]], content: Any, final_schema: dict[str, Any]) -> dict[str, Any]:
    if len(native_calls) > 1: raise CarterPromptPackageError("Multiple native tool calls are invalid.")
    if native_calls:
        call = native_calls[0].get("function", native_calls[0]); arguments = call.get("arguments", {})
        if isinstance(arguments, str): arguments = json.loads(arguments)
        action = {"action": "tool_call", "tool_call": {"name": call.get("name"), "arguments": arguments}}
    else:
        action = content if isinstance(content, dict) else json.loads(content)
    schema = deepcopy(package.schemas_by_id["carter-tool-call-1.0"])
    embedded_final = deepcopy(final_schema)
    # It is embedded beneath the action schema, not registered as a second
    # document.  Retaining its $id would make local refs resolve to the frozen
    # template resource instead of this compiled copy.
    embedded_final.pop("$id", None)
    schema["$defs"]["final_response_placeholder"] = embedded_final
    # The compiled generation schema has local ``#/$defs`` references.  Once it
    # is embedded in the action contract those references resolve against the
    # action root, so carry its definitions across without changing either
    # frozen contract.
    schema["$defs"].update(deepcopy(embedded_final.get("$defs", {})))
    package.validate(schema, action)
    if action["action"] == "final_response": package.validate(final_schema, action["final_response"])
    return action


@dataclass
class CarterAgentTurnState:
    """Application-owned, provider-neutral one-tool-per-turn budget."""
    package: CarterPromptPackage
    final_schema: dict[str, Any]
    rounds_used: int = 0

    def normalize(self, native_calls: list[dict[str, Any]], content: Any) -> dict[str, Any]:
        action = normalize_agent_action(self.package, native_calls, content, self.final_schema)
        if action["action"] == "tool_call":
            if self.rounds_used >= 3: raise CarterPromptPackageError("Carter tool-round limit exceeded.")
            self.rounds_used += 1
        return action


def validate_quality_review(package: CarterPromptPackage, review: dict[str, Any], record_refs: set[str], fields: set[str]) -> None:
    package.validate(package.schemas_by_id["carter-quality-review-result-1.0"], review)
    if (not review["issues"] and review["recommendation"] != "accept") or any(ref not in record_refs or (issue["affected_field"] is not None and issue["affected_field"] not in fields) for issue in review["issues"] for ref in issue["affected_record_refs"]): raise CarterPromptPackageError("Quality review validation failed.")


def authorize_revision(revisions_used: int, application_authorized: bool) -> int:
    if not application_authorized or revisions_used != 0: raise CarterPromptPackageError("Revision is not authorized.")
    return 1


def project_schema_for_provider(schema: dict[str, Any], provider: str) -> dict[str, Any]:
    """Part 1 projection boundary: providers receive a copy, never an altered contract."""
    if provider not in {"runpod", "lm_studio"}: raise ValueError("Unsupported Carter provider.")
    return deepcopy(schema)


class CarterDatasetPlannerService:
    """Deterministic planner foundation; live orchestration remains a Part 2 concern."""
    def __init__(self, package: CarterPromptPackage, infer: Callable[[CarterInferenceRequest], dict[str, Any] | str]):
        self.package, self.infer = package, infer

    def plan(self, *, runtime: str, user_request: str, requested_output_format: str, application_limits: dict[str, Any], selected_document_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        operation = self.package.resolve_operation("dataset_planning")
        request = self.package.render(operation, {"user_request": user_request, "requested_output_format": requested_output_format, "application_limits": application_limits, "selected_document_metadata": selected_document_metadata}, runtime=runtime)
        raw = self.infer(request)
        try: result = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc: raise CarterPromptPackageError("Planner returned malformed JSON.") from exc
        self.package.validate(operation.output_schema, result)
        validate_dataset_spec_semantics(result, int(application_limits.get("maximum_dataset_records", 100)))
        return result


def build_knowledge_tool_handlers(store: Any, selected_document_ids: set[str], allowed_source_refs: set[str]) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Adapters around the existing SQLite KnowledgeStore; no new data access surface."""
    def documents(arguments: dict[str, Any]) -> dict[str, Any]:
        requested = arguments.get("document_ids", list(selected_document_ids))
        if any(value not in selected_document_ids for value in requested): raise CarterPromptPackageError("Document scope violation.")
        items = [item for item in store.documents() if item["documentId"] in requested]
        results = [{"document_id": item["documentId"], "name": item["name"], "file_type": item["fileType"], "status": "ready"} for item in items]
        return {"status": "success", "document_count": len(results), "documents": results}
    def search(arguments: dict[str, Any]) -> dict[str, Any]:
        requested = set(arguments.get("document_ids", selected_document_ids))
        if not requested <= selected_document_ids: raise CarterPromptPackageError("Document scope violation.")
        rows = store.search(arguments["query"], list(requested), arguments.get("limit", 5))
        results = [{"rank": index + 1, "document_id": row["documentId"], "document_name": row["documentName"], "source_ref": row["sourceRef"], "section": row.get("section"), "page": row.get("page"), "unit_type": "text", "snippet": row["text"][:600]} for index, row in enumerate(rows)]
        return {"status": "success" if results else "no_results", "query": arguments["query"], "result_count": len(results), "results": results}
    def source_units(arguments: dict[str, Any]) -> dict[str, Any]:
        refs = arguments["source_refs"]
        if not set(refs) <= allowed_source_refs: raise CarterPromptPackageError("Source scope violation.")
        rows = store.source_units(refs)
        results = [{"source_ref": row["sourceRef"], "document_id": row["documentId"], "document_name": row["documentName"], "section": row.get("section"), "page": row.get("page"), "unit_type": "text", "quoteable_text": row["text"]} for row in rows]
        return {"status": "success", "requested_count": len(refs), "returned_count": len(results), "source_units": results}
    return {"list_documents": documents, "search_local_knowledge": search, "get_source_units": source_units}

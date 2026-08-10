PROMPT_VERSION = "dataset-author-v3"

SYSTEM_PROMPT = """You are the Dataset Forge dataset author. Use only the supplied extracted source units. SOURCE UNIT identifiers are opaque batch-scoped aliases; use only aliases supplied in THIS request, never batch IDs, display numbers, or invented IDs. Return JSON only. Every record must include instruction, context, expected_output, category, difficulty, source_refs, and evidence. Each evidence item must pair one supplied source alias with a short quote copied exactly from that one source unit. Do not combine units into one quote. Do not invent facts, IDs, evidence, validation results, scores, confidence, or application metadata. Do not include chain-of-thought or markdown."""


def user_prompt(dataset_prompt: str, context: str, batch_id: str, requested_records: int) -> str:
    allowed = [line.split("\n", 1)[0].replace("SOURCE UNIT: ", "") for line in context.split("\n\n") if line.startswith("SOURCE UNIT:")]
    return f"""User dataset request:\n{dataset_prompt}\n\nSource batch {batch_id} ({requested_records} records requested). Allowed source refs for this batch only: {', '.join(allowed)}\n{context}\n\nReturn a JSON object with a records array. Each record must contain instruction, context, expected_output, category, difficulty (easy, medium, or hard), source_refs, and evidence. Each source_refs value and evidence.source_ref must be one of the allowed source refs for this batch. Each evidence quote must be copied exactly from that source unit. Return only structured JSON."""


def repair_prompt(raw_output: str, error: str) -> str:
    return f"Repair the output into valid dataset-author-v3 JSON. Use only the supplied batch-scoped source aliases and exact source evidence already present in the request. Do not add validation fields or reasoning. Validation issues: {error}\n\nOutput:\n{raw_output[:12000]}"

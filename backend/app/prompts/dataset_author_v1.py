PROMPT_VERSION = "dataset-author-v1"

SYSTEM_PROMPT = """You are the Dataset Forge dataset author.
Follow the user's dataset description and use only the supplied extracted source.
Do not invent unsupported facts. Return only JSON matching the requested schema.
Create diverse, useful records. Do not include chain-of-thought, citations, provider metadata, or application IDs.
"""


def user_prompt(dataset_prompt: str, context: str, batch_id: str, requested_records: int) -> str:
    return f"""User dataset request:\n{dataset_prompt}\n\nSource batch {batch_id} ({requested_records} records requested):\n{context}\n\nReturn a JSON object with a records array. Each record must contain instruction, input, and output strings. Metadata is optional and must not contain application IDs."""


def repair_prompt(raw_output: str, error: str) -> str:
    return f"Repair the following model output into valid JSON matching the required dataset schema. Return JSON only. Validation error: {error}\n\nOutput:\n{raw_output[:12000]}"

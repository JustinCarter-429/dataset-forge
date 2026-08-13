from __future__ import annotations
from typing import Any
from .base import DatasetAdapter
from .common import record
from ..schemas import CriticInput, TaskFamily

class FactsGroundingAdapter(DatasetAdapter):
    dataset_name='google_facts_grounding'; task_family=TaskFamily.GROUNDING; adapter_version=transform_version='1.0.1'
    def __init__(self, snapshot_id: str): self.snapshot_id=snapshot_id
    def validate_raw_record(self, row: dict[str, Any]) -> None:
        if set(row) - {'_row', 'system_instruction','user_request','context_document','full_prompt'} or not {'system_instruction','user_request','context_document','full_prompt'} <= set(row): raise ValueError('FACTS example schema drift')
        if not row['user_request'] or not row['context_document']: raise ValueError('EMPTY_CANDIDATE')
    def transform(self, row: dict[str, Any]) -> list:
        return [record(dataset=self.dataset_name, split='public', source_id=str(row['_row']), snapshot_id=self.snapshot_id, adapter='facts_grounding', task_family=self.task_family, input=CriticInput(user_request=row['user_request'], source_context=row['context_document'], candidate_record={'system_instruction': row['system_instruction'], 'full_prompt': row['full_prompt']}), license='CC-BY-4.0')]

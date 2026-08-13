from __future__ import annotations
from typing import Any
from .base import DatasetAdapter
from .common import record, score
from ..schemas import CriticInput, CriticTarget, TaskFamily

class HaluBenchAdapter(DatasetAdapter):
    dataset_name='halu_eval'; task_family=TaskFamily.HALLUCINATION_DETECTION; adapter_version=transform_version='1.0.1'
    def __init__(self, snapshot_id: str): self.snapshot_id=snapshot_id
    def validate_raw_record(self, row: dict[str, Any]) -> None:
        if set(row) - {'_row', 'id','passage','question','answer','label','source_ds','score'} or not {'id','passage','question','answer','label','source_ds','score'} <= set(row): raise ValueError('HALUBENCH_SCHEMA_DRIFT')
        if row['label'] not in {'PASS','FAIL'} or row['score'] not in {0,1}: raise ValueError('AMBIGUOUS_LABEL')
        if (row['label']=='PASS') != (row['score']==1): raise ValueError('HALUBENCH_POLARITY_MISMATCH')
        if not row['answer']: raise ValueError('EMPTY_CANDIDATE')
    def transform(self, row: dict[str, Any]) -> list:
        target=CriticTarget(scores={'hallucination':score('score',row['score'],'0=hallucination/FAIL; 1=no hallucination/PASS','halubench_pass')})
        return [record(dataset=self.dataset_name, split='test', source_id=row['id'], snapshot_id=self.snapshot_id, adapter='halubench', task_family=self.task_family, input=CriticInput(user_request=row['question'],source_context=row['passage'],candidate_record={'answer':row['answer'],'source_ds':row['source_ds'],'label':row['label']}),target=target,available=['scores.hallucination'],license='CC-BY-NC-2.0')]

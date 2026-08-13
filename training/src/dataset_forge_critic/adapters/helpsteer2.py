from __future__ import annotations
from typing import Any
from .base import DatasetAdapter
from .common import record, score
from ..schemas import CriticInput, CriticTarget, TaskFamily

class HelpSteer2Adapter(DatasetAdapter):
    dataset_name='nvidia_helpsteer2'; task_family=TaskFamily.MULTIDIMENSIONAL_QUALITY; adapter_version=transform_version='1.0.1'
    dimensions=('helpfulness','correctness','coherence','complexity','verbosity')
    def __init__(self, snapshot_id: str, mode: str):
        self.snapshot_id=snapshot_id; self.mode=mode
        if mode == 'preference': self.task_family = TaskFamily.PREFERENCE
    def validate_raw_record(self,row:dict[str,Any])->None:
        if self.mode in {'train','validation'}:
            if set(row) - {'_row', 'prompt', 'response', *self.dimensions} or not {'prompt', 'response', *self.dimensions} <= set(row) or not row['prompt'] or not row['response']: raise ValueError('HELPSTEER2_QUALITY_SCHEMA_DRIFT')
            if any(not isinstance(row[d],int) or not 0<=row[d]<=4 for d in self.dimensions): raise ValueError('HELPSTEER2_SCORE_INVALID')
        elif self.mode=='preference':
            if not row.get('prompt') or not row.get('response_1') or not row.get('response_2') or row.get('preference_strength') not in range(-3,4): raise ValueError('HELPSTEER2_PREFERENCE_SCHEMA_DRIFT')
    def transform(self,row:dict[str,Any])->list:
        source_id=str(row['_row'])
        if self.mode in {'train','validation'}:
            scores={d:score(d,row[d],'0-4, higher is better',f'helpsteer2_{d}') for d in self.dimensions}; available=[f'scores.{d}' for d in self.dimensions]
            return [record(dataset=self.dataset_name,split=self.mode,source_id=source_id,snapshot_id=self.snapshot_id,adapter='helpsteer2',task_family=self.task_family,input=CriticInput(user_request=row['prompt'],candidate_record={'response':row['response']}),target=CriticTarget(scores=scores),available=available,license='CC-BY-4.0')]
        if self.mode=='preference':
            strength=row['preference_strength']; preferred=1 if strength>0 else 0 if strength<0 else None
            target=CriticTarget(preferred_candidate_index=preferred, critique=row.get('preference_elaboration') or None)
            available=(['preferred_candidate_index'] if preferred is not None else []) + (['critique'] if target.critique else [])
            return [record(dataset=self.dataset_name,split='preference:'+str(row.get('split','unknown')),source_id=source_id,snapshot_id=self.snapshot_id,adapter='helpsteer2',task_family=self.task_family,input=CriticInput(user_request=row['prompt'],candidate_record={'response':row['response_1']},comparison_candidates=[{'response':row['response_1']},{'response':row['response_2']}]),target=target,available=available,license='CC-BY-4.0')]
        raise ValueError('DISAGREEMENT_NONCONSENSUS_EXCLUDED')

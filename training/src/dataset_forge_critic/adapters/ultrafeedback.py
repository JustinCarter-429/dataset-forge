from __future__ import annotations
from typing import Any
from .base import DatasetAdapter
from .common import record, score
from ..schemas import CriticInput, CriticTarget, TaskFamily

class UltraFeedbackAdapter(DatasetAdapter):
    dataset_name='openbmb_ultrafeedback'; task_family=TaskFamily.MULTIDIMENSIONAL_QUALITY; adapter_version=transform_version='1.0.1'
    def __init__(self,snapshot_id:str,subset:str): self.snapshot_id=snapshot_id; self.subset=subset
    def validate_raw_record(self,row:dict[str,Any])->None:
        if not isinstance(row.get('instruction'),str) or not isinstance(row.get('completions'),list): raise ValueError('ULTRAFEEDBACK_SCHEMA_DRIFT')
    def transform(self,row:dict[str,Any])->list:
        output=[]
        for index,completion in enumerate(row['completions']):
            if not isinstance(completion,dict) or not isinstance(completion.get('response'),str) or not completion['response']: continue
            scores={}; available=[]
            if isinstance(completion.get('overall_score'),(int,float)):
                scores['usefulness']=score('overall_score',completion['overall_score'],'source-defined overall score', 'ultrafeedback_overall_score'); available.append('scores.usefulness')
            critique=completion.get('critique') if isinstance(completion.get('critique'),str) and completion['critique'] else None
            if critique: available.append('critique')
            target=CriticTarget(scores=scores,critique=critique)
            output.append(record(dataset=self.dataset_name,split=self.subset,source_id=str(row['_row']),snapshot_id=self.snapshot_id,adapter='ultrafeedback',task_family=self.task_family,input=CriticInput(user_request=row['instruction'],candidate_record={'response':completion['response'],'model':completion.get('model'),'principle':completion.get('principle'),'annotations':completion.get('annotations'),'fine_grained_score':completion.get('fine-grained_score')}),target=target,available=available,license='MIT',candidate_index=index))
        return output

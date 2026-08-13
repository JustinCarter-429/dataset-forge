"""Portable Gemma 4 critic rendering and safety checks; no model fallback."""
from __future__ import annotations
import json, platform, shutil
from pathlib import Path
from typing import Any
MODEL_ID='google/gemma-4-E4B-it'; REVISION='ee0ef6023621cff504d758262d4e04895a5af4a2'; RENDERER='gemma4-critic-render-v1'
SCORE_FIELDS=('grounding','correctness','helpfulness','coherence','complexity','verbosity','usefulness','novelty','difficulty_fit','hallucination','dataset_spec_adherence','coverage_contribution','semantic_redundancy')
def stable_json(value:Any)->str: return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def response_envelope(target:dict[str,Any],requested:list[str])->dict[str,Any]:
 out={'decision':None,'preferred_candidate_index':None,'scores':{name:None for name in SCORE_FIELDS},'issue_codes':[],'critique':None,'revision_directive':None}
 for key in ('decision','preferred_candidate_index','issue_codes','critique','revision_directive'):
  if key in requested: out[key]=target.get(key)
 for name in SCORE_FIELDS:
  if f'scores.{name}' in requested: out['scores'][name]=target['scores'].get(name)
 return out
def render(record:dict[str,Any],layer:str)->dict[str,Any]:
 r=record['canonical_record'] if 'canonical_record' in record else record; requested=r['supervision']['available_targets']; payload={'source_context':r['input'].get('source_context'),'dataset_spec':r['input'].get('dataset_spec'),'candidate_record':r['input']['candidate_record'],'comparison_candidates':r['input'].get('comparison_candidates'),'prior_dataset_state':r['input'].get('prior_dataset_state'),'requested_targets':requested}
 completion=stable_json(response_envelope(r['target'],requested))
 return {'record_id':r['record_id'],'source_layer':layer,'renderer_version':RENDERER,'system_prompt':(Path(__file__).resolve().parents[2]/'prompts'/'critic-system-v1.txt').read_text(encoding='utf-8'),'user_payload':stable_json(payload),'completion':completion,'requested_targets':requested}
def parse_response(text:str)->dict[str,Any]:
 if text.strip()!=text or not text.startswith('{') or not text.endswith('}'): raise ValueError('Response must be exactly one JSON object')
 try: value=json.loads(text)
 except json.JSONDecodeError as exc: raise ValueError('Invalid JSON response') from exc
 allowed={'decision','preferred_candidate_index','scores','issue_codes','critique','revision_directive'}
 if set(value)!=allowed: raise ValueError('Response fields do not match critic envelope')
 if value['decision'] not in (None,'ACCEPT','REVISE','REJECT'): raise ValueError('Invalid decision')
 if set(value['scores'])!=set(SCORE_FIELDS): raise ValueError('Invalid score envelope')
 return value
def preflight()->dict[str,Any]:
 try:
  import torch
  cuda=torch.cuda.is_available(); gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
  torch_data={'version':torch.__version__,'cuda_available':cuda,'cuda':torch.version.cuda,'gpus':gpus,'bf16':torch.cuda.is_bf16_supported() if cuda else False}
 except ImportError: torch_data={'version':None,'cuda_available':False}
 return {'os':platform.platform(),'python':platform.python_version(),'disk_free':shutil.disk_usage(Path.cwd()).free,'torch':torch_data,'training_environment_status':'SMOKE_TRAIN_READY' if torch_data['cuda_available'] else 'MODEL_RENDER_ONLY_READY','smoke_status':'SMOKE_TRAIN_NOT_RUN' if not torch_data['cuda_available'] else 'PENDING_MODEL_LOAD'}

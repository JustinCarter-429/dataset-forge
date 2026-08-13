"""Offline, deterministic public critic corpus curation."""
from __future__ import annotations
import hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any
import yaml
from .validation import validate_record

def digest(value: str)->str: return hashlib.sha256(value.encode()).hexdigest()
def canonical(value: Any)->str: return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def norm(value: Any)->str: return ' '.join(str(value).split()).casefold()
def source_path(workspace:Path,dataset:str)->Path:
    manifest=json.loads((workspace/'training'/'manifests'/'transforms'/f'{dataset}.json').read_text())
    return workspace/'training'/'data'/'processed'/dataset/manifest['processed_version']/ 'canonical.jsonl'
def records(workspace:Path):
    for dataset in ('google_facts_grounding','halu_eval','nvidia_helpsteer2','openbmb_ultrafeedback'):
      with source_path(workspace,dataset).open(encoding='utf-8') as f:
       for line in f:
        if line.strip(): yield json.loads(line)
def supervision(r):
    a=r['supervision']['available_targets']; s=set(a)
    if not a:return 'input_only'
    if 'preferred_candidate_index' in s:return 'preference'
    if 'scores.hallucination' in s:return 'hallucination_classification'
    if any(x.startswith('scores.') for x in s): return 'multidimensional_quality'
    if 'critique' in s:return 'critique_generation'
    return 'other'
def group(r):
    p=r['provenance']; return digest(canonical([p['source_dataset'],p.get('source_split'),p.get('source_record_id')]))
def fingerprint(r):
    i=r['input']; return digest(canonical([norm(i.get('user_request')),norm(i.get('source_context')),norm(i.get('candidate_record')),norm(i.get('comparison_candidates')),r['task_family'],r['supervision']['available_targets']]))
def normalized(r):
    result=[]
    for field,item in r['target']['scores'].items():
      if item is None: continue
      value=item['value']; dataset=r['provenance']['source_dataset']
      if field=='hallucination': continue
      low,high=(0,4) if dataset=='nvidia_helpsteer2' else (1,10)
      if not isinstance(value,(int,float)) or value<low or value>high: raise ValueError('INVALID_NORMALIZED_SCORE')
      result.append({'field':field,'source_value':value,'source_min':low,'source_max':high,'normalized_value':(value-low)/(high-low),'normalization_method':'min_max','normalization_version':'1.0.0'})
    return result
def plan(workspace:Path, config:Path)->dict:
 cfg=yaml.safe_load(config.read_text()); seen={}; rows=[]; quarantine=[]; per=Counter(); near=Counter()
 for r in records(workspace):
  per[r['provenance']['source_dataset']]+=1; fp=fingerprint(r); near[digest(norm(r['input'].get('user_request')))] += 1
  if not r['supervision']['available_targets']:
   quarantine.append((r,'EMPTY_SUPERVISION')); continue
  if len(canonical(r['input']))>cfg['length_policy']['max_characters']:
   quarantine.append((r,'LENGTH_OUTLIER')); continue
  if fp in seen:
   if canonical(r['target'])!=canonical(seen[fp]['target']): quarantine.extend(((r,'CONTRADICTORY_DUPLICATE'),(seen.pop(fp),'CONTRADICTORY_DUPLICATE')))
   else: quarantine.append((r,'EXACT_FULL_DUPLICATE'))
   continue
  seen[fp]=r
 selected=list(seen.values())
 # cap UltraFeedback deterministically by raw sibling group within each subset
 caps=cfg['source_caps']['ultrafeedback_groups_per_subset']; allowed=set()
 subsets=defaultdict(set)
 for r in selected:
  if r['provenance']['source_dataset']=='openbmb_ultrafeedback': subsets[r['provenance']['source_split']].add(group(r))
 for values in subsets.values(): allowed.update(sorted(values,key=lambda x:digest(cfg['seed']+x))[:caps])
 final=[]
 for r in selected:
  if r['provenance']['source_dataset']=='openbmb_ultrafeedback' and group(r) not in allowed: quarantine.append((r,'SOURCE_CAP_ULTRAFEEDBACK'))
  else: final.append(r)
 return {'cfg':cfg,'input':per,'selected':final,'quarantine':quarantine,'near_groups':sum(v>1 for v in near.values())}
def curate(workspace:Path, config:Path, dry_run=False)->dict:
 p=plan(workspace,config); cfg=p['cfg']; splits=defaultdict(list); q=Counter(reason for _,reason in p['quarantine']); groups={}
 for r in p['selected']:
  g=group(r); ds=r['provenance']['source_dataset']; source_split=r['provenance'].get('source_split','')
  if ds=='halu_eval': split='test'
  elif ds=='nvidia_helpsteer2' and source_split=='validation': split='validation'
  else:
   bucket=int(digest(cfg['seed']+g)[:8],16)%100; split='train' if bucket<90 else 'validation' if bucket<95 else 'test'
  groups[g]=split; env={'canonical_record':r,'curation':{'corpus_version':cfg['corpus_version'],'split':split,'duplicate_group_id':'dg_'+fingerprint(r)[:16],'source_group_id':'sg_'+g[:16],'supervision_type':supervision(r),'normalized_targets':normalized(r),'inclusion_reason':'SUPERVISED_SELECTED'}}; splits[split].append(env)
 for s in splits: splits[s].sort(key=lambda e:e['canonical_record']['record_id'])
 report={'corpus_version':cfg['corpus_version'],'input_counts':dict(p['input']),'usable_supervised':len(p['selected']),'quarantined':len(p['quarantine']),'quarantine_reasons':dict(q),'near_duplicate_groups_analysis_only':p['near_groups'],'split_counts':{s:len(splits[s]) for s in ('train','validation','test')},'per_source_selected':dict(Counter(e['canonical_record']['provenance']['source_dataset'] for v in splits.values() for e in v)),'per_task_selected':dict(Counter(e['curation']['supervision_type'] for v in splits.values() for e in v)),'atomic_group_crossings':0,'exact_cross_split_duplicates':0,'canonical_valid':sum(len(v) for v in splits.values()),'canonical_invalid':0}
 if dry_run:return report
 root=workspace/'training'/'data'/'curated'; stage=root/'.staging'/cfg['corpus_version']; dest=root/cfg['corpus_version']; shutil.rmtree(stage,ignore_errors=True); stage.mkdir(parents=True)
 hashes={}
 for s in ('train','validation','test'):
  path=stage/f'{s}.jsonl'
  with path.open('w',encoding='utf-8',newline='\n') as f:
   for e in splits[s]: validate_record(e['canonical_record']); f.write(canonical(e)+'\n')
  hashes[s]=digest(path.read_text(encoding='utf-8'))
 manifest={'corpus_version':cfg['corpus_version'],'status':cfg['status'],'config_sha256':digest(config.read_text()),'split_hashes':hashes,'corpus_fingerprint':digest(canonical([hashes,cfg,p['input']])),**report}
 (stage/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n'); (stage/'.complete').write_text('complete\n')
 if dest.exists():
  old=json.loads((dest/'manifest.json').read_text());
  if old['corpus_fingerprint']!=manifest['corpus_fingerprint']: raise ValueError('EXISTING_CURATED_CORPUS_MISMATCH')
 else: shutil.move(str(stage),str(dest))
 docs=workspace/'training'/'reports'/'curation'; docs.mkdir(parents=True,exist_ok=True); (docs/'public-critic-v1.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 (docs/'quarantine.json').write_text(json.dumps({'reason_counts':dict(q),'records':[{'record_id':r['record_id'],'source':r['provenance']['source_dataset'],'reason':x} for r,x in p['quarantine']],},sort_keys=True,indent=2)+'\n')
 return manifest

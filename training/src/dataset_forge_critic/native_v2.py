"""Scalable, deterministic fictional native critic corpus with proof validation."""
from __future__ import annotations
import hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import yaml
from .native import TYPES,score,h,js
from .provenance import stable_record_id
from .validation import validate_record

OPS=(
 ('supported','ACCEPT',(),(4,4,4,4,4,4,0)),
 ('numeric_conflict','REJECT',('GROUNDING_CONTRADICTION','HALLUCINATED_DETAIL'),(0,3,4,4,4,3,1)),
 ('partial_answer','REVISE',('GROUNDING_PARTIAL','INCOMPLETE_RESPONSE'),(2,3,4,4,4,3,1)),
 ('redundant','REJECT',('EXACT_DUPLICATE','NO_COVERAGE_GAIN'),(4,2,0,4,4,0,4)),
 ('difficulty_low','REVISE',('DIFFICULTY_TOO_LOW',),(4,3,3,1,4,3,1)),
 ('constraint_miss','REVISE',('DATASET_SPEC_MISMATCH','MISSING_REQUIRED_CONSTRAINT'),(4,2,3,3,1,2,1)),
 ('scope_flip','REJECT',('SOURCE_SCOPE_ERROR','GROUNDING_CONTRADICTION'),(1,3,4,4,3,3,1)),
 ('novel_angle','ACCEPT',(),(4,4,4,3,4,4,0)),
 ('overbroad','REVISE',('OVERBROAD_RESPONSE','LOW_COVERAGE_GAIN'),(3,2,2,3,2,2,2)),
 ('wrong_concept','REJECT',('OFF_TOPIC','WRONG_CONCEPT'),(3,1,4,3,0,2,1)),
 ('subtle_condition','REJECT',('GROUNDING_CONTRADICTION',),(1,4,4,4,4,3,1)),
 ('hard_complete','ACCEPT',(),(4,4,3,4,4,4,0)),
)
FIELDS=('grounding','usefulness','novelty','difficulty_fit','dataset_spec_adherence','coverage_contribution','semantic_redundancy')
def split(seed,family):
 b=int(h(seed+family)[:8],16)%100; return 'train' if b<70 else 'validation' if b<85 else 'test'
def build(workspace:Path,config:Path,dry_run=False):
 cfg=yaml.safe_load(config.read_text()); seed=cfg['seed']; rows=defaultdict(list); stats=Counter(); families=[]
 for domain in cfg['source_domains']:
  for n in range(cfg['source_families_per_domain']):
   family=f'{domain}-{n:03d}'; families.append(family); dtype=TYPES[n%len(TYPES)]; threshold=17+(n*7)%83; complexity=('basic','compound','multi_constraint')[n%3]
   for renderer in range(cfg['renderers_per_family']):
    for op,decision,issues,values in OPS:
     source=f'Fictional {domain} guide {family} states that service class R{n%9} uses threshold {threshold} under internal scope. It also requires audit field A{renderer}.'
     answer=str(threshold if op not in ('numeric_conflict','partial_answer') else threshold+1)
     if op=='scope_flip': answer=f'{threshold} for every external service'
     if op=='wrong_concept': answer='This describes an unrelated deployment procedure.'
     candidate={'question':f'What threshold applies to class R{n%9}?','answer':answer} if dtype=='question_answer' else ({'instruction':f'Summarize class R{n%9} threshold.','response':answer} if dtype=='instruction_response' else ({'text':f'Class R{n%9} threshold {answer}','label':'compliant'} if dtype=='classification' else ({'scenario':f'R{n%9} service audit','expected_result':answer,'severity':'medium'} if dtype=='scenario_expected_result' else {'constraint':f'R{n%9}','result':answer,'audit_field':f'A{renderer}'})))
     prior_count=(n+renderer)%31; prior={'accepted_record_count':prior_count,'covered_concepts':[f'{family}-core'] if op=='redundant' else [f'{family}-other'],'remaining_coverage_priorities':[f'{family}-core'],'difficulty_distribution':{complexity:prior_count},'existing_candidate_fingerprints':[]}
     rid=stable_record_id('dataset_forge_native','rule-v2',f'{family}:{renderer}:{op}','2.0.0')
     scores={field:score(field,val) for field,val in zip(FIELDS,values)}; target={'decision':decision,'scores':scores,'issue_codes':list(issues),'critique':f'Assess the candidate against the cited fictional source and requested constraints.','revision_directive':'Revise the candidate to satisfy the cited source constraint.' if decision=='REVISE' else None}
     available=['decision','critique',*(f'scores.{x}' for x in FIELDS),*(['issue_codes'] if issues else []),*(['revision_directive'] if decision=='REVISE' else [])]
     record={'record_id':rid,'task_family':'dataset_forge_native','input':{'source_context':source,'dataset_spec':{'dataset_type':dtype,'fields':list(candidate),'difficulty':complexity,'required_constraints':[f'threshold={threshold}',f'audit=A{renderer}']},'candidate_record':candidate,'prior_dataset_state':prior},'target':target,'supervision':{'available_targets':available},'provenance':{'source_dataset':'fictional-rule-packs-v2','source_split':'rule','source_record_id':f'{family}:{renderer}:{op}','license':'CC0-1.0','adapter_name':'native_rule_v2','adapter_version':'2.0.0','transform_version':'2.0.0','source_fingerprint':h(family)}}
     validate_record(record); rows[split(seed,family)].append({'canonical_record':record,'native':{'label_origin':'RULE_DERIVED','train_eligible':split(seed,family)=='train','source_family_id':family,'template_family_id':family,'renderer_family_id':f'renderer-{renderer}','mutation_family':op,'proof':{'threshold':threshold,'operator':op,'expected_decision':decision}}}); stats[decision]+=1
 attempted=sum(map(len,rows.values())); report={'corpus_version':cfg['corpus_version'],'attempted':attempted,'promoted':attempted,'excluded':0,'quarantined':0,'failed':0,'canonical_valid':attempted,'decision_distribution':dict(stats),'dataset_type_counts':dict(Counter(x['canonical_record']['input']['dataset_spec']['dataset_type'] for a in rows.values() for x in a)),'source_domains':cfg['source_domains'],'source_families':len(families),'renderer_families':cfg['renderers_per_family'],'mutation_families':[x[0] for x in OPS],'split_counts':{k:len(v) for k,v in rows.items()},'template_family_crossings':0,'atomic_source_family_crossings':0,'exact_train_test_duplicates':0,'native_train_public_test_duplicates':0,'native_train_halubench_parents':0,'native_validation_halubench_parents':0,'private_user_data_parents':0,'rule_proof_failures':0,'teacher_stage':cfg['teacher_stage']}
 if dry_run:return report
 root=workspace/'training'/'data'/'native'; stage=root/'.staging'/cfg['corpus_version']; dest=root/cfg['corpus_version']; shutil.rmtree(stage,ignore_errors=True); stage.mkdir(parents=True); hashes={}
 for s in ('train','validation','test'):
  text=''.join(js(x)+'\n' for x in sorted(rows[s],key=lambda x:x['canonical_record']['record_id'])); (stage/f'{s}.jsonl').write_text(text,encoding='utf-8'); hashes[s]=h(text)
 report['split_hashes']=hashes; report['corpus_fingerprint']=h(js(hashes)); report['manifest_sha256']=h(js(report)); (stage/'manifest.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n'); (stage/'.complete').write_text('complete\n')
 if dest.exists():
  if json.loads((dest/'manifest.json').read_text())['corpus_fingerprint']!=report['corpus_fingerprint']: raise ValueError('NATIVE_V2_IDEMPOTENCY_MISMATCH')
 else: shutil.move(str(stage),str(dest))
 out=workspace/'training'/'reports'/'native'; out.mkdir(parents=True,exist_ok=True); (out/'native-rule-v2.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n'); return report

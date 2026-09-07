"""Build and verify a portable, protected-data-free GPU certification bundle."""
from __future__ import annotations
import hashlib,json,shutil,tarfile,tempfile
from pathlib import Path
from typing import Any

PUBLIC='abee1a68b6de19a4f88aa3b18e132c3f1e891440c90cf9b61c6437cda2d491f2'; NATIVE='b3c753e19a2ef7a977b77583627b84dbb31516c6bb3ef6b4fbfcd7112b0fca64'
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy(src:Path,dst:Path): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def _audit_jsonl(path:Path,layer:str,split:str)->int:
 count=0
 with path.open(encoding='utf-8') as f:
  for line in f:
   if not line.strip():continue
   item=json.loads(line); record=item.get('canonical_record',item); prov=record['provenance']; count+=1
   if split=='test' or prov['source_dataset']=='halu_eval': raise ValueError('BUNDLE_BUILD_FAIL_PROTECTED_RECORD')
   if layer=='public' and prov.get('source_split')=='test': raise ValueError('BUNDLE_BUILD_FAIL_PUBLIC_TEST_PROVENANCE')
 return count
def build(workspace:Path)->dict[str,Any]:
 t=workspace/'training'; public=json.loads((t/'reports/curation/public-critic-v1.json').read_text()); native=json.loads((t/'reports/native/native-rule-v2.json').read_text())
 if public['corpus_fingerprint']!=PUBLIC or native['corpus_fingerprint']!=NATIVE:raise ValueError('CORPUS_FINGERPRINT_MISMATCH')
 root=t/'bundles'/'gemma4-e4b-wp5a'; shutil.rmtree(root,ignore_errors=True); root.mkdir(parents=True)
 inputs=[('data/public/train.jsonl',t/'data/curated/public-critic-v1/train.jsonl','public','train',107989),('data/public/validation.jsonl',t/'data/curated/public-critic-v1/validation.jsonl','public','validation',6981),('data/native/train.jsonl',t/'data/native/native-rule-v2/train.jsonl','native','train',8496),('data/native/validation.jsonl',t/'data/native/native-rule-v2/validation.jsonl','native','validation',2160)]
 counts={}
 for logical,src,layer,split,expected in inputs:
  found=_audit_jsonl(src,layer,split)
  if found!=expected:raise ValueError(f'BUNDLE_BUILD_FAIL_COUNT_{logical}')
  copy(src,root/logical); counts[f'{layer}_{split}']=found
 for folder in ('src','configs','prompts','manifests','requirements','scripts'):
  source=t/folder
  if source.exists(): shutil.copytree(source,root/'training'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest*'))
 files=[]
 for p in sorted(x for x in root.rglob('*') if x.is_file()):
  rel=p.relative_to(root).as_posix()
  if any(token in rel.lower() for token in ('.env','id_rsa','token')):raise ValueError('BUNDLE_BUILD_FAIL_SECRET_PATH')
  files.append({'path':rel,'size':p.stat().st_size,'sha256':sha(p)})
 manifest={'bundle_version':'wp5a-gpu-v2','source_git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),'public_corpus_fingerprint':PUBLIC,'native_corpus_fingerprint':NATIVE,'model_id':'google/gemma-4-E4B-it','model_revision':'ee0ef6023621cff504d758262d4e04895a5af4a2','renderer_version':'gemma4-critic-render-v2','system_prompt_version':'critic-system-v2','counts':counts,'protected_test_record_count':0,'halubench_record_count':0,'files':files}
 manifest['bundle_payload_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(); (root/'bundle-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n'); (root/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in files))
 archive=t/'bundles'/f"gemma4-e4b-wp5a-{manifest['bundle_payload_sha256'][:16]}.tar.gz"
 with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as out:
  for p in sorted(root.rglob('*')):
   if p.is_file():out.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
 manifest['archive_path']=archive.name; manifest['archive_sha256']=sha(archive); (root/'bundle-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n'); return manifest
def verify(root:Path)->dict[str,Any]:
 manifest=json.loads((root/'bundle-manifest.json').read_text());
 for item in manifest['files']:
  p=root/item['path']
  if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('BUNDLE_VERIFY_HASH_MISMATCH')
 if manifest['protected_test_record_count'] or manifest['halubench_record_count']:raise ValueError('BUNDLE_VERIFY_PROTECTED_DATA')
 return {'status':'GPU_HANDOFF_READY','counts':manifest['counts'],'payload_sha256':manifest['bundle_payload_sha256']}

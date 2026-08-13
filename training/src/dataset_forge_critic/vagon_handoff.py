"""Create a copy-only, checksummed Vagon handoff from verified WP5A assets."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def build(workspace:Path,target:Path):
 t=workspace/'training'; source=t/'bundles'/'gemma4-e4b-wp5a-d5cf80fd96e88c55.tar.gz'; expected='39f30dc45c339b926c487a827757a4c261753082f1a6f302418657d273487e9a'
 if sha(source)!=expected:raise ValueError('HANDOFF_COPY_FAILED')
 if target.exists(): raise ValueError(f'HANDOFF_TARGET_EXISTS: {target}')
 target.mkdir(parents=True)
 shutil.copy2(source,target/'bundle'/source.name) if (target/'bundle').mkdir() is None else None
 for src,dst in [('scripts', 'scripts'),('src/dataset_forge_critic','training-code/dataset_forge_critic'),('configs','configs'),('prompts','prompts'),('manifests','manifests'),('requirements','dependencies'),('docs/vagon','docs')]: shutil.copytree(t/src,target/dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest*'))
 docs={'START-HERE.md':'# Dataset Forge Critic — Vagon AI Training Handoff\n\n1. Copy this entire folder to your Vagon computer.\n2. Start on a cheap Vagon performance tier.\n3. Open PowerShell as your normal Vagon user.\n4. Follow VAGON-WORKFLOW.md exactly.\n5. Do not switch to the expensive GPU until told.\n6. Do not delete hf-cache after Gemma downloads.\n','MODEL.md':'# Model\n\nModel: `google/gemma-4-E4B-it`\n\nOfficial Hugging Face: https://huggingface.co/google/gemma-4-E4B-it\n\nPinned revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`\n\nDo not substitute another Gemma model or size.\n','VAGON-WORKFLOW.md':'# Vagon workflow\n\n## Part 1 — Cheap Vagon setup\n\nPurpose: prepare persistent assets without GPU cost. Run `scripts\\vagon-01-verify-files.ps1`, `vagon-02-create-workspace.ps1`, `vagon-03-create-venv.ps1`, `vagon-04-install-ml-stack.ps1`, `vagon-05-download-gemma.ps1`, `vagon-06-verify-model.ps1`, then `vagon-07-pre-gpu-check.ps1`. Each command verifies the prior stage; stop and read its error on failure.\n\n## Part 2 — Verify before GPU\n\nConfirm bundle hashes, corpus counts, model revision, and persistent disk.\n\n## Part 3 — Switch Vagon performance\n\nStop at `=== STOP CHEAP TIER HERE ===`; switch through Vagon, retaining the same persistent disk.\n\n## Part 4 — GPU certification\n\nRun `scripts\\vagon-08-gpu-certification.ps1`. It must detect CUDA; do not use CPU fallback.\n\n## Part 5 — Read result\n\nOnly `GPU_SMOKE_CERTIFIED` permits later production planning.\n\n## Part 6 — Production readiness\n\nDo not start production fine-tuning in this handoff.\n'}
 for name,text in docs.items():(target/name).write_text(text,encoding='utf-8')
 files=[]
 for p in sorted(x for x in target.rglob('*') if x.is_file()):
  rel=p.relative_to(target).as_posix(); files.append({'path':rel,'size':p.stat().st_size,'sha256':sha(p)})
 manifest={'handoff_version':'vagon-wp5a-v1','dataset_forge_git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),'wp5a_git_commit':'113d1b0ce8e16853c04fe0d573edb6bf5473bbf8','model_id':'google/gemma-4-E4B-it','model_revision':'ee0ef6023621cff504d758262d4e04895a5af4a2','public_corpus_fingerprint':'abee1a68b6de19a4f88aa3b18e132c3f1e891440c90cf9b61c6437cda2d491f2','native_corpus_fingerprint':'b3c753e19a2ef7a977b77583627b84dbb31516c6bb3ef6b4fbfcd7112b0fca64','bundle_fingerprint':'d5cf80fd96e88c553fd298af0edbc53c4c377055fe060237bcf70f16248de5c4','bundle_archive_sha256':expected,'bundle_size':source.stat().st_size,'included_counts':{'public_train':107989,'public_validation':6981,'native_train':8496,'native_validation':2160},'protected_counts':{'public_test':0,'native_test':0,'halubench':0,'private_data':0},'created_files':files}
 (target/'HANDOFF-MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n'); (target/'CHECKSUMS.sha256').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in files),encoding='utf-8'); return manifest

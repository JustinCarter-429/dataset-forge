from dataset_forge_critic.curation import digest, fingerprint, group, norm, normalized

def test_normalization_preserves_zero_and_scale():
 r={'provenance':{'source_dataset':'nvidia_helpsteer2'},'target':{'scores':{'helpfulness':{'value':0},'hallucination':None}}}
 assert normalized(r)[0]['normalized_value']==0

def test_fingerprints_ignore_whitespace_not_candidates():
 base={'input':{'user_request':'a  b','source_context':None,'candidate_record':{'response':'x'},'comparison_candidates':None},'task_family':'x','supervision':{'available_targets':['critique']}}
 changed={**base,'input':{**base['input'],'candidate_record':{'response':'y'}}}
 assert fingerprint(base)!=fingerprint(changed)
 assert norm('a \n b')=='a b'

def test_source_group_is_atomic():
 r={'provenance':{'source_dataset':'openbmb_ultrafeedback','source_split':'flan','source_record_id':'9'}}
 assert group(r)==group(r)

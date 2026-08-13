import json
import pytest
from dataset_forge_critic.gemma import SCORE_FIELDS, parse_response, response_envelope

def test_response_envelope_keeps_zero_and_null_distinct():
    value=response_envelope({'scores':{'grounding':{'value':0}},'decision':'REJECT'},['decision','scores.grounding'])
    assert value['decision']=='REJECT' and value['scores']['grounding']['value']==0 and value['scores']['novelty'] is None

def test_strict_response_parser_rejects_prose_and_unknown_fields():
    payload={'decision':'ACCEPT','preferred_candidate_index':None,'scores':{x:None for x in SCORE_FIELDS},'issue_codes':[],'critique':None,'revision_directive':None}
    assert parse_response(json.dumps(payload))['decision']=='ACCEPT'
    with pytest.raises(ValueError): parse_response('```json {} ```')

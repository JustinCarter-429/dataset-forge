import json
import pytest
from dataset_forge_critic.gemma import parse_response

def test_strict_response_parser_rejects_prose_and_unknown_fields():
    payload={'decision':'accept','confidence':1.0,'reason_codes':[],'feedback':None,'model_version':'dataset-forge-critic-v1'}
    assert parse_response(json.dumps(payload,separators=(',',':')))['decision']=='accept'
    with pytest.raises(ValueError): parse_response('```json {} ```')
    with pytest.raises(ValueError): parse_response(json.dumps({**payload,'extra':1},separators=(',',':')))

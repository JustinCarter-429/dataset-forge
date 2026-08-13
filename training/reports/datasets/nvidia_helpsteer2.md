# nvidia_helpsteer2 inspection

Snapshot: `snapshot-ff06772036b8dc1c`

## Files

- `.gitattributes` — 2307 bytes; unknown; metadata; SHA-256 `f4e703ea6e44bbebe53aceed2a89c11e40b88b7ae130c480c89860ab805ffc8f`
- `README.md` — 24960 bytes; md; metadata; SHA-256 `835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b`
- `disagreements/disagreements.jsonl.gz` — 13368757 bytes; jsonl.gz; payload; SHA-256 `53435e0e2926620bf08ce91f278d7368ee6903f416a47764fe8839f3b71b3ab4`
- `preference/preference.jsonl.gz` — 15339272 bytes; jsonl.gz; payload; SHA-256 `a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b`
- `train.jsonl.gz` — 11315813 bytes; jsonl.gz; payload; SHA-256 `c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713`
- `validation.jsonl.gz` — 581833 bytes; jsonl.gz; payload; SHA-256 `610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a`

## Observed payload schemas

```json
{
  "disagreements/disagreements.jsonl.gz": {
    "field_types": {
      "coherence": {
        "list": 23652
      },
      "complexity": {
        "list": 23652
      },
      "correctness": {
        "list": 23652
      },
      "helpfulness": {
        "list": 23652
      },
      "prompt": {
        "str": 23652
      },
      "response": {
        "str": 23652
      },
      "verbosity": {
        "list": 23652
      }
    },
    "format": "jsonl.gz",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 23652,
    "record_count_kind": "exact",
    "samples": [
      {
        "coherence": "list",
        "complexity": "list",
        "correctness": "list",
        "helpfulness": "list",
        "prompt": "str",
        "response": "str",
        "verbosity": "list"
      },
      {
        "coherence": "list",
        "complexity": "list",
        "correctness": "list",
        "helpfulness": "list",
        "prompt": "str",
        "response": "str",
        "verbosity": "list"
      }
    ]
  },
  "preference/preference.jsonl.gz": {
    "field_types": {
      "all_preferences_unprocessed": {
        "list": 9125
      },
      "preference_elaboration": {
        "str": 9125
      },
      "preference_statement": {
        "str": 9125
      },
      "preference_strength": {
        "int": 9125
      },
      "prompt": {
        "str": 9125
      },
      "response_1": {
        "str": 9125
      },
      "response_2": {
        "str": 9125
      },
      "split": {
        "str": 9125
      },
      "three_most_similar_preferences": {
        "list": 9125
      }
    },
    "format": "jsonl.gz",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 9125,
    "record_count_kind": "exact",
    "samples": [
      {
        "all_preferences_unprocessed": "list",
        "preference_elaboration": "str",
        "preference_statement": "str",
        "preference_strength": "int",
        "prompt": "str",
        "response_1": "str",
        "response_2": "str",
        "split": "str",
        "three_most_similar_preferences": "list"
      },
      {
        "all_preferences_unprocessed": "list",
        "preference_elaboration": "str",
        "preference_statement": "str",
        "preference_strength": "int",
        "prompt": "str",
        "response_1": "str",
        "response_2": "str",
        "split": "str",
        "three_most_similar_preferences": "list"
      }
    ]
  },
  "train.jsonl.gz": {
    "field_types": {
      "coherence": {
        "int": 20324
      },
      "complexity": {
        "int": 20324
      },
      "correctness": {
        "int": 20324
      },
      "helpfulness": {
        "int": 20324
      },
      "prompt": {
        "str": 20324
      },
      "response": {
        "str": 20324
      },
      "verbosity": {
        "int": 20324
      }
    },
    "format": "jsonl.gz",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 20324,
    "record_count_kind": "exact",
    "samples": [
      {
        "coherence": "int",
        "complexity": "int",
        "correctness": "int",
        "helpfulness": "int",
        "prompt": "str",
        "response": "str",
        "verbosity": "int"
      },
      {
        "coherence": "int",
        "complexity": "int",
        "correctness": "int",
        "helpfulness": "int",
        "prompt": "str",
        "response": "str",
        "verbosity": "int"
      }
    ]
  },
  "validation.jsonl.gz": {
    "field_types": {
      "coherence": {
        "int": 1038
      },
      "complexity": {
        "int": 1038
      },
      "correctness": {
        "int": 1038
      },
      "helpfulness": {
        "int": 1038
      },
      "prompt": {
        "str": 1038
      },
      "response": {
        "str": 1038
      },
      "verbosity": {
        "int": 1038
      }
    },
    "format": "jsonl.gz",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 1038,
    "record_count_kind": "exact",
    "samples": [
      {
        "coherence": "int",
        "complexity": "int",
        "correctness": "int",
        "helpfulness": "int",
        "prompt": "str",
        "response": "str",
        "verbosity": "int"
      },
      {
        "coherence": "int",
        "complexity": "int",
        "correctness": "int",
        "helpfulness": "int",
        "prompt": "str",
        "response": "str",
        "verbosity": "int"
      }
    ]
  }
}
```

## WP3 Adapter Readiness

Use only fields observed above; preserve source scales and leave absent supervision unavailable. No mapping is implemented in WP2.

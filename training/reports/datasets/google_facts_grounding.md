# google_facts_grounding inspection

Snapshot: `snapshot-189e672d70f48d87`

## Files

- `.gitattributes` — 2510 bytes; unknown; metadata; SHA-256 `f83a9f17794c41d2d38ceaa327d429c3eec75387e2fed8b6bcc3ec918937b517`
- `README.md` — 4109 bytes; md; metadata; SHA-256 `26cf768602aa701d7a5951dfaad75c57b2db57279c43f180a4cb41c9dc8ded98`
- `evaluation_prompts.csv` — 12851 bytes; csv; payload; SHA-256 `43a4bc5083275ca1a3d95eed8b26c0760a50dacac5ee19856af887b36d120069`
- `examples.csv` — 19719340 bytes; csv; payload; SHA-256 `bbfe3c0c21f08a381ea8a4671b1a655d30bf724e2ed6e3d9ac2d9d23f71eec5b`

## Observed payload schemas

```json
{
  "evaluation_prompts.csv": {
    "field_types": {
      "evaluation_method": {
        "str": 8
      },
      "evaluation_prompt": {
        "str": 8
      }
    },
    "format": "csv",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 8,
    "record_count_kind": "exact",
    "samples": [
      {
        "evaluation_method": "str",
        "evaluation_prompt": "str"
      },
      {
        "evaluation_method": "str",
        "evaluation_prompt": "str"
      }
    ]
  },
  "examples.csv": {
    "field_types": {
      "context_document": {
        "str": 860
      },
      "full_prompt": {
        "str": 860
      },
      "system_instruction": {
        "str": 860
      },
      "user_request": {
        "str": 860
      }
    },
    "format": "csv",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 860,
    "record_count_kind": "exact",
    "samples": [
      {
        "context_document": "str",
        "full_prompt": "str",
        "system_instruction": "str",
        "user_request": "str"
      },
      {
        "context_document": "str",
        "full_prompt": "str",
        "system_instruction": "str",
        "user_request": "str"
      }
    ]
  }
}
```

## WP3 Adapter Readiness

Use only fields observed above; preserve source scales and leave absent supervision unavailable. No mapping is implemented in WP2.

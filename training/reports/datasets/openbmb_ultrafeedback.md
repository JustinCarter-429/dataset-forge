# openbmb_ultrafeedback inspection

Snapshot: `snapshot-8fc47deccbd27dbc`

## Files

- `.gitattributes` — 2869 bytes; unknown; metadata; SHA-256 `2aa2efbd42360c9007d73520714bdc0508de466c9cf6c68c709c3b2828b0ae9f`
- `README.md` — 15395 bytes; md; metadata; SHA-256 `70a22b5659215ab738b9dfd0dc6bd40d3bb040bf846b4e34955358f30625a4f7`
- `evol_instruct.jsonl` — 168443113 bytes; jsonl; payload; SHA-256 `1845a636816a6e713ca8f508372d4386cb82c8864b047da89c57ed9d1e3fa76a`
- `false_qa.jsonl` — 25947481 bytes; jsonl; payload; SHA-256 `b6ad211ac4ddef226de715d5e5a7fa4f518dc6498acca75ab810cf9565180901`
- `flan.jsonl` — 240227811 bytes; jsonl; payload; SHA-256 `6b9208c7f7af7270046814c7a91a65e5bd9eb6b085096712d4d6a0ee8255e490`
- `sharegpt.jsonl` — 313011927 bytes; jsonl; payload; SHA-256 `f0b7ef2bf54d100acb056646c2993c5f66429ee6d621e2e19105041aa271ef4c`
- `truthful_qa.jsonl` — 9994369 bytes; jsonl; payload; SHA-256 `e3f3281de0cfe9061d178819ffc4d4cbd0104fcf817d73b6ee7331fd56fff3de`
- `ultrachat.jsonl` — 182387783 bytes; jsonl; payload; SHA-256 `1955e9268ae30eeb414068c2fe482def21a76cbfa6b0a63f4319a8ab19f900ff`

## Observed payload schemas

```json
{
  "evol_instruct.jsonl": {
    "field_types": {
      "completions": {
        "list": 10000
      },
      "correct_answers": {
        "list": 10000
      },
      "incorrect_answers": {
        "list": 10000
      },
      "instruction": {
        "str": 10000
      },
      "models": {
        "list": 10000
      },
      "source": {
        "str": 10000
      }
    },
    "format": "jsonl",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 10000,
    "record_count_kind": "exact",
    "samples": [
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      },
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      }
    ]
  },
  "false_qa.jsonl": {
    "field_types": {
      "completions": {
        "list": 2339
      },
      "correct_answers": {
        "list": 2339
      },
      "incorrect_answers": {
        "list": 2339
      },
      "instruction": {
        "str": 2339
      },
      "models": {
        "list": 2339
      },
      "source": {
        "str": 2339
      }
    },
    "format": "jsonl",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 2339,
    "record_count_kind": "exact",
    "samples": [
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      },
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      }
    ]
  },
  "flan.jsonl": {
    "field_types": {
      "completions": {
        "list": 20939
      },
      "correct_answers": {
        "list": 20939
      },
      "incorrect_answers": {
        "list": 20939
      },
      "instruction": {
        "str": 20939
      },
      "models": {
        "list": 20939
      },
      "source": {
        "str": 20939
      }
    },
    "format": "jsonl",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 20939,
    "record_count_kind": "exact",
    "samples": [
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      },
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      }
    ]
  },
  "sharegpt.jsonl": {
    "field_types": {
      "completions": {
        "list": 19949
      },
      "correct_answers": {
        "list": 19949
      },
      "incorrect_answers": {
        "list": 19949
      },
      "instruction": {
        "str": 19949
      },
      "models": {
        "list": 19949
      },
      "source": {
        "str": 19949
      }
    },
    "format": "jsonl",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 19949,
    "record_count_kind": "exact",
    "samples": [
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      },
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      }
    ]
  },
  "truthful_qa.jsonl": {
    "field_types": {
      "completions": {
        "list": 811
      },
      "correct_answers": {
        "list": 811
      },
      "incorrect_answers": {
        "list": 811
      },
      "instruction": {
        "str": 811
      },
      "models": {
        "list": 811
      },
      "source": {
        "str": 811
      }
    },
    "format": "jsonl",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 811,
    "record_count_kind": "exact",
    "samples": [
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      },
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      }
    ]
  },
  "ultrachat.jsonl": {
    "field_types": {
      "completions": {
        "list": 9929
      },
      "correct_answers": {
        "list": 9929
      },
      "incorrect_answers": {
        "list": 9929
      },
      "instruction": {
        "str": 9929
      },
      "models": {
        "list": 9929
      },
      "source": {
        "str": 9929
      }
    },
    "format": "jsonl",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 9929,
    "record_count_kind": "exact",
    "samples": [
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      },
      {
        "completions": "list",
        "correct_answers": "list",
        "incorrect_answers": "list",
        "instruction": "str",
        "models": "list",
        "source": "str"
      }
    ]
  }
}
```

## WP3 Adapter Readiness

Use only fields observed above; preserve source scales and leave absent supervision unavailable. No mapping is implemented in WP2.

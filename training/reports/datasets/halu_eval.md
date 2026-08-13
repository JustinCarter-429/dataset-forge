# halu_eval inspection

Snapshot: `snapshot-29c4c6c6d2d4d532`

## Files

- `.gitattributes` — 2419 bytes; unknown; metadata; SHA-256 `a2bdf429c1bc3858213bb4b7a748fa36a71536a5a1ff98861281853c3c2be1f2`
- `README.md` — 1989 bytes; md; metadata; SHA-256 `73b27f588dcc3bdfd29d796fb068027b09ffc1ba64997def9735fcde9c07b62c`
- `data/test-00000-of-00001.parquet` — 3500858 bytes; parquet; payload; SHA-256 `b1c4827b1138a1f09ab1c47fb3c740a0afc12dabfa162bc492d74c947db30f44`

## Observed payload schemas

```json
{
  "data/test-00000-of-00001.parquet": {
    "field_types": {
      "answer": {
        "string": 10000
      },
      "id": {
        "string": 10000
      },
      "label": {
        "string": 10000
      },
      "passage": {
        "string": 10000
      },
      "question": {
        "string": 10000
      },
      "score": {
        "int64": 10000
      },
      "source_ds": {
        "string": 10000
      }
    },
    "format": "parquet",
    "malformed_lines": [],
    "null_counts": {},
    "record_count": 10000,
    "record_count_kind": "exact",
    "samples": [
      {
        "answer": "string",
        "id": "string",
        "label": "string",
        "passage": "string",
        "question": "string",
        "score": "int64",
        "source_ds": "string"
      }
    ]
  }
}
```

## WP3 Adapter Readiness

Use only fields observed above; preserve source scales and leave absent supervision unavailable. No mapping is implemented in WP2.

# Carter 1.0 Part 3: Quality Gate and Export Eligibility

The Carter production path now applies an application-owned quality gate after
canonical generation and before any JSON, CSV, or ZIP artifact is produced.

The gate revalidates every record against its dynamic DatasetSpec and selected
source references, removes exact semantic duplicates, and quarantines records
containing likely secrets or PII. Findings contain only stable codes and safe
messages; model output and sensitive values are never copied into diagnostics.

Only accepted records are exported. A zero-accepted result is not export
eligible. The ZIP contains `quality-report.json`, the generation manifest, and
the accepted dataset. CSV export also neutralizes formula-leading values.

This is deterministic application validation, not an additional runtime or
model fallback. RunPod remains the only enabled Carter PoC runtime.

## Final certification

Certification was run from base commit `f5ca6b3`. The complete backend suite
collected and passed 132 nodes (0 skipped, 0 failed, 0 errors). The focused
quality coverage exercised clean acceptance, schema and fabricated-evidence
rejection, exact duplicates, secret/PII quarantine, dynamic fields,
accepted-only JSON/CSV output, CSV formula protection, and ZIP contents.

The deterministic browser suite passed 5/5 scenarios. Its completed Carter
flow verified upload, generation, visible quality summary/counts, enabled ZIP
download, and the RunPod-only runtime selector. Quarantine and zero-accepted
behavior is certified by the deterministic backend quality-gate fixtures;
the current browser fixture intentionally has no provider scenario that emits
unsafe or all-rejected records, and no production/runtime behavior was changed
to manufacture one for certification.

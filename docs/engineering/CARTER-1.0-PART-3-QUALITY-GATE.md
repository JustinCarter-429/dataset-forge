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

# Provenance policy

Every transformed record must retain its source dataset, split and source ID when available, adapter name/version, transformation version, and optional source URL, license, fingerprint, and timestamp. Unknown values remain null or `unverified`; provenance and license metadata are never fabricated.

Transformations must be reproducible and deterministic. Changes to transformation behavior require a transform-version change. Stable IDs are derived from non-content source identity and version data, preventing raw or sensitive content from appearing in filenames. Avoid storing unnecessary personal information.

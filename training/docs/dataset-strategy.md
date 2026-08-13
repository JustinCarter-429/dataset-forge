# Dataset strategy

The registry contains configuration only; all entries are disabled and no license is asserted. License status is `unverified` pending WP2 verification.

Current priority: 1. Google FACTS Grounding (source-grounding evaluation), 2. OpenBMB UltraFeedback (general quality and comparative judgments), 3. NVIDIA HelpSteer2 (multidimensional scoring), 4. HaluEval (hallucination detection), 5. Stanford Human Preferences/SHP (human preferences), and 6. Nectar (ranking/preferences). Sampling ratios remain undecided.

The long-term highest-value source is Dataset Forge’s own source context + DatasetSpec + candidate + prior state + critic/human judgment + corrected-record history. Its labels may include decisions, scores, issue codes, critiques, and revision directives. No ingestion or adapter implementation occurs in WP1.

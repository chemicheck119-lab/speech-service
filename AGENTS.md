# AGENTS.md

## Scope

These instructions apply to the entire speech service repository.

## Boundaries

- This service performs ASR and returns transcripts, segments, timestamps, and confidence metadata. It must not confirm a CAS number, decide chemical compatibility, or issue operational safety instructions.
- Keep Parser, Resolver, Retriever, Agent, and CAMEO behavior in `analysis-engine`; keep authentication, incident state, and audit history in `back`.
- Never commit raw incident audio, personal information, downloaded datasets, or model weights.

## Review Rules

- Flag metrics that mix WER/CER with chemical-entity recall, or that present synthetic/noise-augmented audio as real field-radio validation.
- Flag train/dev/test leakage, speaker or source overlap, unversioned dataset manifests, and non-reproducible preprocessing.
- Flag code that silently invents missing words, rewrites uncertain chemical terms without preserving the original transcript, or hides low-confidence/failed segments.
- Flag unbounded audio uploads, unsafe media decoding, sensitive transcript logging, and GPU paths without a CPU-safe failure state.
- Require proportional tests for preprocessing, timestamps, confidence mapping, and API contract changes.

## Validation

- Run `python -m compileall -q src tests`.
- Run `PYTHONPATH=src python -m unittest discover -s tests -v`.

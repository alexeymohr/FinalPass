"""Model-agnostic harness for local speech-quality-assessment evaluations.

NOT part of the shipped FinalPass package. Nothing here is imported by
`finalpass`; it adds no runtime dependency to the CLI. The pure-logic modules
(config, chunking, events, model_files, netguard, speech_content) are
deliberately torch-free so they can be tested inside FinalPass's own
environment; only `paderborn_model` and the driver scripts need the isolated
evaluation venv.

The chunk/coverage, exact-sample-mapping, fail-closed model loading, network
guard and candidate-merging ideas were built for the retired LAION artifact
evaluation (see `docs/LAION_RETIREMENT_NOTE.md`) and kept because they are
independent of which model does the scoring.
"""

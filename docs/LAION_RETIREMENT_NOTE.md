# Retired: LAION Speech Artifact Detectors

**Status: evaluated and rejected for FinalPass's audiobook-artifact use case.
Do not reintroduce without new evidence.**

Evaluated 2026-09-03 against a real 5 h 31 m AI-narrated ElevenLabs audiobook
(`laion/speech-artifact-detectors`, revision
`9fba0fdad96e322df0c24321b7b730dbf829a1f2`, 10 detector checkpoints, fully
local, zero network attempts, bit-exact determinism). Retired 2026-09-11.

## Why it was rejected

The bank is technically sound and the harness ran cleanly. The problem is a
scope mismatch, and it is not fixable by retuning a threshold:

1. The detectors answer **"what was done to this recording?"** (denoising,
   codec, pitch shift, comb filtering, synthesis provenance). The operator needs
   **"where did the generated speech go audibly wrong?"** Different questions.
2. Its three highest-signal detectors here (`stft_classifier`, `waveform_1d`,
   `spectral_denoising`) are provenance detectors. Every window of an
   AI-narrated audiobook is vocoder-synthesised, so they fire on the material
   rather than on its defects. Excluding them on principle leaves the bank
   nearly silent.
3. What remained fired almost exclusively on **near-silence**, which is
   out-of-distribution for models trained on speech crops. Median speech
   content was 0.431 in candidate windows versus 0.696 in non-candidates: in
   effect a silence detector. On the 3,879 windows with >= 50 % speech content,
   `phase_vocoder` (warble), `comb_filtering` and `bandwidth_limitation`
   (muffling) never once crossed 0.50 in 5 h 31 m of speech.
4. Operator audition of the flagged interior regions found no defects.

The predeclared workload gate returned a *passing* 2.08 candidate minutes per
finished hour. **The metric passed and the result was still wrong** — the gate
measured how much was flagged, not whether it was a defect.

## Lessons that outlived the experiment

- **A speech-content precondition is mandatory** for any interior-artifact
  detector on this material. A defect has to be *in the speech* to matter. This
  is the same lesson as the audibility gate that rescued the truncation model.
- **A burden gate is not an accuracy gate.** Report both, and never call a
  detector useful before a human has auditioned its output.
- **The harness was model-agnostic and was kept.** Deterministic chunk planning
  with proven full coverage, exact source-sample mapping, fail-closed local
  model loading, the process-local network guard, candidate merging and
  deterministic ranking now live in `tools/sqa_eval/` and serve the Paderborn
  frame-level speech-quality evaluation.

## What was removed

`tools/laion_eval/` (LAION policy, scanner, detector-score clustering,
architecture-registry checkpoint validation, determinism driver, summariser)
and `tests/test_laion_harness.py`. The LAION mission brief, evaluation report,
and numeric scan results are archived locally and untracked under
`resources/laion_retired/`; the model weights and the LAION evaluation venv
were deleted. Nothing LAION-specific ever entered a released FinalPass schema,
CLI surface, or dependency, so no compatibility shim was needed.

# Reconstructed Git History for ReQAP Retrieval Project

This repository is a transparent reconstruction of the project development history from local archives, patch bundles, reports, and the final working tree.

It is not presented as the original chronological Git repository. Instead, it is a version-control reconstruction intended to make the algorithmic evolution reviewable and submit-ready.

## Source artifacts

The reconstruction used the following local materials:

1. `ReQAP-main/reqap_code_only.tgz`
   - Used as the original ReQAP baseline snapshot.
   - Contains the base ReQAP code and retrieval modules.

2. `ReQAP-main/reqap-retrieval-modular.zip`
   - Used as the first modular retrieval snapshot.
   - Introduces `reqap-retrieval-modular` with BM25, dense, SPLADE adapters and basic fusion experiments.

3. `ReQAP-main/reqap-retrieval-modular (RRF).zip`
   - Used as the RRF experimental snapshot.
   - Adds Dense + SPLADE reciprocal-rank fusion and related evaluation assets.

4. `ReQAP-main/reqap-retrieval-modular (WRRF融合).zip`
   - Used as the weighted-RRF / frequency-routing snapshot.
   - Adds weighted RRF, query-frequency routing, semantic cache, and BM25 cold-query branch.

5. `ReQAP-main/reqap_retrieval_patch_bundle_20260420_192525.tar.gz`
   - Used as the Grid-48 / learned-router patch snapshot.
   - Adds rule routing, score-level weighted-sum fusion, Grid-48 classes, learned router training, multi-task benchmark scripts, and SPLADE retrieval patches.

6. `ReQAP-main/reqap-retrieval-modular/`
   - Used as the final working tree snapshot.
   - Adds final training/evaluation scripts and replay verification artifacts.

7. `报告/1.md` to `报告/5.md`
   - Used as development notes to validate the reconstructed sequence.

## Reconstructed commits

The commit order is:

1. `chore: import original ReQAP code-only baseline`
2. `feat: extract modular retrieval package with BM25 dense and SPLADE adapters`
3. `feat: add dense SPLADE RRF dynamic fusion experiment`
4. `feat: add query-frequency routing semantic cache and weighted RRF`
5. `feat: replace rank fusion with weighted-score routing and Grid-48 learned router`
6. `feat: finalize retrieval benchmarks SPLADE training scripts and replay verification`

## Interpretation

The history reflects the real available artifacts, but some commit boundaries are reconstructed from archive-level snapshots rather than original Git metadata. Dates were assigned from archive modification times or report-supported chronology.

The most important algorithmic evolution is:

- Original ReQAP retrieval baseline
- Modular retrieval abstraction
- Dense + SPLADE RRF
- Frequency-aware weighted RRF and cache experiments
- Score-level BM25 / dense / SPLADE fusion with rule routing
- Grid-48 and learned router
- Final benchmark/replay verification workflow

## Submission note

For formal submission, this repository can be used to show iterative development. If a stricter code-only submission is required, generated indexes, datasets, and benchmark output files should be removed or kept in a separate artifact archive.


## Code-only repository note

This code-only repository replays the same reconstructed commits while excluding benchmark data, generated indexes, model files, caches, and large result artifacts. It is intended for source-code submission.

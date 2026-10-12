# F2 Real-Data Recovery Results

This directory contains the small, frozen reports pulled from the server after
the real-data single-seed run. The model checkpoint is intentionally not copied
into the repository.

| Stage | Samples | Correct | Accuracy | Format failures | Reported errors |
|---|---:|---:|---:|---:|---:|
| Base | 48 | 0 | 0% | 18 | 29 |
| Recovery SFT validation | 16 | 16 | 100% | 0 | 0 |
| Recovery SFT held-out test | 16 | 16 | 100% | 0 | 0 |

## Recipe

- Base model: `Qwen3-0.6B`
- Training: full-parameter BF16, 48 steps, `lr=5e-6`, seed `7`
- Training set: 16 real-data recovery samples; each exposed three times
- Validation/test: disjoint entity-pool splits, 16 samples each
- Candidate digest: `sha256:79e62b311240654dd7abcab577b1b27a9dcad61fcd59e2d025b023f7d253d185`

The validation and test correct rows are direct correct tool-call trajectories;
they do not show a model making an error and then recovering at inference time.
The result therefore supports narrow schema/key-selection learning, not a broad
online recovery or general GIS-agent accuracy claim.

The server result archive was verified with SHA256
`92463c99504093c93149843ac0bda680487c8e5cffda330be72561140f8dc127`.

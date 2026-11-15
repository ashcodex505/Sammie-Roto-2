# M4 benchmark contract

This directory is reserved for the first implementation milestone described in `docs/M4_OPTIMIZATION_PLAN.md`.

Every benchmark result must record:

- Git commit and whether the worktree is dirty;
- macOS, machine model, memory, Python, torch, and torchvision versions;
- MPS environment variables and runtime profile;
- checkpoint filename plus SHA-256;
- input clip fingerprint, frame range, prompts, and internal resolution;
- cold-load time, first inference, steady-state time, and total wall time;
- FPS, RSS, MPS current/driver allocation, and thermal/power samples;
- output mask hashes and quality comparison against the stored baseline.

Rules:

1. Synchronize MPS only at explicit measurement boundaries.
2. Run one cold pass and at least three warm repetitions.
3. Report the median and range; do not select the fastest run.
4. Change one performance hypothesis at a time.
5. Reject speedups that fail the quality tolerance.
6. Keep generated results and large media out of Git.

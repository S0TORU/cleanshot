# Release Checklist

Use this checklist before any public release or announcement.

## Quality gate (must pass)

- [ ] GitHub Actions `CI` workflow is green on the target commit.
- [ ] `python -m pytest -q` passes locally.
- [ ] Installed CLI smoke tests pass (`cleanshot` command, schema/report/split/fusion flows).
- [ ] Wheel and source distribution build successfully.
- [ ] `python -m twine check dist/*` passes.

## Product readiness

- [ ] README examples are accurate for the current CLI flags.
- [ ] `--help` output has no broken or unclear option descriptions.
- [ ] Error paths have clear hints for common failures (bad input, missing columns, bad split ratios).
- [ ] Sample datasets still run end-to-end.

## Risk and trust

- [ ] Known critical/high bugs are resolved or explicitly documented.
- [ ] License file is present and correct.
- [ ] No sensitive/private data is committed in examples, tests, or docs.
- [ ] Version in `pyproject.toml` is updated for the release.

## Release execution

- [ ] Tag and release notes are prepared.
- [ ] Rollback plan is documented (how to yank/revert release quickly).

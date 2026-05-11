# Agent Instructions

This repo uses three living status documents to keep implementation, parity, and
test coverage understandable across agent handoffs:

- `map.md`
- `devtools_coverage.md`
- `docs/python-test-suite-summary.md`

Before committing any code or test change, review whether each document needs an
update. Update the relevant files in the same commit as the code change.

Treat these files as part of the changed behavior. A feature, parity port, test
addition, fixture refresh, or refactor is incomplete until the relevant living
docs either reflect it or the commit/PR text explicitly says why no living-doc
change was needed.

## Living Doc Rules

- Update `map.md` when public APIs, source-tree structure, implemented MATLAB
  parity, feature status, known limitations, or the verification snapshot
  changes.
- Update `devtools_coverage.md` when MATLAB devtools parity coverage changes,
  including golden fixtures, MATLAB fixture generators, Python fixture
  generators, parity tests, ranked port status, or suggested next ports.
- Update `docs/python-test-suite-summary.md` when tests are added, removed,
  renamed, parametrized, or when the behavior covered by existing tests changes
  meaningfully.

## Commit Checklist

Before making a commit:

1. Run the relevant tests. Prefer `uv run pytest` for changes with broad impact.
2. Check `git diff --stat` and identify whether `src/`, `tests/`, `scripts/`,
   `tests/golden/`, or docs changed.
3. Explicitly decide whether each living status document needs an update.
4. If code or tests changed and none of the three status documents changed,
   mention why in the commit message or PR summary.
5. Keep living-doc edits factual: record what changed, what is covered, what is
   still deferred, and the exact test snapshot when it changes.

## Pre-Commit Gate

Use this quick gate before every agent-authored commit:

- If `src/` changed, check `map.md`.
- If `tests/` changed, check `docs/python-test-suite-summary.md`.
- If MATLAB parity scripts, golden data, or `tests/test_devtools_parity.py`
  changed, check `devtools_coverage.md`.
- If pytest collection counts, pass/fail/xfail totals, or broad coverage claims
  changed, update the affected snapshot text.
- If no living-doc update is needed, include a short note such as
  `Living docs: no update needed; change is internal-only` in the commit body or
  PR summary.

## Usual Mapping

- Changes under `src/` usually require a `map.md` update.
- Changes under `tests/` usually require a
  `docs/python-test-suite-summary.md` update.
- Changes under `tests/golden/`, `scripts/matlab/`,
  `scripts/generate_devtools*`, or `tests/test_devtools_parity.py` usually
  require a `devtools_coverage.md` update.
- Feature or parity commits that change the pytest result count should update
  the verification snapshot in `map.md`.

Do not edit `external/chunkie-matlab/devtools/test` for parity work. Add MATLAB
wrappers under `scripts/matlab`, compact fixture data under `tests/golden`, and
Python comparisons under `tests`.

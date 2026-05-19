# Agent Instructions

Read `CONTRIBUTING.md` before making code, test, fixture, or documentation
changes. It defines the repo's Python-first design philosophy, naming standard,
tensor notation, testing expectations, tooling, and living-doc policy.

This repo uses one living status document to keep implementation, parity, and
test coverage understandable across agent handoffs:

- `map.md`

Before committing any code or test change, review whether `map.md` needs an
update. Update it in the same commit as the code change when relevant.

Treat `map.md` as part of the changed behavior. A feature, parity port, test
addition, fixture refresh, or refactor is incomplete until `map.md` either
reflects it or the commit/PR text explicitly says why no living-doc change was
needed.

## Commit Granularity

Make small, logically complete commits as code changes are made. Prefer a series
of focused commits over one large end-of-task commit, especially when a task
touches multiple modules, tests, fixtures, or living docs.

Each commit should include the relevant code, tests, fixtures, and living-doc
updates for that specific change. Do not bundle unrelated refactors, formatting,
or follow-up work into the same commit. If the worktree contains unrelated
uncommitted user changes, leave them unstaged and commit only the files that
belong to the current change.

## Living Doc Rules

All project-owned Markdown docs are living docs: keep them current when their
subject changes, or delete them when they are obsolete. Markdown files under
`external/` are upstream/reference material and should only change through the
corresponding vendored checkout or submodule. `map.md` has additional
commit-gate rules because it summarizes implementation, parity, and test
coverage across handoffs.

- Update `map.md` when public APIs, source-tree structure, implemented MATLAB
  parity, feature status, known limitations, or the verification snapshot
  changes.
- Keep every `map.md` status table ordered by maturity from least mature to most
  mature: limited/deferred/non-goal rows first, implemented-only rows next,
  implemented plus Python-tested rows next, and fully implemented plus
  Python-tested plus MATLAB-parity-tested rows last. Treat private/internal
  helper status as metadata rather than its own maturity tier, and preserve
  relative order within the same tier when practical.

## Commit Checklist

Before making a commit:

1. Run the relevant tests. Prefer `uv run pytest` for changes with broad impact.
2. Check `git diff --stat` and identify whether `src/`, `tests/`, `scripts/`,
   `tests/golden/`, or docs changed.
3. Explicitly decide whether each living status document needs an update.
4. If code or tests changed and `map.md` did not change, mention why in the
   commit message or PR summary.
5. Keep living-doc edits factual: record what changed, what is covered, what is
   still deferred, and the exact test snapshot when it changes.

## Pre-Commit Gate

Use this quick gate before every agent-authored commit:

- If `src/` changed, check `map.md`.
- If `tests/`, MATLAB parity scripts, or golden data changed, check `map.md`.
- If pytest collection counts, pass/fail/xfail totals, or broad coverage claims
  changed, update the `map.md` snapshot text.
- If no living-doc update is needed, include a short note such as
  `Living docs: no update needed; change is internal-only` in the commit body or
  PR summary.

## Usual Mapping

- Changes under `src/` usually require a `map.md` update.
- Changes under `tests/`, `tests/golden/`, `scripts/matlab/`,
  `scripts/generate_devtools*`, or parity tests usually require a `map.md`
  update.
- Feature or parity commits that change the pytest result count should update
  the verification snapshot in `map.md`.

Do not edit `external/chunkie-matlab/devtools/test` for parity work. Add MATLAB
wrappers under `scripts/matlab`, compact fixture data under `tests/golden`, and
Python comparisons under `tests`.

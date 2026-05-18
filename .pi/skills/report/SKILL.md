---
name: report
description: Summarises all code changes made since AGENT.md (or AGENTS.md) was last updated and presents the summary to the user. Use when the user asks for a report or a summary of recent changes.
---

# Report

Produce a summary of every code change made since the project's agent context file was last
updated and present it to the user. Do not modify AGENTS.md or any other file.

## Steps

### 1 — Locate the agent context file

Check for `AGENT.md` first, then `AGENTS.md`. Use whichever exists. If neither exists, tell the
user and stop.

### 2 — Find the last commit that touched the file

```bash
git log -1 --format="%H %s" -- AGENT.md AGENTS.md
```

If the file has never been committed (no output), treat the entire git history as the relevant
range. If git is not available or the directory is not a repository, read the file's modification
timestamp with `stat` as a fallback, and note this in the report.

### 3 — Collect changes since that commit

Get a concise list of commits:

```bash
git log --oneline <hash>..HEAD
```

Get the full diff of source and test files:

```bash
git diff <hash>..HEAD -- src/ tests/ pyproject.toml
```

If the file was never committed, use `git log --oneline` and `git diff HEAD -- src/ tests/ pyproject.toml`.

### 4 — Inspect changed source files

For any source or test file that appears in the diff, use `read` to load its current contents so
the report is grounded in the actual code rather than just the diff text.

### 5 — Produce the summary

Present the user with a clear prose summary structured as follows. Keep it concise.

**What changed** — a bullet per meaningful change: new functions/classes, modified behaviour,
bug fixes, renamed symbols, moved files, config changes. Group related items together. Do not
list trivial whitespace or formatting changes unless they were the explicit goal.

**Bugs caught or fixed** — if any bugs were identified and fixed during the session, call them
out explicitly with a brief explanation of the root cause.

**Outstanding issues** — note anything known to be incomplete, stubbed, or carrying active
linter errors.

Stop here. Do not modify any files.

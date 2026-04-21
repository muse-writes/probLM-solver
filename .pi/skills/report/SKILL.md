---
name: report
description: Summarises all code changes made since AGENT.md (or AGENTS.md) was last updated, presents the summary to the user, then updates the file. Use when the user asks for a report, a summary of recent changes, or asks to update AGENT.md/AGENTS.md.
---

# Report

Produce a summary of every code change made since the project's agent context file was last
updated, present it to the user, then apply it to the file.

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

### 4 — Read the current agent context file

Use the `read` tool to load the full contents of AGENT.md / AGENTS.md. This is the baseline to
update against.

### 5 — Inspect changed source files

For any source or test file that appears in the diff, use `read` to load its current contents so
the report is grounded in the actual code rather than just the diff text.

### 6 — Produce the summary

Present the user with a clear prose summary structured as follows. Keep it concise.

**What changed** — a bullet per meaningful change: new functions/classes, modified behaviour,
bug fixes, renamed symbols, moved files, config changes. Group related items together. Do not
list trivial whitespace or formatting changes unless they were the explicit goal.

**Bugs caught or fixed** — if any bugs were identified and fixed during the session, call them
out explicitly with a brief explanation of the root cause.

**Outstanding issues** — note anything known to be incomplete, stubbed, or carrying active
linter errors.

### 7 — Confirm with the user before writing

Ask: *"Would you like me to update AGENT.md with these changes?"* and wait for confirmation
before proceeding to step 8. If the user declines, stop here.

### 8 — Update the agent context file

Apply all of the following that are relevant:

- **Directory structure** — reflect any new or moved files
- **Core modules** — update method signatures, add new methods/classes, remove deleted ones
- **Architecture / dependency diagram** — update if imports or module relationships changed
- **Data flow** — update if the runtime workflow changed
- **Constants / functions lists** — keep in sync with the actual code
- **Code Quality / Testing** — update test file list, mocking strategies, doctest notes
- **Known Issues — Active** — add new issues discovered; update descriptions if partially fixed
- **Known Issues — Resolved** — append one ✓ line per resolved issue; do not remove old entries

Write the updated file with the `write` tool. Confirm to the user once done.

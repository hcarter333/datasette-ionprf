## PR workflow (for agents)

- Always start from the default branch HEAD:
  - `git fetch origin`
  - `git checkout -B codex/<short-slug> origin/main`

- Make only the requested edits.

- Stage **only tracked changes** (no untracked files):
  - `git add -u`                # modified + deleted only
  - (If adding new files, list them explicitly: `git add path/to/newfile`)

- Run formatters/tests **only for staged files**:
  - JS/TS example: `prettier --write $(git diff --name-only --cached)`
  - Python example: `ruff check --fix --select I --exit-zero $(git diff --name-only --cached)`

- Commit & push:
  - `git commit -m "<concise title>"`
  - `git push -u origin HEAD`

- Open a PR with base=`main`, head=`current branch`. Title: `<scope>: <summary>`. Body: short rationale and test notes.

- Do **not** touch other files. If more than 5 files are staged, stop and ask for confirmation.

- Normalize noisy diffs:
  - `git config core.autocrlf false`
  - `git config core.filemode false`

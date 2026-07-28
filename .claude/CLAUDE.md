# Agent instructions

## Commits

**The agent commits; the user approves every commit.** Creating commits with
`git commit` is allowed — but only through the workflow below, and each individual
commit must be approved by the user immediately beforehand. **Never run `git push`,
under any circumstances — not even when asked.** Publishing history is the user's
action alone. Do not rewrite existing history either: no `git commit --amend`, no
rebase, no reset of committed work.

**The gate: a yes/no question right before every commit.** Immediately before each
`git commit`, display the full commit message and ask the user for approval with an
AskUserQuestion (yes/no). Commit only on an explicit "yes". On "no", revise per the
user's feedback and ask again, or stop. Never batch approvals: one question per
commit, asked after that commit's changes are staged, never before the staging is
done or for several commits at once.

**Workflow for each commit:**

1. **Stage exactly the changes the message will describe.** Stage files partially
   (`git add -p`, or `git apply --cached` with a crafted patch) whenever only part
   of a file's changes belongs in this commit — this is the normal case when one
   session's edits are being split across multiple commits, not an exotic one.
2. **Compose the message** following the `git-commit-message` skill
   (`.claude/skills/git-commit-message/SKILL.md`): Conventional Commits, the git
   50/72 rule, detailed body, and the exact `Co-authored-by:` trailer with your own
   display name spelled as words — never the API model ID, never extra angle
   brackets.
3. **Verify the staged diff against the message.** Re-read `git diff --staged` and
   `git diff --staged --stat` and check both directions: every claim in the message
   is true of the staged diff, and everything staged is accounted for in the
   message. No unmentioned functional changes, no other change's churn (lockfiles,
   formatting) riding along, no staged file the message does not explain. Fix the
   staging or the message until they match.
4. **Ask the yes/no AskUserQuestion**, showing the message.
5. **Commit** with that exact message — only after the "yes".

**Splitting work across commits.** When the session's changes are logically several
commits, run the full loop above once per commit: stage the first commit's hunks,
verify, ask, commit; then stage the next commit's hunks, verify, ask, commit; and
so on. Partial staging is what makes the split honest — never lump unrelated
changes into one commit because they landed in the same file.

**The git history is this project's design record.** Write detailed bodies: why the
change was made, alternatives rejected and why, constraints, evidence, and
references — including the durable reasoning from the session that produced it.
Prefer putting that history in the commit message over explanatory code comments or
standalone notes files, which drift out of date.

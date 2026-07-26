# Agent instructions

## Commits

**Never commit.** Do not run `git commit`, `git commit --amend`, `git revert`,
`git push`, or anything else that creates or rewrites history — even if asked.
Committing is the user's action.

**After every code change, print a sample commit message.** Once you finish editing
files, display a ready-to-copy commit message in a fenced code block, following the
rules in the `git-commit-message` skill (`.claude/skills/git-commit-message/SKILL.md`):
Conventional Commits format, the git 50/72 rule, and a trailing
`Co-authored-by: Claude <model> <noreply@anthropic.com>` footer.

**The git history is this project's design record.** Write detailed bodies: why the
change was made, alternatives rejected and why, constraints, evidence, and
references — including the durable reasoning from the session that produced it.
Prefer putting that history in the commit message over explanatory code comments or
standalone notes files, which drift out of date.

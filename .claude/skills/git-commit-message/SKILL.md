---
name: git-commit-message
description: Rules for composing Conventional Commit messages that obey the 50/72 rule and carry a detailed body — why the change was made, alternatives rejected, evidence, and references — because this project's git history is its living design record rather than comments and docs files. Use whenever a commit message is written, proposed, or reviewed — including after any code change, when the user asks "what should I commit this as", or when drafting a message for the user to copy. Never commits; only displays the message.
---

# Git Commit Messages

## Hard rule: never commit

**Never run `git commit`, `git commit --amend`, `git revert`, `git push`, or any other
command that creates or rewrites a commit — not even when asked, not even with `--dry-run`.**

Your only output is the commit message itself, displayed in a fenced code block so the
user can copy it. Staging inspection (`git status`, `git diff`, `git diff --staged`,
`git log`) is fine and encouraged — it is how you learn what the message should say.

If the user asks you to commit, print the message and tell them to run the commit
themselves.

## The 50/72 rule

This is the formatting rule that matters most. Enforce it on every message.

1. **Subject line ≤ 50 characters.** Hard ceiling; treat 50 as a limit, not a target.
   Includes the `type(scope):` prefix. If it doesn't fit, narrow the scope or move
   detail into the body — never let it spill.
2. **Blank line after the subject.** Always — before a body, and before the footers
   when there is no body. Git treats the first paragraph as the subject, and its
   trailer parser only recognizes a footer block that is preceded by a blank line.
   Without it, `Co-authored-by:` is silently swallowed into the subject and the
   attribution never registers.
3. **Body wrapped at 72 characters.** Wrap hard at 72 — git does not reflow, and
   `git log` indents by 4, so 72 keeps it readable in an 80-column terminal.
4. **Blank line between the body and the footers.** Whenever both exist, exactly one
   empty line separates the last body line from the first footer. The footer block is
   only a footer block because a blank line precedes it — run the two together and
   git reads your trailers as one more paragraph of body text.
5. **Blank line between body paragraphs.** One empty line, never two.
6. No trailing whitespace anywhere, and no trailing period on the subject.

Every part of the message is separated from the next by exactly one blank line:
subject → body → footers. The only place a blank line must *not* appear is inside
the footer block itself.

Do not exempt URLs, long identifiers, or code snippets from the wrap unless breaking
them would corrupt the value; if so, put them on their own line.

## Structure

```
<type>(<optional scope>)<optional !>: <description>
                                              <- blank line
<optional body>
                                              <- blank line
<optional single-line footers>
Co-authored-by: Claude <model> <noreply@anthropic.com>
```

The final block is the trailer block: one `Token: value` per line, no blank lines
inside it, `Co-authored-by:` last.

## Types

| Type | Use for |
| --- | --- |
| `feat` | Adds, adjusts, or removes a feature affecting the API or UI |
| `fix` | Resolves a bug in a preceding feature |
| `refactor` | Restructures code without altering behavior |
| `perf` | A refactor that specifically improves performance |
| `style` | Formatting only — whitespace, semicolons; no functional change |
| `test` | Adds or corrects tests |
| `docs` | Documentation only |
| `build` | Build tooling, dependencies, versioning |
| `ops` | Infrastructure, deployment, CI/CD, monitoring, recovery |
| `chore` | Routine tasks, initial commits, `.gitignore` edits |

The type is REQUIRED and is followed by an optional scope, an optional `!`, then a
REQUIRED colon and space.

## Scope

- Optional; a noun naming a section of the codebase, in parentheses: `fix(parser):`.
- Prefer scopes already used in this repo's `git log`.
- Never use an issue ID as a scope.

## Description

- REQUIRED, immediately after the colon and space.
- Imperative present tense: "add", not "added" or "adds".
- Lowercase first letter. No terminal period.
- Say what the change does, not what you did.

## Body

**The commit history is this project's living record of how it came to be — a
narrative of the decisions and the thinking behind them, not a changelog.** It exists
so the codebase does not have to carry that weight as explanatory comments and
`NOTES.md` / `DECISIONS.md` files that drift out of date the moment someone edits the
code beside them. A commit is timestamped, immutable, and permanently attached to the
exact diff it describes, so it cannot drift. Write the body accordingly.

Assume the reader is someone — a person or an agent — who lands on this commit via
`git log`, `git blame`, or `git bisect` months from now, knows nothing about the
session that produced it, and needs to understand the change well enough to safely
modify or revert it. Everything they would need must be *in the message*, because
nothing else will have kept it.

**Default to writing a body.** Skip it only for changes with genuinely nothing to
explain — a typo fix, a version bump, a rename with no judgement in it.

Cover, in roughly this order, whichever apply:

- **Why.** The problem, symptom, or requirement that prompted the change. State it
  before the solution. A reader who disagrees with the *why* needs to see it to
  challenge it.
- **What changed, at a high level.** Enough shape to orient someone before they read
  the diff. Do not narrate the diff line by line — it is right there.
- **Alternatives considered, and why they lost.** This is the single most valuable
  thing in the message and the first thing lost if unwritten. Name each option and
  give the concrete reason it was rejected. A rejected approach recorded here stops
  the next person from spending a day rediscovering the same dead end — and tells
  them when the trade-off is worth revisiting.
- **Constraints and assumptions.** Anything that bounded the solution: a requirement
  from the user, a platform limit, an API that behaves unexpectedly, a deadline
  shortcut. Say plainly when something is a stopgap and what the real fix would be.
- **Evidence.** Benchmark numbers, profiler output, the failing case that proved the
  bug, the command that verified the fix. Include the command itself so it can be
  re-run. Quote figures rather than characterising them: "11.2 ms → 2.4 ms" beats
  "much faster".
- **Consequences.** What this makes newly possible, what it forecloses, what still
  needs doing. Known limitations belong here, not in a comment.
- **References.** See below.

Use imperative present tense for statements of what the change does, matching the
subject. Ordinary prose is fine and clearer for rationale, alternatives, and
findings — do not contort an explanation to stay imperative.

Length follows substance: as long as the reasoning genuinely requires, no longer.
A body that restates the subject in more words is worse than none.

### References

Cite the sources behind the change so the reader can check the reasoning instead of
taking it on faith:

- Specs, RFCs, and standards — with the section: `Conventional Commits v1.0.0 §16`.
- Documentation, articles, papers, issues, PRs, Stack Overflow answers, vendor bug
  reports — anything consulted while deciding.
- Prior commits in this repo, by short SHA and subject:
  `see b20fdb7 "Render a triangle in a winit window with wgpu"`.

Put a bare URL on its own line and leave it unwrapped — a broken URL is useless, and
this is the one accepted exception to the 72-column limit. Short, single-line
references may go in the footer block as `Ref: <value>`; anything longer or needing
explanation goes in the body next to the claim it supports.

Never cite something you did not actually consult, and never invent a plausible-
looking URL. An unverifiable citation is worse than none.

### Carrying over the session

When the change came out of an agent session, the reasoning already happened — most
of it lives in that conversation and is about to be discarded. Move the durable parts
into the message:

- Requirements, corrections, and preferences the user stated in their own words, when
  they explain why the code looks the way it does.
- Findings from investigation: what was searched for, what turned out to be true, what
  turned out to be false. Disproved theories are worth a line — they stop the next
  reader from re-forming them.
- Empirical results: commands run and what they showed, including the ones that
  falsified an earlier assumption.
- Approaches tried and abandoned during the session, and the reason.

Translate all of it into standalone prose. The commit must stand on its own: no
"as discussed", no "per your earlier message", no references to the conversation as
a thing the reader can see. Filter hard — transcript noise, false starts already
corrected, and routine tool output do not belong.

Never carry over secrets, credentials, tokens, personal data, or internal URLs that
happened to appear in the session.

## Breaking changes

Mark with `!` before the colon: `feat(api)!: drop v1 auth endpoint`.

Also describe the break unless the description alone makes the impact clear. Two
shapes, both valid:

- **Fits on one line** — put it in the trailer block, hyphenated:
  `BREAKING-CHANGE: v1 auth is gone, use v2`
- **Needs more than one line** — write it as its own paragraph above the trailer
  block, separated by a blank line. This paragraph is prose, *not* a footer: git
  reads it as body text. It is sufficient only because the `!` already marks the
  break — which is why the `!` is never optional here.

Prefer the hyphenated `BREAKING-CHANGE`. Conventional Commits v1.0.0 §16 makes it an
exact synonym — "BREAKING-CHANGE MUST be synonymous with BREAKING CHANGE, when used
as a token in a footer" — and it matches the general token rule in §9 that footer
tokens use `-` in place of whitespace. Separately, git's own trailer parser rejects
the spaced form outright, since a trailer token may not contain spaces.

Either spelling MUST be uppercase (§15): it is the one case-sensitive token in the
spec; everything else is case-insensitive.

## Footers

- Always preceded by a blank line — after the body, or after the subject when there
  is no body. Never let a footer touch the line above it.
- The footers themselves are one contiguous block: no blank lines *between* them.
- **Every footer in the final block is one line, `Token: value`, colon separator.**
  Git's trailer parser only recognizes the last block when *every* line in it is a
  valid trailer, and its default separator is `:`. One offending line and the whole
  block — `Co-authored-by:` included — is ignored.
- Tokens use hyphens instead of spaces (`Reviewed-by`, `Acked-by`, `BREAKING-CHANGE`).
- Reference issues with a colon: `Closes: #123`, `Fixes: JIRA-456`. The spec also
  permits the `Token #value` form, but git does not parse it — do not use it.
- The spec allows a footer value to span lines, ending at the next valid token. Never
  do that in the final block: a wrapped value is not a trailer line and kills the
  block. Put long prose in a paragraph above the trailer block instead.
- If a value cannot fit in 72 characters on one line, it belongs in the body or in its
  own paragraph — not in the trailer block.

### Required co-author footer

Every message you produce MUST end with, as the last line:

```
Co-authored-by: Claude <model> <noreply@anthropic.com>
```

where `<model>` is the display name of the model writing the message — e.g.
`Co-authored-by: Claude Opus 5 <noreply@anthropic.com>` or
`Co-authored-by: Claude Fable 5 <noreply@anthropic.com>`. Use the model you are
actually running as; do not hardcode a name from an example.

## Examples

A change with real reasoning behind it — why, alternatives and why they lost, the
constraint that shaped it, evidence, and a reference. This is the default shape:

```
perf(terrain): swap quadtree LOD for clipmaps

Frame time on the 4k heightmap was dominated by quadtree node
rebuilds: 11.2 ms of a 16.6 ms budget, measured with
`cargo run --release -- --bench terrain`. Geometry clipmaps make
the vertex layout static, so the per-frame cost becomes a texture
upload of the ring deltas — 2.4 ms in the same benchmark.

Alternatives considered:

- Cache rebuilt quadtree nodes. Gets to ~7 ms, but only while the
  camera is slow; a fast pan still stutters.
- CDLOD. Comparable frame time, but it needs a morph factor in
  the vertex shader and we would carry both paths through the
  migration. Worth revisiting if we ever need per-vertex
  displacement.

This assumes the camera-relative origin shift already in place;
without it the rings lose precision past roughly 50 km out.

Ring count of 5 is from Asirvatham & Hoppe, who report it as the
point where popping stops being visible at 60 fps.

Ref: GPU Gems 2 ch. 2, Asirvatham & Hoppe
Closes: #118
Co-authored-by: Claude Opus 5 <noreply@anthropic.com>
```

A smaller change still earns a body — the symptom and the reason are not in the diff:

```
feat(input): add yoke deadzone calibration

Raw joystick axes jitter around center on cheap hardware, causing
constant micro-corrections in the flight model. Apply a configurable
deadzone before the axes reach the control surfaces.

Per-axis rather than a single global value: the two test sticks
drift by very different amounts on pitch versus roll.

Closes: #42
Co-authored-by: Claude Opus 5 <noreply@anthropic.com>
```

Bodyless commits are the exception, not the norm — reserve them for changes with
nothing to explain:

```
fix(render): clamp depth range to stop z-fighting

Co-authored-by: Claude Opus 5 <noreply@anthropic.com>
```

```
feat(api)!: remove deprecated /v1/telemetry

BREAKING CHANGE: /v1/telemetry is gone; use /v2/telemetry, which
returns the same payload under a `data` key.

Co-authored-by: Claude Opus 5 <noreply@anthropic.com>
```

The wrapped break notice is its own paragraph, so the trailer block below it stays
clean. When it fits on one line, fold it in instead:

```
feat(api)!: remove deprecated /v1/telemetry

BREAKING-CHANGE: /v1/telemetry is gone, use /v2/telemetry
Co-authored-by: Claude Opus 5 <noreply@anthropic.com>
```

## Checklist before displaying

- [ ] Subject ≤ 50 chars, imperative, lowercase, no period
- [ ] Valid type; scope only if it helps
- [ ] Blank line after the subject, before any body or footer
- [ ] Blank line between the body and the footers, whenever both exist
- [ ] Body wrapped at 72
- [ ] Body present unless the change truly has nothing to explain
- [ ] Body gives the *why*, not just the what
- [ ] Alternatives considered are named, with the reason each lost
- [ ] Constraints, assumptions, stopgaps, and known limitations are stated
- [ ] Evidence included — numbers, failing case, or the verifying command
- [ ] Sources cited, and every citation is one you actually consulted
- [ ] Durable reasoning from the session carried over as standalone prose
- [ ] No secrets, credentials, or "as discussed" references to the conversation
- [ ] Breaking change marked with `!` and/or an uppercase `BREAKING-CHANGE:` notice
- [ ] Final block is all single-line `Token: value` trailers — no wrapped values, no
      `Token #value`, no blank lines inside it
- [ ] `Co-authored-by: Claude <model> <noreply@anthropic.com>` is the last line
- [ ] You did not run `git commit`

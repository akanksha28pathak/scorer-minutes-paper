# Gemini review, gate G0 (baseline draft, Nov 6; pasted by Akanksha)

Average 3.6 / 5: Novelty 3.0, Quality 3.5, Relevance 4.5, Verifiability 3.5, Clarity 3.5.
"Practical Best New Resource toolkit held back by circular statistical validation, unanchored reference speeds, and
internal number discrepancies."

Weaknesses (most serious first):
1. Circular validation: the cross-hardware v12e pair (17 / 19) is inside the "identical-run" noise pool, then that SD
   is used to call 17 vs 19 "within noise"; the only completed conversion test is at an 18% speed gap. Fix: [L4] pair;
   drop v12e from the pure noise pool (5 single-hardware pairs) and test v12e against that independent SD.
2. Three competing reference speeds (33.8 from 1 run / 5 sessions, 31.6 proxy, 30.4 A100 median). Fix: anchor every
   factor on the sequential 4x L4 speed; more sequential sessions (15-20).
3. Headline 4/10 -> 9/10 is confounded (cap over-scaled by 24%, 10 tasks, no reference baseline). Fix: lead with the
   55-task [L4] converted vs nominal pair; keep v9b as an anecdote about guessed factors.
4. Count discrepancies: 81 runs / 3,513 sessions / five GPU types / seven setups vs table rows summing to 40 runs,
   "54 full runs", "15 full runs" vs 12 in section 5.1. Fix: one accounting line; reconcile.
5. Leaderboard reversal dismissed as noise although local ~32% -> public 6-12% is a distribution shift; task-level SD
   shrinks near p ~ 0.06. Fix: name the shift; restrict the noise argument to 3.5 vs 7 on the public split.

Unsupported / inconsistent: "five GPU types" (four models appear); "15 full runs" vs 12; "96-100% of every session"
(the table is per run); "2.0-2.3x" computed against the 31.6 proxy, not the 4x L4 sequential run (33.8 / 14.0-15.8 =
2.14-2.41); SD 1.6 applied near the zero boundary.

Unclear to outsiders: draft header + [L4] markers; E4B, 26B-A4B, "teacher"; g31_* agent labels; the 55 / 71 / 58 /
120 task sets; TP4 vs 4 servers TP1, w4a16; whether `context` in the per-call fit is total or cached tokens and whether
the fit is per run.

Highest-impact change: complete the [L4] pair, re-anchor on 33.8, promote the 55-task converted vs nominal comparison
to the abstract in place of 4/10 -> 9/10 and the 0.84-factor check.

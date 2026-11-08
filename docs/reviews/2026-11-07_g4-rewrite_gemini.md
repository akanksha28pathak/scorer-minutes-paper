# G4 Gemini review (pasted by Akanksha, Nov 7; condensed by Claude, scores and objections kept)

Average **4.6 / 5** (G0 3.6, G1 4.1, G1b 4.3): Novelty 4.0, Quality 4.5, Relevance 5.0, Verifiability 4.5, Clarity 5.0.
Checks passed: table 4.1 sums to 56 runs; table 4.3 decode columns match 1000 / (b + d x ctx).

Five weaknesses:
1. Table 4.2 row 3 labelled "(v15)" while the control claims one agent (g31_v12e); fix the label or explain.
2. Section 4 intro: the 7 control setups vs the 24 matrix runs overlap is unclear; suggest "(8 runs, 2 shared with
   the matrix)".
3. Table 4.3 "(min R^2)" unexplained for single-run rows; adapter B's c drops 0.26 -> 0.08 while a rises 0.17 -> 0.24
   (intercept / slope trade-off), so "prefill +41%" invites pushback; keep "decode b +12% (+11% at 24k)".
4. Section 5.1 switches between SD 1.66, 1.52 and 1.13; write "0.9 train task-level SD of a difference (1.3
   run-level SD)".
5. Abstract "11-20%" vs section 5.1 "11% with two runs each": two different 11%s; state both.

Unclear to outsiders: E4B and 26B-A4B definitions were cut; "2 attempts x 5" could read "5 tasks x 2 parallel
attempts"; add how the decode column derives from b and d.

Highest-impact change: fix the (v15) label in table 4.2 and drop the collinear "prefill (a +41%)" claim.

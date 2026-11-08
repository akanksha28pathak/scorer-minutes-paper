# Gemini review, gate G1 (with the six-setup control; Nov 7; pasted by Akanksha)

Average **4.1 / 5** (G0: 3.6): Novelty 3.5, Quality 4.0, Relevance 4.5, Verifiability 4.0, Clarity 4.5.
Statistics in 4.3 checked and correct (McNemar p = 1.0 for (3,3), (1,1), (2,3); (6,1) -> 0.125; 5 / (1.13 x sqrt 2) =
3.1 SD).

Weaknesses:
1. Table 4.1 omits the new setups (H100, L40S, the 56 sequential scorer sessions); its rows sum to 55 runs / 3,472
   sessions, not the claimed 67 / 4,101; scorer row "1 (5)" violates ">= 20 sessions"; abstract says "four GPU models"
   (now six).
2. 4.3 says the cap was set to 7 x f, but 7 x f != wall cap: caps came from the 5-session pre-calibration, f in the table
   is the full-run factor, so realized budgets are 6.3-7.4 scorer min. Say so; it strengthens the point (scores stay at
   19-20 across that range).
3. Stale G0 text: section 7 ("anchor has 5 sessions", "six repeated pairs"), Reproducibility ("81 runs").
4. Section 5.2: unconditional binomial SDs (1.8 / 2.5 of 58) contradict the paper's own task-coin model (local
   task-level SD 1.5-1.7 at 33%); use the community resubmission SD 1.4-1.8: a 3.5-task gap is ~1.6-1.8 SD of a
   difference, still "suggestive, not decisive".
5. 3,837 words vs 3,000; the 4x L4 10-session numbers are narrated four times; cut header, A100-proxy paragraph,
   related work by 35%, section 4.2 to two sentences.

Inconsistencies: run matrix "24 runs" vs table 5.1 (16) + 4.3: say what the rest are; add the A100 v12e run (17) to
table 4.3 for transparency.
Clarity: why 56 sessions for 55 tasks (two parts); header "Task overlap vs scorer (both / scorer-only / run-only)";
which "serving benchmark" in 4.5; define E4B.
Highest impact: reconcile all runs into tables 4.1 and 4.3 and cut under 2,800 words.

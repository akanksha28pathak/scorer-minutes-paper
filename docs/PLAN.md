# Paper track: plan of action and log (scorer-minutes)

Paper: **Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets**
(draft `docs/paper/SCORER_MINUTES.md`; checklist and judge-objection fixes: `1_scorer_minutes_requirements.md`
section 6). Deadline Thu **Nov 12 23:59 UTC**; our target: **submit Tue Nov 10, 18:00 ET**. Team: Akanksha + Claude
(Lily not available for active review).

Keep this file current: tick boxes, add a dated line to the log (newest first). One file for the plan; older versions
are listed in the history table.

| Version | Date | Change |
|---|---|---|
| v1 | Nov 5 | Shortest path to Nov 10 (requirements doc section 5) |
| v2 | Nov 6 | Grant review (Fable agent + competitor notebooks): stay on scorer-minutes; 5 fixes; day plan incl. Lily |
| v3 | Nov 6 11:40 ET | Without Lily; public toolkit repo + notebook live Nov 9-10 (Akanksha) |
| v3.1 | Nov 6 12:00 ET | Gemini review at every step (Akanksha): review gates G0-G5 |

## Decisions

- Scorer-minutes, no pivot (localization/graph papers: ~12 competing public notebooks and our own result there is
  negative; the hint-leak finding, D40, needs a week). Aim: Best New Resource, while scoring high on the average for
  Best Paper.
- Differentiation from the competing paper-track notebooks (cite each): cross-hardware cap conversion validated by a
  control at f ~ 2; local repeated-run noise equal to leaderboard noise; a LoRA adapter changes the speed factor
  (D41); pip-installable library + per-session dataset.
- Only GPU job: the L4 control pair (approved Nov 6). No Vast runs.
- ~~Public toolkit-only GitHub repo~~ **Superseded Nov 7 (Akanksha): everything stays in the main repo, no separate
  repo.** Toolkit `tools/scorer_minutes/`, data `docs/paper/aggregates/`, notebook `tools/scorer_minutes/notebook/`. Open:
  the main repo is private until Dec 2, so the Writeup link is not judge-visible as is. Sole author: Akanksha.
- Reviews: **Gemini review at every step** (Akanksha, Nov 6; gates below) + a fresh-agent cold read at the end.
- Main board: daily submissions continue with prompt experiments; each needs Akanksha's OK.

## Plan of action v3 (times ET)

Decided: scorer-minutes paper (no pivot); public toolkit repo + public Kaggle notebook go live **Nov 9-10**, with the
paper; the L4 control is the only GPU job; main-board submissions continue daily with prompt experiments (each needs
Akanksha's OK). Times ET.

**Nov 6 (Fri)**
- [ ] 19:05 auto: L4 control pair launches (`scripts/paper/l4_control.sh`; logs `~/gemma4_work/logs/kaggle_g31_v12e_l4ctl_{conv15,nom7}.log`); 19:05 auto: queued g31_v15_sft submission. Claude reports both ~21:00.
- [ ] Claude: draft sections to the new anchor (speed factors vs the sequential 4x L4 run, `docs/paper/aggregates/speed_table.md`); section 5 with the pooled CI and the val55 replication (`noise_table.md`); new subsection "A LoRA adapter changes the factor" (D41); figure script `scripts/paper/figures.py` (fig 1: tok/s by setup x concurrency; fig 2: outcome by cap, filled after L4).
- [ ] Akanksha: nothing required.

**Nov 7 (Sat)**
- [ ] Claude (morning): L4 results in: `bash scripts/paper/scorer_minutes_pc.sh`, label the two runs in `run_labels.tsv`, fill the [L4] rows, paired per-task table (both / only converted / only nominal, McNemar), minimal detectable effect, fig 2.
- [ ] Claude: toolkit release staged privately (repo `scorer-minutes`: `tools/scorer_minutes` only + LICENSE Apache-2.0 + README + CITATION); Kaggle notebook (private): pip install, calibrate + timing on synthetic logs, rebuild the speed and noise tables from the aggregate CSVs.
- [ ] Akanksha: approve the day's prompt-experiment submission.
- [ ] **20:00 go / no-go** (below).

**Nov 8 (Sun)**
- [ ] Claude: full rewrite to <= 2,800 words, resource first (problem, toolkit, validation, noise, LoRA, protocol, limits); Writeup-ready markdown + figure PNGs; a number-to-source list (every number -> file).
- [ ] Akanksha: check numbers against that list.
- [ ] 12:00 cut-line check: L4 not graded -> drop the [L4] rows, keep the RTX PRO 6000 control + v9b 4/10 -> 9/10, state the limit.

**Nov 9 (Mon)**
- [ ] Claude: cold judge read by a fresh agent (5 criteria, objections); Gemini review packet (paper + rubric prompt) for Akanksha; fix objections; freeze numbers (tag `paper-v1`).
- [ ] Akanksha: Gemini review; create the Kaggle Writeup draft (title, subtitle, abstract, body, figures, links).
- [ ] 12:00 cut-line check: notebook not working -> repo release only.
- [ ] Publish: Claude audits every file (no competition data, no task text); Akanksha OKs; repo public + release v0.1.0; notebook public; links added to the Writeup.

**Nov 10 (Tue)**
- [ ] Akanksha: **submit the Writeup by 18:00 ET** (ties go to the earliest entry). Claude: check the public links logged out; log in NEXT_STEPS.
- Nov 11-12: buffer; fix only broken links or typos.

**Go / no-go: Nov 7 20:00 ET**, after the L4 pair grades: go if converted vs reference differ by <= 4 tasks; else
rewrite section 4.3 as a negative result ("one factor is not enough at 10 workers; use the per-call fit"). Cut line:
L4 not graded by Nov 8 noon: drop [L4] rows, keep the RTX PRO 6000 control + v9b 4/10 -> 9/10 and state the limit;
notebook not working by Nov 9 noon: ship the repo release only. Always dropped: sequential anchor rerun, TraceLab
contrast, a third figure. Reviewer's estimate: average ~2.9 now, ~3.9 after the plan (judgment, not data).

## Review gates (Gemini at every step)

Workflow: Claude runs `python3 scripts/paper/review_packet.py <step> [--sections ...] [--extra ...] --note "..."`,
which writes `docs/paper-track/reviews/<date>_<step>_packet.md` (rubric + strict-judge instructions + the text). Akanksha
pastes it into Gemini and saves the answer as `reviews/<date>_<step>_gemini.md` (or pastes it into the chat). Claude
logs each objection below with the action taken (fixed / rejected with reason), before the next step starts.
The packets hold only paper text and aggregate numbers, never competition task text or code.

| Gate | When | Step | Packet |
|---|---|---|---|
| G0 | Nov 6 now | baseline score of the current draft | full draft |
| G1 | **ready Nov 7 03:00** | the G0 fixes **plus** the new data: L4 converted / nominal pair, 55-task sequential scorer reference, H100 + L40S conversion checks (Akanksha: no Gemini review before this data is in) | sections 1, 3, 4, 5 |
| G2 | Nov 7 evening (after the go / no-go) | L4 results in section 4.3 | sections 4, 5 |
| G1b | Nov 7 03:40 (extra round, Akanksha) | re-review after the G1 fixes, before the rewrite | full draft |
| G3 | Nov 8 morning | the resource: toolkit README + notebook outline | README + outline + section 3 |
| G4 | Nov 8 evening | the rewrite (<= 2,800 words) | full draft |
| G5 | Nov 9 morning | final, with the fresh-agent cold read | full draft + Writeup fields |

### Objections and actions

| Gate | Objection (Gemini) | Action |
|---|---|---|
| G4 (Gemini 4.6) | 1. Table 4.2 row "RTX PRO 6000 (v15)" contradicts "one agent" | **Fixed**: Claude's bug (a find-and-replace meant for table 4.3 also hit table 4.2); the row is the g31_v12e run (control_43.csv); suffixes removed from 4.2 |
| G4 | 2. Control vs matrix overlap unclear | **Fixed**: "seven setups (8 runs, 2 shared with the matrix)" |
| G4 | 3. "min R^2" header; adapter `a` +41% vs `c` -69% is collinear | **Fixed**: lead-in says calls summed, lowest R^2 over 1-2 runs, decode = 1000 / (b + d x ctx); 4.4 keeps decode b +12% (+10% at 24k), says c and a are not read separately |
| G4 | 4-5. Which SD (1.13 / 1.52 / 1.66) in 5.1 and the abstract's 11-20% | **Fixed**: 11% (run SD) / 20% (task-level, held-out; used for comparisons); pair = 0.9 train task-level SD (1.3 run-level) |
| G4 | E4B / 26B-A4B definitions cut; "2 attempts x 5" label | **Fixed**: definitions restored; "5 workers x 2 parallel attempts". Draft 2,876 words (limit 3,000) |
| G5 Fable review (edits in place) | Abstract used "factor" before defining it; thesis buried at the end; section 4 preamble was bookkeeping; p = 0.13 vs "3 SD" unexplained; 5.2 rank reversal and the serve-check anecdote not traceable to released data | **Accepted** (scores 3/3.5/5/4/2.5 -> est. 4/4/5/4.5/4): idea-first abstract and intro, "nominal" defined, preamble = three subsets / three questions, paired-test power explained; the two untraceable claims cut. Claude follow-ups: twin McNemar now in `control_twin.csv` (6 / 1, p = 0.125) and session lengths in `lora_sessions.csv` (`analysis/lora_sessions.py`: 230.3 vs 258.5 s, x1.122), which corrects the projection to **11.6**-12.1 h; abstract "2.1-2.6x slower per session than the scorer reference" |
| G5 cold read (fresh agent) | 10 number errors: 58 -> **56** full 31B runs (G1 count was wrong; Gemini's check reproduced it); 80 -> 79 31B + 1 12B; flips 4-8 -> 4-6; A100 +-5% -> factors 1.01-1.16; adapter A 18% -> 16% vs its paired base; model share max 0.998; prefill median 2.3% (not 2.7); 2.1-2.6 stated as factors; late/early 0.96-1.05; McNemar sentence scoped | **Fixed** (all verified against speed_table.csv / control_43.csv / noise_long.csv) |
| G5 cold read | Abstract mixes run SD (1.13 -> 11% wrong order) and task-level SD (1.66 -> 20%) | **Fixed**: "SD 1.1-1.7 (run- and task-level): wrong 11-20%"; 5.1 / 4.2 name which SD |
| G5 cold read | Control underpowered: one run per converted setup cannot resolve a ~10% factor error | **Stated** in Limitations; budget-response curve (caps 3.5 / 5 / 10) would need new runs: proposed, not run |
| G5 cold read | Table 4.3 mixes agents; terms undefined (model share, late/early, task-level SD, grading, adapter training) | **Fixed**: agent per row; definitions in 3.2 / 3.3 / 4; adapter B "fine-tuned on solved agent sessions" |
| G5 cold read | Resource under-shown in the paper (no link, no example) | Link in Reproducibility (github.com/akanksha28pathak/scorer-minutes, pending publish); README has examples incl. custom log fields; data card added |
| G1b | 1. Two anomalous rows in table 4.1 (A100 "4 sessions" 16.9; RTX "2 attempts per task" 26.4) | **Fixed**: both ran parallel attempts (eval_agent starts attempts x workers sessions): 3 x 4 = 12 and 2 x 5 = 10 concurrent sessions; labels, `speed_groups.py` and table rows now show the real concurrency; reading (1) cites them as concurrency evidence |
| G1b | 2. Run accounting double-counts (24 + 7) | **Fixed**: "the Section 4.2 control reuses that pair and adds 5 setups in 6 runs" |
| G1b | 3. Novelty: fitted (c, a, b, d) never shown | **Fixed**: new table 4.3 (`analysis/timing_coef_table.py` -> `timing_coefs.csv`): scorer vs 4x L4 x10 vs H100 / L40S / RTX / RTX + adapter B; prose: decode b carries the hardware gap (26.6 -> 62.5 s per 1k at 10 sessions), adapter cost sits in b (+12%) and a (+41%), not overhead |
| G1b | 4. "seven setups" stale | **Fixed**: "six GPU models and 19 serving setups" (19 = rows of speed_groups.csv) |
| G1b | 5. 2.1-2.4x vs 2.13-2.55; "1.02 (anchor)" | **Fixed**: 2.1-2.6x; "1.02 (reference run)" |
| G1b | "from 0.65x to 2.4x the scorer's speed" mixes f and speed | **Fixed**: "at speed factors from 0.65 to 2.40" |
| G1b | McNemar p = 1.0 "for each" wrong with the A100 row (5, 3 -> 0.73) | **Fixed**: "p >= 0.73 for each; p = 1.0 for the four pre-calibrated runs" |
| G1b | Why v_ref stays 33.8 | **Fixed** (3.1): fixed before the control; the 56-session reference measured 33.4 / 32.9, within 2.5% |
| G1b | Length | **Fixed** (G4 rewrite): 2,775 words (`scripts/paper/word_count.py`); 4.2 (v9b anecdote) folded into the control section; related work one paragraph; draft header removed; g31_*, splits, TP4, w4a16 defined |
| G1 | 1. Table 4.1 omits H100 / L40S / sequential scorer; rows do not add up; "four GPU models" | **Fixed**: `analysis/speed_groups.py` -> `speed_groups.csv`; table 4.1 = all 58 full 31B runs (other 15 full runs in the CSV), scorer row 2 runs / 56 sessions; abstract: 103 runs, 4,592 sessions, six GPU models, 13-52 tok/s |
| G1 | 2. 7 x f != wall cap in 4.3 | **Fixed**: caps were set in advance (5-session pre-calibration, earlier run, or guessed 2.13); f is the full-run factor; realized 6.3-7.4 scorer min stated as such; columns renamed |
| G1 | 3. Stale text (section 7, Reproducibility) | **Fixed**: anchor 61 sequential sessions; 7 repeated pairs; 103 runs; new CSVs listed |
| G1 | 4. 5.2 binomial SDs inconsistent with the task-coin model | **Fixed**: uses the resubmission SD 1.4-1.8; gap = 1.4-1.8 SD of a difference |
| G1 | 5. 4,000+ words | Nov 8 rewrite (cuts listed by Gemini adopted: header, A100-proxy paragraph, related work -35%, 4.2 to two sentences, no repeated numbers) |
| G1 | Run matrix 24 runs; A100 v12e missing from 4.3 | **Fixed**: accounting line lists the 24 (14 pool + 2 pair + 8 singles) + 7 control runs; A100 row added to table 4.3 |
| G1 | Clarity: 56 sessions, overlap header, serve check, E4B | **Fixed** (two parts + one rerun session; header renamed; "our vLLM serve check, 12 replayed prompts, most under 15 tokens"; E4B defined) |
| G0 | 1. Circular validation: cross-hardware v12e pair inside the noise pool used to judge it | **Fixed**: `analysis/noise_table.py --pool-exclude g31_v12e` (same-hardware repeats only: train55 SD 1.14, 5 dof; both splits 1.13, CI 0.75-2.31, 7 dof); 17 vs 19 judged against it (0.9 SD). The [L4] pair runs tonight |
| G0 | 2. Three reference speeds; anchor on 5 sessions | **Fixed** in text (all factors vs 33.8; proxy only in an appendix table). **New run**: sequential 4x L4 reference, v12e train55, 1 session, 7 min (`scripts/paper/l4_sequential.sh`, ~8 GPU h, tonight): 55-session anchor + true reference score |
| G0 | 3. 4/10 -> 9/10 headline confounded | Planned Nov 7: lead with the 55-task converted / nominal / sequential-reference trio; v9b stays as the "guessed factor" anecdote |
| G0 | 4. Count discrepancies (81/54/40 runs, 15 vs 12, five GPU types) | **Fixed**: accounting line in section 4 (95 runs; 67 full, 4,101 sessions; run matrix 24 runs; noise pool 14 runs / 7 configs); "four GPU models"; contributions updated |
| G0 | 5. Leaderboard reversal dismissed as noise | **Fixed**: section 5.2 names the local 24-40% vs public 5-17% difficulty shift; noise argument limited to the public gap (binomial SDs 1.8 / 2.5 of 58: about one SD) |
| G0 | "96-100% of every session" | **Fixed**: "of each run's session time" |
| G0 | Clarity: header + markers, E4B / 26B-A4B / teacher, g31_* labels, task sets, TP / w4a16, fit variables | 26B-A4B defined, "teacher" dropped in contributions; the rest in the Nov 8 rewrite (header and markers removed at submission) |

## Log (newest first)

- **Nov 7 15:50** Submission folder `docs/paper/submission/` (Akanksha): body.md with the figures placed (2,944 words), fig1 / fig2 / card.png, FIELDS.md, CHECKLIST.md; built by `scripts/paper/build_submission.py` (rerun after every edit). Adapter submission 58072912 scored 0.15.
- **Nov 7 13:10** Budget-response curve in (g31_v12e, train55, RTX PRO 6000, 5 sessions; `analysis/budget_curve.py`): 3.5 / 7.1 / 10.0 scorer min -> **16 / 18 / 21** solved, 35 / 11 / 0 timed out (7-min control run: 19). Slope 0.76 tasks per scorer minute: a 10% factor error at 7 min costs ~0.5 task. Added to section 4.2 and Limitations; totals now 106 runs / 4,757 sessions, 76 full, 59 full 31B; 2,922 words. Vast cost ~$6.3 over three instances (one host failed the probe, one run refused by the credit guard); credit $9.74, no instances.
- **Nov 7 09:05** Akanksha: sole author; no separate repo (all in the main repo); yes to the budget-curve runs. Chain `scripts/vast/chains/chain_paper_budget.sh` on Vast 59578597 (RTX PRO 6000, Wisconsin, $1.60/h; credit $16.2): g31_v12e train55 at 3.5 / 7 (repeat) / 10 scorer min (caps 3.1 / 6.3 / 9.0), ~4.5 h, ~$7.5. First try found no offer at reliability >= 0.999 (nothing rented); relaunched at >= 0.998.
- **Nov 7 09:00** G4 Gemini review in: **4.6 / 5** (`reviews/2026-11-07_g4-rewrite_gemini.md`); all objections fixed (table above); 2,876 words. G5 packet rebuilt.
- **Nov 7 (overnight, end)** Fable critical review + edit (Akanksha): accepted after checking every changed number; two claims made traceable, one corrected (11.5 -> 11.6 h). Draft **2,800 words**. Packets for Gemini: G3 (resource), G4 (rewrite, full draft), G5 (final: full draft + Writeup fields + release outline). Open for Akanksha: repo owner / authors; a budget-response curve (2-3 runs) to answer the "one run per converted setup" objection.
- **Nov 7 (overnight, later)** Fresh-agent cold read: 3.4 / 5 with 10 number errors, all verified and fixed (table above); draft 2,798 words. Akanksha asked for a Fable critical review that also edits the paper: running.
- **Nov 7 (overnight)** G1b review in: **4.3 / 5** (`reviews/2026-11-07_g1b-fixes_gemini.md`); all objections fixed (table above). **G4 rewrite done** (2,775 words, sections renumbered: 4.2 control, 4.3 timing fit, 4.4 LoRA); one wrong number found while staging (wrong-order 19% / 33% -> 20% / 34%, from noise_table.md). **G3 staged**: `scripts/paper/stage_release.py` -> ~/gemma4_work/release/scorer-minutes (toolkit + LICENSE + CITATION + data/ with anonymized task ids + notebook that executes end to end and reproduces tables 4.1-4.3 and the pooled SD 1.13, CI 0.75-2.31); audit OK. Figure 1 now includes H100 / L40S, full runs only (3.7x spread). Writeup fields: `docs/paper/WRITEUP_FIELDS.md`. Packets G3 / G4 / G5 built for Akanksha's Gemini pass.
- **Nov 7 03:30** G1 Gemini review in: **4.1 / 5** (G0 3.6; `reviews/2026-11-07_g1-data_gemini.md`); 4.3 statistics confirmed correct. All objections fixed except length (Nov 8 rewrite). Draft 4,052 words.
- **Nov 7 03:10** **Correction** (Akanksha spotted 15 solved + 47 "at the cap" > 55): the cap-hit counts used a loose threshold (duration > cap - 20 s). The harness's own field ("Agent exceeded session timeout") gives scorer 11, H100 11, RTX 15, L40S 7, L4 converted 8, **L4 nominal 37** (not 47); no timed-out session was solved in any run. `control_table.py`, section 4.3, abstract and figure 2 now use the harness field; G1 packet rebuilt.
- **Nov 7 03:00** Section 4.3 rewritten as the six-setup control (scorer 19; converted 19-20 at f 0.65-2.40; nominal 15, 47 cut by the cap; paired McNemar vs scorer p = 1.0 for every converted run; converted vs nominal 6 vs 1, p = 0.13; ~3 SD at run level); abstract and introduction lead with it (4/10 -> 9/10 is now only the guessed-factor anecdote of 4.2). `analysis/control_table.py` -> `docs/paper/aggregates/control_43.csv`; figure 2 (`fig2_control.png`). Anchor confirmed by 56 sequential sessions. G1 packet built.
- **Nov 7 01:24** **Scorer reference complete**: g31_v12e on all 55 train55 tasks on Kaggle 4x L4, 1 session at a time, 7 min (exactly the scorer setup): **19/55**, 14 at the cap (part 1 13/32 + part 2 6/23). Speed 33.4 / 32.6 tok/s (56 sessions; anchor confirmed). Converted runs 19-20 vs scorer 19; nominal L4 15. Kaggle quota 13.7 / 30 h used this week.
- **Nov 6 23:15** Sequential reference part 1 (g31_v12e_l4seq55): kernel's default 150-min deadline stopped it at 32/55 (13 solved; 33 sessions at median 33.4 tok/s = the anchor). Part 2 (23 tasks, `splits/seq55_rest.txt`, `--deadline-min 330`) running as g31_v12e_l4seq55b, ~2 h, ~4 GPU h (week total ~13.5 h). Collect after a WSL restart: `NO_PUSH=1 bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/seq55_rest.txt g31_v12e_l4seq55b --workers 1 --deadline-min 330 --eval-args=--max-time-minutes=7`.
- **Nov 6 20:50** **Control data in** (v12e, train55, 55 tasks each; measured factor vs the 33.75 anchor; scorer minutes = wall cap / factor):

  | Setup | Factor | Wall cap | Scorer min | Solved | At cap |
  |---|---|---|---|---|---|
  | A100, 5 sessions (old reference) | 1.08 | 7.0 | 6.5 | 17 | 12 |
  | RTX PRO 6000, 5 sessions, converted | 0.90 | 6.0 | 6.7 | 19 | 15 |
  | H100 SXM, 5 sessions, converted | 0.65 | 4.8 | 7.4 | 19 | 15 |
  | L40S, 5 sessions, converted | 1.63 | 11.7 | 7.2 | 20 | 11 |
  | 4x L4, 10 sessions, converted | 2.40 | 15 | 6.3 | 20 | 10 |
  | 4x L4, 10 sessions, nominal | 2.55 | 7 | 2.7 | **15** | **47** |

  Converted runs: 19-20 on factors 0.65-2.40; unconverted L4: 15 with 47/55 sessions cut. Pair used 4.2 GPU h (not ~7). Sequential scorer reference (55 tasks) launched 20:25 on Kaggle, ~01:00 done. H100 ~$6.3, L40S ~$2.4; no Vast instances left. Aggregates regenerated (101 logged runs).
- **Nov 6 17:52** G1 on hold until the new data is in (Akanksha); packet page marked "not ready". G1 = Nov 7 morning, after the L4 pair, the sequential reference and the H100 / L40S runs are scored and written into sections 4.3 / abstract.
- **Nov 6 17:50** Full Kaggle quota allowed for the paper (Akanksha): the sequential scorer reference is now **all 55 train55 tasks** (`g31_v12e_l4seq55`, ~4.5 h wall, ~16 GPU h; launches after the pair if <= 13 h used). Same 55 tasks as the converted / nominal pair: a clean three-way comparison (scorer / converted / nominal). Collect after a WSL restart: `NO_PUSH=1 bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/train55.txt g31_v12e_l4seq55 --workers 1 --eval-args=--max-time-minutes=7`. Leaves ~7 GPU h for the rest of the week.
- **Nov 6 17:49** Kaggle budget raised to **<= 15 GPU h** (Akanksha): sequential anchor = **20 tasks** (`splits/seq_anchor_20.txt`, ~6 GPU h; launches if <= 9.0 h used after the pair). With the 5-session run: 25 sequential sessions + a scorer reference score on 20 tasks. Collect after a WSL restart: `NO_PUSH=1 bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/seq_anchor_20.txt g31_v12e_l4seq20 --workers 1 --eval-args=--max-time-minutes=7`.
- **Nov 6 17:50** Kaggle budget for paper runs: **<= 10 GPU h** (Akanksha). Measured charge ~3.6 GPU h per wall hour on 4x L4: L4 pair ~7 h; the full 55-task sequential run (~16 h) replaced by an 8-task sequential anchor (`splits/seq_anchor_8.txt`, ~3 h), launched only if the pair leaves room (<= 7.0 h used). Collect after a WSL restart: `NO_PUSH=1 bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/seq_anchor_8.txt g31_v12e_l4seq8 --workers 1 --eval-args=--max-time-minutes=7`. G1 packet published (sections 3-5 + note of the G0 fixes).
- **Nov 6 17:44** Vast, in parallel (Akanksha): calibrate-then-convert on two new GPU types, L40S (59479417) and H100 SXM (59479473), `chain_paper_gpu.sh` (5-task smoke -> factor vs 33.75 -> v12e train55 at 7 x f). Expected done ~20:30 ET, ~$11 total.
- **Nov 6 17:40** G0 Gemini review in (avg **3.6 / 5**; `reviews/2026-11-06_g0-baseline_gemini.md`); fixes above. New Kaggle run queued: sequential 4x L4 reference (19:12 ET, starts when the L4 pair frees a slot, ~8 GPU h). Draft 3,517 words (trim Nov 8).
- **Nov 6 12:40** Draft updated (G1 edits): factors against the sequential 4x L4 anchor everywhere (table 4.1 regrouped
  over 95 runs / 4,306 sessions; v9b factor 2.12, 11 min = 5.2 scorer min; v12e control 6.5 vs 6.7 scorer min); new
  section 4.5 "A LoRA adapter changes the factor" (D41); section 5.1 with the held-out split and the pooled CI. 14 new
  Vast runs labelled in `run_labels.tsv`. Draft 3,344 words (trim on Nov 8).
- **Nov 6 12:00** Review gates G0-G5 added (Gemini at every step); `scripts/paper/review_packet.py`.
- **Nov 6 11:45** This file created; plan v3 moved here from the requirements doc.
- **Nov 6 11:40** Citations checked against the arXiv API: 7 ids real; unverified venues ("ICLR 2026") and TraceLab
  numbers cut; related work cites the competing notebooks. Draft 3,079 words.
- **Nov 6 11:20** Speed table regenerated over 95 logged runs, factors against the sequential 4x L4 run (33.75 tok/s;
  A100 proxy kept as `docs/paper/aggregates/speed_table_a100proxy.md`). Noise: chi-square CI + pooling across splits:
  train55+val55 pooled SD **1.17 (95% CI 0.79-2.25), dof 8**; val55 alone 1.12 / task-level 1.66 matches train55
  1.19 / 1.61. Toolkit: 16 tests pass; fresh-venv `pip install` works.
- **Nov 6 11:18** L4 control pair scheduled for 19:05 ET (`scripts/paper/l4_control.sh`): g31_v12e train55, Kaggle
  4x L4, 10 workers, caps 15 min (converted) and 7 min (nominal), ~4-5 GPU h.
- **Nov 6 11:00** Research-track focus until ~Nov 10 (Akanksha). Grant review done (decisions above).

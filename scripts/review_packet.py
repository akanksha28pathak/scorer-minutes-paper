"""Build a Gemini review packet for one step of the scorer-minutes paper (docs/paper-track/PLAN.md, "Review gates").

  python3 scripts/paper/review_packet.py <step> [--sections "4.3,5"] [--extra FILE ...] [--note "what changed"]

Writes docs/paper-track/reviews/<YYYY-MM-DD>_<step>_packet.md: the judging rubric of the paper track, the reviewer
instructions, what changed in this step, the paper text (whole draft or the named sections) and any extra files
(e.g. a toolkit README or notebook outline). Akanksha pastes the packet into Gemini and saves the answer next to it as
<YYYY-MM-DD>_<step>_gemini.md; Claude then logs each objection in PLAN.md with what was done about it.
The packet holds only paper text and aggregate numbers (no competition task text or code), so it may leave the PC.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper/SCORER_MINUTES.md"
OUT = ROOT / "docs/paper-track/reviews"

RUBRIC = """You are a judge for the Kaggle "Google - The Gemma 4 Developer Agent Paper Track" (judges are Google
researchers in graph ML and agentic software engineering). Submissions are Kaggle Writeups of at most 3,000 words
reporting original research on agentic software engineering. Each is scored 0-5 on five equally weighted criteria:
- Novelty: new insights, deeper understanding, or important properties of existing methods.
- Quality: how general the approach is beyond the competition; does it translate to other problems?
- Relevance: significance and potential impact for software engineering and agentic learning.
- Verifiability: enough information to understand how it works and how data was obtained, analyzed and interpreted.
- Clarity: clear presentation and writing.
Prizes: Best Paper (highest average), Best New Resource (a new dataset, library, tool or software), Best New
Application. This paper targets Best New Resource and a high average."""

ASK = """Review the material below as a strict judge. Answer in this structure:
1. Scores 0-5 for each criterion, one sentence of justification each, and the average.
2. The 5 most serious weaknesses, most serious first. For each: where (section), why a judge would mark it down,
   and a concrete fix that fits in the remaining time (the deadline is days away; no new large experiments).
3. Claims that are not supported by the numbers shown, or numbers that look inconsistent with each other.
4. Anything unclear to a reader outside this competition (jargon, undefined terms, missing setup details).
5. One change that would raise the score the most.
Be specific and quote the text you object to. Do not rewrite the paper."""


def sections(text: str, wanted: list[str]) -> str:
    """Keep the title, abstract and the requested numbered sections (e.g. '4.3', '5')."""
    parts = re.split(r"(?m)^(?=#{2,3} )", text)
    keep = [parts[0]]
    for p in parts[1:]:
        head = p.splitlines()[0]
        m = re.match(r"#{2,3} (\d+(?:\.\d+)?)", head)
        if "Abstract" in head or (m and any(m.group(1) == w or m.group(1).startswith(w + ".") for w in wanted)):
            keep.append(p)
    return "".join(keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", help="short step name, e.g. s1-sections, s2-l4, s3-toolkit, s4-rewrite, s5-final")
    ap.add_argument("--sections", help="comma-separated section numbers to include (default: the whole draft)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra files to append (README, notebook outline, ...)")
    ap.add_argument("--note", default="", help="what changed since the last review")
    a = ap.parse_args()
    text = PAPER.read_text()
    body = sections(text, a.sections.split(",")) if a.sections else text
    words = sum(1 for w in text.split() if re.search(r"[A-Za-z0-9]", w))  # as scripts/paper/word_count.py
    parts = [f"# Review packet: {a.step} ({dt.date.today()})", "", RUBRIC, "", ASK, ""]
    if a.note:
        parts += ["## What changed in this step", "", a.note, ""]
    parts += [f"## Paper ({'sections ' + a.sections if a.sections else 'full draft'}; full draft is {words} words, not counting markdown table pipes)",
              "", body]
    for f in a.extra:
        p = ROOT / f
        parts += ["", f"## Extra: {f}", "", p.read_text()]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{dt.date.today()}_{a.step}_packet.md"
    out.write_text("\n".join(parts) + "\n")
    print(f"{out.relative_to(ROOT)} ({len(re.findall(r'\S+', out.read_text()))} words)")


if __name__ == "__main__":
    main()

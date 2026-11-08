"""Word count of the paper body as a reader sees it (tokens with a letter or digit; markdown pipes and dashes ignored).

  python3 scripts/paper/word_count.py [docs/paper/SCORER_MINUTES.md]
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "docs/paper/SCORER_MINUTES.md"
text = open(path).read()
words = [w for w in text.split() if re.search(r"[A-Za-z0-9]", w)]
table = [w for line in text.splitlines() if line.startswith("|") for w in line.split() if re.search(r"[A-Za-z0-9]", w)]
print(f"{len(words)} words ({len(table)} in tables); limit 3,000")

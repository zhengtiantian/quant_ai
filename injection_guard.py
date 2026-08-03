"""S.1 — defences against indirect prompt injection through retrieved documents.

The exposure R.10 opened: `/api/ask/news` retrieves from 716,074 articles scraped from
the open web and places their body text in the model's context. Nobody has to attack
anything to put words there -- publishing an article that GDELT ingests is enough. The
system prompt then tells the model to answer *only* from those excerpts, which makes an
injected instruction more persuasive rather than less.

It matters more here than in a chat product because R.7 intends to feed retrieval output
into the feature store: a poisoned answer becomes a poisoned position, not a bad sentence.

**Four layers, and they are not equally strong. Being clear about which is which is the
point of this module.**

1. *Structural separation* (in news_rag.py, not here): retrieved text never enters the
   system prompt, only a user message, inside explicit delimiters, labelled as data.
   This is the real defence -- it is about where authority lives, not about what the text
   says.
2. *Sanitisation* (`sanitise`): strip the characters used to smuggle text past a human
   reviewer or to break out of a delimiter -- zero-width, bidi overrides, control codes.
   Deterministic, so it cannot be evaded by rephrasing.
3. *Pattern detection* (`scan`): flag known injection phrasings. **This is the weak
   layer.** A regex list stops copy-pasted attacks and nothing more; anyone willing to
   rewrite a sentence walks through it. It exists to leave a trace and to make the
   frequency measurable, not to block.
4. *Output validation* (`check_answer`): the answer must cite retrieved sources and must
   not read as though it adopted instructions from them.

Treating layer 3 as security would be the mistake. It is reported, not enforced.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Characters that carry no meaning in news text but are used to hide instructions from a
# human reading the same string, or to terminate a delimiter early.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_BIDI = dict.fromkeys(map(ord, "‪‫‬‭‮⁦⁧⁨⁩"), None)

# Deliberately narrow. Every pattern here has to survive the false-positive measurement
# in tools/measure_injection_fp.py against the real corpus -- a rule that fires on
# ordinary financial journalism is worse than no rule, because it trains you to ignore
# the flag.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override", re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+"
        r"(instruction|instructions|prompt|prompts|direction|directions|rule|rules|context)\b", re.I)),
    ("new_instructions", re.compile(
        r"\b(new|updated|revised)\s+(instruction|instructions|system\s+prompt|directive|directives)\s*:", re.I)),
    ("role_reassign", re.compile(
        r"\byou\s+are\s+now\s+(a|an|the)\b|\bfrom\s+now\s+on,?\s+you\s+(are|will|must)\b", re.I)),
    ("role_marker", re.compile(
        r"(^|\n)\s*(system|assistant|user)\s*:\s*\S", re.I)),
    ("fake_delimiter", re.compile(
        r"(^|\n)\s*(-{3,}|={3,}|#{3,})\s*(end\s+of\s+(document|context|excerpt)|"
        r"begin\s+(instruction|system))", re.I)),
    # The bare word "instructions" was the first version and fired 16 times in 40,000
    # articles, every one of them an earnings-call transcript where the operator asks
    # analysts to "repeat your instructions". A qualifier is now required, which keeps
    # "reveal your system prompt" and drops the transcript idiom.
    ("exfiltration", re.compile(
        r"\b(reveal|print|repeat|output|show|display)\s+(me\s+)?(your\s+|the\s+)?"
        r"(system|initial|original|previous|preceding|hidden)\s+"
        r"(prompt|prompts|instruction|instructions|message|messages)\b", re.I)),
    ("tool_directive", re.compile(
        r"\b(call|invoke|execute|run)\s+the\s+\w+\s+(tool|function)\b", re.I)),
]


@dataclass
class Screening:
    text: str
    flags: list[str] = field(default_factory=list)
    removed_chars: int = 0

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def sanitise(text: str) -> tuple[str, int]:
    """Remove characters that exist to hide text or break out of a delimiter.

    Layer 2, and deterministic: rephrasing cannot evade it because it does not look at
    wording. Control characters other than newline and tab are dropped too -- news bodies
    have no legitimate use for them, and they are a standard way to smuggle content past
    a reviewer who is reading the rendered string.
    """
    # Counted, not derived from a length difference. These characters are *replaced*
    # with a space rather than deleted, so a length delta would report 0 no matter how
    # much was neutralised -- a metric that is always zero reads as "nothing happened"
    # and would hide exactly the signal it exists to surface.
    def _is_hidden(ch: str) -> bool:
        return ch not in "\n\t" and (
            ord(ch) in _ZERO_WIDTH or ord(ch) in _BIDI
            or unicodedata.category(ch) in ("Cc", "Cf")
        )

    replaced = sum(1 for ch in text if _is_hidden(ch))
    # Replaced with a space, not deleted. Deleting them was the first version and it
    # helped the attacker: zero-width characters used as word separators
    # ("Ignore<ZWSP>all<ZWSP>previous<ZWSP>instructions") collapse into one unmatchable
    # token, while a model reads the phrase perfectly well through the missing spaces.
    # Neutralising a hiding character must not also destroy the evidence.
    text = "".join(" " if _is_hidden(ch) else ch for ch in text)
    # A long run of delimiter characters is how a document tries to look like the end of
    # the document. Collapse rather than drop, so the text stays readable.
    text = re.sub(r"([-=#_*])\1{9,}", r"\1\1\1", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text, replaced


def scan(text: str) -> list[str]:
    """Layer 3. Returns the names of patterns that matched, never a verdict.

    Reported, not enforced: an article that trips a rule is still shown to the model,
    labelled. Dropping documents on a regex match would let anyone censor coverage of a
    company by publishing one article containing the right sentence.
    """
    return [name for name, pat in _PATTERNS if pat.search(text)]


def screen(text: str) -> Screening:
    cleaned, removed = sanitise(text or "")
    return Screening(text=cleaned, flags=scan(cleaned), removed_chars=removed)


def check_answer(answer: str, n_sources: int, cited: list[int]) -> list[str]:
    """Layer 4 — cheap structural checks on what came back.

    Not a claim that the answer is correct; that is R.11's job with a grounding gate.
    These only catch an answer that stopped behaving like an answer.
    """
    problems = []
    if n_sources > 0 and not cited:
        problems.append("no_citations")
    if re.search(r"\b(as instructed|per your instruction|following the instruction"
                 r"s? in the (article|document|excerpt))\b", answer, re.I):
        problems.append("adopted_document_instruction")
    if re.search(r"\b(system prompt|my instructions are|i am instructed to)\b", answer, re.I):
        problems.append("prompt_disclosure")
    return problems

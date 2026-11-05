"""Typo-tolerant questions (owner 2026-10-15).

Before a question is understood, misspelled security words and chat short forms are
corrected toward the assistant's own vocabulary:

  * short forms: "whats" -> "what is", "pls" -> "please", "privesc" -> "privilege
    escalation", "t1059" -> "T1059";
  * misspellings of DOMAIN words only (attack / tactic / technique names, ATT&CK
    aliases, question cue words such as "happening"): "brut force" -> "brute force",
    "pishing" -> "phishing", "ransomwear" -> "ransomware".

Ordinary words, names, IP addresses and technique ids are never changed: a word is
corrected only if it is unknown to the vocabulary AND close to a domain word
(difflib ratio >= ``UNSURE``). A ratio below ``SURE`` marks the correction as
uncertain, so the answer adds "Did you mean ...?". Pure Python, cached, ~0.1 ms per
word.
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache

SURE = 0.85  # at or above: corrected silently
UNSURE = 0.75  # between UNSURE and SURE: corrected, but "Did you mean ...?" is shown
MIN_LEN = 4  # shorter words are left alone (too many near neighbours)

SHORT_FORMS = {
    "whats": "what is", "wats": "what is", "wat": "what", "whos": "who is",
    "hows": "how is", "hw": "how", "pls": "please", "plz": "please", "abt": "about",
    "u": "you", "r": "are", "ur": "your", "thx": "thanks",
    "privesc": "privilege escalation", "exfil": "exfiltration", "sqli": "sql injection",
    "creds": "credentials", "c&c": "command and control", "cnc": "command and control",
    "bf": "brute force", "pw": "password", "pwd": "password", "msg": "message",
    "isnt": "is not", "dont": "do not", "cant": "cannot", "didnt": "did not",
}  # fmt: skip

# question words the assistant relies on (corrected toward when misspelled)
CUE_WORDS = (
    "happening", "happened", "explain", "describe", "attack", "attacks", "attacker",
    "attacking", "incident", "incidents", "alert", "alerts", "defend", "defense",
    "defence", "prevent", "protect", "detect", "detected", "ongoing", "today",
    "yesterday", "results", "hypothesis", "hypotheses", "accuracy", "source",
    "sources", "suspicious", "malicious", "evidence", "logs", "network", "password",
    "passwords", "account", "accounts", "server", "servers", "firewall", "malware",
    "ransomware", "phishing", "spearphishing", "keylogger", "botnet", "trojan",
    "exploit", "vulnerability", "injection", "scanning", "credential", "credentials",
)  # fmt: skip

# common English words that must never be "corrected" into a security term
COMMON = (
    "a about above after again against all also am an and any are as at be because "
    "been before being below between both but by can could did do does doing down "
    "during each few for from further had has have having he her here hers him his "
    "how i if in into is it its itself just me more most my no nor not now of off on "
    "once only or other our ours out over own same she should so some such than that "
    "the their them then there these they this those through to too under until up "
    "very was we were what when where which while who whom why will with would you "
    "your yours yes please thanks thank show tell give list find check look need want "
    "know think going still right last first next week hour hours minute minutes day "
    "days time times many much number count stop block happen happens done doing "
    "someone anyone something anything everything nothing people person user users "
    "host hosts machine computer laptop office home company team today now ever new "
    "old good bad best worst help hello hey hi kind type types example examples way "
    "ways work works working start started started real false true sure maybe should "
    "could would might must shall again ago since between across inside outside "
    "happen happened system systems file files folder folders email emails mail "
    "website websites internet traffic data report reports summary details detail "
    "result question questions answer answers seen sign signs trace traces"
).split()

_WORD = re.compile(r"[A-Za-z][A-Za-z'&]*")
_TID = re.compile(r"\bt(\d{4}(?:\.\d{3})?)\b", re.I)


@lru_cache(maxsize=1)
def _vocab() -> tuple[frozenset[str], tuple[str, ...]]:
    """(every known word, domain words corrections may go to)."""
    from src.knowledge.attack_details import (
        DETAILS,
        TACTIC_SYNONYMS,
        TECHNIQUE_SYNONYMS,
        display,
    )
    from src.knowledge.explain import _ALIASES, kb
    from src.nlp.intent import ATTACK_SYNONYMS, TEMPLATES

    def words(text: str) -> list[str]:
        return [w.lower() for w in _WORD.findall(text)]

    domain: set[str] = set(CUE_WORDS)
    for phrase in [*ATTACK_SYNONYMS, *TACTIC_SYNONYMS, *TECHNIQUE_SYNONYMS, *_ALIASES]:
        domain.update(words(phrase))
    for name in DETAILS:
        domain.update(words(display(name)))
    for v in kb().values():
        domain.update(words(v["name"]))
    domain = {w for w in domain if len(w) >= MIN_LEN}
    known = set(domain) | set(COMMON) | set(SHORT_FORMS)
    for temps in TEMPLATES.values():
        for t in temps:
            known.update(words(re.sub(r"\{[^}]*\}", " ", t)))
    # searches and reports ("show failed logins for bob", "make a report of today"):
    # their words are correct as typed and must never be "fixed" to something else
    from src.soc import search

    for phrase in [*search.KINDS, *search.SOURCES, *search.TACTICS, *search._STOP]:
        known.update(words(phrase.replace("-", " ")))
    known.update(_SEARCH_WORDS)
    return frozenset(known), tuple(sorted(domain))


_SEARCH_WORDS = frozenset(
    """failed failure failures successful success login logins logon logons logged
    event events log logs alert alerts show find search list report reports generate
    make create weekly daily monthly hours minutes seconds process processes command
    commands powershell cmd connection connections request requests blocked denied
    file files hash host hosts user users account accounts after before since between
    today yesterday week month entries lines activity""".split()
)


def correct(q: str) -> tuple[str, list[dict[str, object]]]:
    """(corrected question, changes). Each change: {"from", "to", "sure"}."""
    known, domain = _vocab()
    changes: list[dict[str, object]] = []

    def fix_tid(m: re.Match[str]) -> str:
        return "T" + m.group(1)

    q2 = _TID.sub(fix_tid, q)

    def fix(m: re.Match[str]) -> str:
        w = m.group(0)
        low = w.lower().strip("'")
        # never touch words glued to digits/dots (IPs, versions, hostnames)
        s, e = m.start(), m.end()
        if (s > 0 and (q2[s - 1].isdigit() or q2[s - 1] in "._-/\\@")) or (
            e < len(q2) and (q2[e].isdigit() or q2[e] in "._-/\\@")
        ):
            return w
        if low in SHORT_FORMS and low not in known - set(SHORT_FORMS):
            to = SHORT_FORMS[low]
            changes.append({"from": w, "to": to, "sure": True})
            return to
        if len(low) < MIN_LEN or low in known:
            return w
        if s > 0 and w[:1].isupper():  # capitalised mid-sentence: a name, leave it
            return w
        best = difflib.get_close_matches(low, domain, n=1, cutoff=UNSURE)
        if not best:
            return w
        ratio = difflib.SequenceMatcher(None, low, best[0]).ratio()
        changes.append({"from": w, "to": best[0], "sure": ratio >= SURE})
        return best[0]

    out = _WORD.sub(fix, q2)
    return out, changes


def off_topic(q: str) -> bool:
    """True when more than half of the question's content words are unknown to the
    assistant's vocabulary ("what is the weather today", "tell me a joke"). Such a
    question is answered with what the assistant covers instead of being "corrected"
    toward a security word."""
    known = _vocab()[0]
    content = [w.lower().strip("'") for w in _WORD.findall(q)]
    content = [w for w in content if len(w) >= 3 and w not in COMMON]
    if not content:
        return False
    _, domain = _vocab()

    def unknown(w: str) -> bool:  # a confident misspelling ("brut forse") is not unknown
        if w in known:
            return False
        best = difflib.get_close_matches(w, domain, n=1, cutoff=UNSURE)
        return not best or difflib.SequenceMatcher(None, w, best[0]).ratio() < SURE

    return 2 * sum(map(unknown, content)) > len(content)


def did_you_mean(corrected: str, changes: list[dict[str, object]]) -> str | None:
    """One line when any correction was uncertain, else None."""
    unsure = [c for c in changes if not c["sure"]]
    if not unsure:
        return None
    fixed = ", ".join(f"*{c['from']}* → *{c['to']}*" for c in unsure)
    return f"🔤 Did you mean *{corrected}*? I read {fixed}."

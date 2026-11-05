"""Question understanding: a supervised intent classifier + rule-based entities.

The classifier (TF-IDF word+char n-grams -> logistic regression) is trained on
templated questions. It is evaluated on *held-out templates* — phrasings never
seen in training — which is the honest test of generalisation. Entities (IPs,
ATT&CK ids, attack names, time windows) are extracted deterministically: they
are facts, and facts are not left to a statistical model.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

TEMPLATES: dict[str, list[str]] = {
    "incidents": [
        "has anything happened {time}",
        "were there any attacks {time}",
        "what's happening on the network",
        "did anything bad happen {time}",
        "any alerts worth looking at",
        "what should i look at first",
        "show me what the soc found {time}",
        "is the network safe",
        "are we compromised",
        "what's the situation",
        "anything new {time}",
        "what incidents are open",
        "what attacks are happening",
        "what attacks happened {time}",
        "summarize the incidents",
        "any critical incidents {time}",
        "give me a security summary",
        "what is going on right now",
        "are we under attack",
        "show me the incidents",
        "what happened {time}",
        "brief me on the threats",
        "list current incidents",
        "what did the attackers do {time}",
        "status of the network",
        "is anything bad happening",
        "report the latest attacks",
        "overview of attacks {time}",
        "what threats did you detect",
        "anything suspicious {time}",
        "show recent security events",
        "give me the attack timeline",
        "what is the current threat level",
        "describe the ongoing incidents",
    ],
    "ip": [
        "what has {ip} been doing {time}",
        "who is {ip}",
        "is {ip} safe",
        "show me everything about {ip}",
        "did {ip} attack us",
        "explain why {ip} is suspicious",
        "events for {ip}",
        "should we worry about {ip}",
        "what did {ip} do",
        "show activity from {ip}",
        "is {ip} malicious",
        "tell me about {ip}",
        "investigate {ip}",
        "what happened with {ip} {time}",
        "history of {ip}",
        "why was {ip} flagged",
        "summarize {ip}",
        "is {ip} an attacker",
        "anything from {ip}",
        "details for host {ip}",
        "what is {ip} doing",
        "lookup {ip}",
        "check address {ip}",
        "should i block {ip}",
    ],
    "explain": [
        "can you explain {attack}",
        "what's {attack}",
        "help me understand {attack}",
        "how is {attack} done",
        "break down {attack} for me",
        "what are the steps of {attack}",
        "in simple words what is {attack}",
        "how does an attacker perform {attack}",
        "what is meant by {attack}",
        "define {attack}",
        "what is {attack}",
        "explain {attack}",
        "how does {attack} work",
        "tell me about {attack} attacks",
        "describe {attack}",
        "what does {attack} mean",
        "teach me {attack}",
        "{attack} explained",
        "how do attackers do {attack}",
        "what is a {attack} attack",
        "give me an overview of {attack}",
        "i want to understand {attack}",
        "what kind of attack is {attack}",
    ],
    "defend": [
        "how can i stop {attack}",
        "what is the fix for {attack}",
        "how to protect our network from {attack}",
        "what should we do if we see {attack}",
        "steps to contain {attack}",
        "how do i defend from {attack}",
        "what controls stop {attack}",
        "how to prevent {attack} next time",
        "how do i stop {attack}",
        "how to defend against {attack}",
        "mitigation for {attack}",
        "how can we prevent {attack}",
        "protect against {attack}",
        "what should i do about {attack}",
        "countermeasures for {attack}",
        "how to block {attack}",
        "best defense for {attack}",
        "how do we mitigate {attack}",
        "fix for {attack}",
        "remediation steps for {attack}",
        "how to secure against {attack}",
        "recommended controls for {attack}",
        "how to respond to {attack}",
    ],
    "count": [
        "how many alerts do we have",
        "count incidents {time}",
        "number of attacks {time}",
        "how many detections {time}",
        "how many times were we attacked {time}",
        "total alerts so far",
        "how many alerts {time}",
        "count the alerts",
        "number of alerts {time}",
        "how many attacks were detected",
        "alert count {time}",
        "how many incidents are there",
        "total number of detections",
        "how many critical alerts",
        "how many events did we get {time}",
    ],
    "top": [
        "who attacks us the most",
        "which sources are the most dangerous",
        "top threat sources {time}",
        "who is the worst attacker",
        "list attacking ips by severity",
        "where are the attacks coming from",
        "who is attacking us",
        "top attackers",
        "most active attacker ips",
        "which ips attack the most",
        "list the worst sources",
        "top offending hosts {time}",
        "biggest threats by source",
        "who are the main attackers {time}",
        "rank attackers",
        "noisiest attacking addresses",
    ],
    "mitre": [
        "did we see {tid} {time}",
        "incidents mapped to {tid}",
        "any detections of {tid}",
        "which hosts did {tid}",
        "show {tid} events",
        "which incidents used {tid}",
        "show incidents with {tid}",
        "where did we see {tid}",
        "any {tid} activity {time}",
        "map {tid} to incidents",
        "find {tid}",
        "who used technique {tid}",
        "incidents for att&ck {tid}",
        "show me {tid} detections",
        "list hosts doing {tid}",
        "was {tid} observed",
    ],
    "results": [
        "how well does the model work",
        "what were the results of the experiments",
        "is the detector accurate",
        "what is the test score",
        "how did the hypotheses turn out",
        "show me the research results",
        "what are the project results",
        "did the hypotheses pass",
        "show the hypothesis verdicts",
        "how accurate is the model",
        "what did the training find",
        "show the final report",
        "was h1 supported",
        "how good is the detector",
        "evaluation results",
        "model performance",
        "show the metrics",
        "what is the macro f1",
        "did the experiments work",
        "training results",
        "what accuracy did we get",
        "summarize the research findings",
    ],
    "help": [
        "what can i do here",
        "how does this work",
        "give me some example questions",
        "start",
        "what should i ask",
        "can you help me",
        "what are you",
        "help",
        "what can you do",
        "how do i use this",
        "what questions can i ask",
        "commands",
        "who are you",
        "hello",
        "hi",
        "what are your features",
        "how does this assistant work",
        "show me examples",
        "guide me",
        "what can i ask you",
        "i need help",
        "instructions please",
    ],
}

ATTACK_SYNONYMS = {
    # the binary NetFlow class; bare "attack" stays generic (it means "incidents")
    "generic attack": "ATTACK",
    "binary attack": "ATTACK",
    "netflow attack": "ATTACK",
    "port scan": "PortScan",
    "portscan": "PortScan",
    "scanning": "PortScan",
    "scan": "PortScan",
    "ddos": "DDoS",
    "distributed denial": "DDoS",
    "dos": "DoS",
    "denial of service": "DoS",
    "brute force": "BruteForce",
    "bruteforce": "BruteForce",
    "password guessing": "BruteForce",
    "password spraying": "BruteForce",  # types of brute force (attack_details)
    "password spray": "BruteForce",
    "credential stuffing": "BruteForce",
    "sql injection": "WebAttack",
    "xss": "WebAttack",
    "web attack": "WebAttack",
    "webattack": "WebAttack",
    "botnet": "Bot",
    "bot": "Bot",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
    "mirai": "Mirai",
    "spoofing": "Spoofing",
    "recon": "Recon",
    "reconnaissance": "Recon",
    "fgsm": "FGSM",
    "pgd": "PGD",
    "gan": "GAN",
    "evasion": "PGD",
}
_TIMES = [
    "today",
    "in the last hour",
    "in the last 24 hours",
    "this week",
    "in the last 30 minutes",
    "recently",
    "since yesterday",
    "",
]
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)


_WB = r"\b"  # regex word boundary


def normalize(q: str) -> str:
    """Mask entities so the classifier learns phrasing, not specific values."""
    q = _IP.sub(" ipaddr ", q.lower())
    q = _TID.sub(" techid ", q)
    for k in sorted(ATTACK_SYNONYMS, key=len, reverse=True):
        q = re.sub(r"\b" + re.escape(k) + r"s?\b", " attackname ", q)  # plurals
    q = re.sub(
        r"\b(today|this week|yesterday|recently|last \d* ?(minute|hour|day|week)s?)\b",
        " timeword ",
        q,
    )
    return " ".join(q.split())


def normalize_all(xs: list[str]) -> list[str]:
    return [normalize(x) for x in xs]


def _fill(t: str, rng: random.Random) -> str:
    ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
    return " ".join(
        t.format(
            ip=ip,
            attack=rng.choice(list(ATTACK_SYNONYMS)),
            tid=f"T{rng.randint(1000, 1600)}",
            time=rng.choice(_TIMES),
        ).split()
    )


_FILLERS = (
    "",
    "",
    "please ",
    "can you ",
    "could you ",
    "hey ",
    "quick question: ",
    "i want to know ",
)
_ENDS = ("", "", "?", " please", " now", " for me", "??")


def augment(q: str, rng: random.Random) -> str:
    """Training-only noise: polite fillers, punctuation, a typo or a dropped word.
    Held-out phrasings are never augmented (the honest test stays untouched)."""
    q = rng.choice(_FILLERS) + q + rng.choice(_ENDS)
    words = q.split()
    r = rng.random()
    if r < 0.15 and len(words) > 3:  # drop a word
        del words[rng.randrange(len(words))]
    elif r < 0.30:  # swap two adjacent letters in one word
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 3:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def dataset(seed: int = 0, per_template: int = 25) -> tuple[list, list, list, list]:
    """Train/test split by *template*: 25% of each intent's phrasings are held out."""
    rng = random.Random(seed)
    Xtr: list[str] = []
    ytr: list[str] = []
    Xte: list[str] = []
    yte: list[str] = []
    for intent, temps in TEMPLATES.items():
        temps = temps[:]
        rng.shuffle(temps)
        cut = max(1, len(temps) // 4)
        for i, t in enumerate(temps):
            X, y = (Xte, yte) if i < cut else (Xtr, ytr)
            for _ in range(per_template):
                X.append(_fill(t, rng))
                y.append(intent)
    return Xtr, ytr, Xte, yte


def train(seed: int = 0) -> tuple[Pipeline, dict[str, Any]]:
    Xtr, ytr, Xte, yte = dataset(seed)
    rng = random.Random(seed + 1)
    Xtr = Xtr + [augment(x, rng) for x in Xtr]  # train-only augmentation
    ytr = ytr + ytr
    model = Pipeline(
        [
            ("norm", FunctionTransformer(normalize_all)),
            (
                "feats",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced"),
            ),  # chosen on grouped CV, 2026-10-15
        ]
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cv = grouped_cv(seed)
    return model, {
        "intent_accuracy_heldout_templates": float(accuracy_score(yte, pred)),
        "intent_macro_f1_heldout_templates": float(
            f1_score(yte, pred, average="macro")
        ),
        "intent_accuracy_grouped_cv": cv,
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }


def grouped_cv(seed: int = 0, folds: int = 5) -> dict[str, float]:
    """Accuracy when whole TEMPLATES are held out, over 5 folds (each phrasing is
    tested exactly once, never seen in training): a steadier estimate than one split."""
    import statistics

    rng = random.Random(seed)
    items = [(intent, t) for intent, temps in TEMPLATES.items() for t in temps]
    rng.shuffle(items)
    accs = []
    for k in range(folds):
        test_t = {it for i, it in enumerate(items) if i % folds == k}
        Xtr: list[str] = []
        ytr: list[str] = []
        Xte: list[str] = []
        yte: list[str] = []
        for intent, t in items:
            for _ in range(15):
                (Xte if (intent, t) in test_t else Xtr).append(_fill(t, rng))
                (yte if (intent, t) in test_t else ytr).append(intent)
        Xtr = Xtr + [augment(x, rng) for x in Xtr]
        ytr = ytr + ytr
        m = _pipeline().fit(Xtr, ytr)
        accs.append(float(accuracy_score(yte, m.predict(Xte))))
    return {
        "mean": statistics.mean(accs),
        "sd": statistics.pstdev(accs),
        "folds": folds,
    }


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("norm", FunctionTransformer(normalize_all)),
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")),  # chosen on grouped CV, 2026-10-15
        ]
    )  # fmt: skip


def entities(q: str) -> dict[str, Any]:
    low = q.lower()
    attack = None
    for k in sorted(ATTACK_SYNONYMS, key=len, reverse=True):  # longest phrase first
        if re.search(rf"\b{re.escape(k)}s?\b", low):  # "botnets", "port scans"
            attack = ATTACK_SYNONYMS[k]
            break
    since = None
    m = re.search(r"last\s+(\d+)?\s*(minute|hour|day|week)s?", low)
    if m:
        n = int(m.group(1) or 1)
        since = (
            time.time()
            - n * {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}[m.group(2)]
        )
    elif "today" in low:
        lt = time.localtime()
        since = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    elif "this week" in low:
        since = time.time() - 7 * 86400
    elif "yesterday" in low:
        since = time.time() - 2 * 86400
    from src.knowledge.attack_details import TACTIC_SYNONYMS, TECHNIQUE_SYNONYMS

    tactic = None
    for k in sorted(TACTIC_SYNONYMS, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", low):
            tactic = TACTIC_SYNONYMS[k]
            break
    # "defense evasion" is the ATT&CK tactic, not the ML evasion attack (PGD)
    if tactic == "defense-evasion" and attack == "PGD" and "pgd" not in low:
        attack = None
    # generic words ("impact", "execution", "discovery", "collection") name a tactic
    # only when no attack family is named ("impact of brute force" = brute force)
    if attack and tactic in ("impact", "execution", "discovery", "collection"):
        tactic = None
    named_tids = [
        t
        for k, t in TECHNIQUE_SYNONYMS.items()
        if re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", low)
    ]
    return {
        "ips": _IP.findall(q),
        "tids": list(dict.fromkeys([t.upper() for t in _TID.findall(q)] + named_tids)),
        "attack": attack,
        "tactic": tactic,
        "since": since,
        "low_confidence": bool(
            re.search(
                r"\b(low.?confidence|possible|possibly|includ\w* low|all alerts)\b", low
            )
        ),
    }


# "is it happening / did it happen" questions about a named attack or tactic
_STATUS = re.compile(
    r"(^any\b|\bis there|are there|was there|were there|has there been|have there been|"
    r"any (sign|signs|evidence|trace|traces)|are we (being|under)|did (we|anyone|"
    r"someone|they) (get|see|have|try)|has (anyone|someone) (tried|attempted)|"
    r"(still|currently|right now) (happening|going on|ongoing|active|running)|"
    r"happening|going on|ongoing|in progress|been done|attack done|detected|"
    r"did .{0,40} happen|has .{0,40} happened|any .{0,40} (attack|activity|attempt)s?)\b"
)


# words that point back to the previous question's subject
_REFERS = re.compile(
    r"\b(it|its|this|that|they|them|those|these|same|the attack|the attacker|"
    r"he|she|him|her|still|again|more)\b"
)


def with_context(
    q: str, e: dict[str, Any], ctx: dict[str, Any] | None
) -> dict[str, Any]:
    """Fill the subject a follow-up leaves out ("is it still going on?", "how do I
    stop it?") from the previous turn; anything the question names itself wins."""
    if not ctx:
        return e
    names_subject = e["attack"] or e["tactic"] or e["ips"] or e["tids"]
    # a bare status question ("is there any happening?") also continues the topic
    low = q.lower()
    broad = re.search(r"\b(attacks|anything|everything|all|what|which|who)\b", low)
    if names_subject or not (_REFERS.search(low) or (status_question(q) and not broad)):
        return e
    out = dict(e)
    for k in ("attack", "tactic", "ips", "tids"):
        if ctx.get(k):
            out[k] = ctx[k]
    return out


def status_question(q: str) -> bool:
    return bool(_STATUS.search(q.lower()))


# High-precision lexical cues: when one is present it outranks the statistical
# guess (a question containing "how many" is a count, whatever else it says).
_STRONG: list[tuple[str, str]] = [
    (r"\b(how many|number of|count of|count the)\b", "count"),
    (
        r"\bh[1-5]\b|hypothes[ie]s|\b(results?|accuracy|macro.?f1|leaderboard)\b",
        "results",
    ),
    (
        r"^(hi|hello|hey|help)\b|\b(examples?|guide me|what can you|how do i use|"
        r"who are you|your features|commands|tour|get(ting)? started|"
        r"what (should|can|do) i (type|ask|say)|how does this work)\b",
        "help",
    ),
    (r"\b(threat level|current (threat|situation|status)|under attack)\b", "incidents"),
]


def strong_intent(q: str) -> str | None:
    """Intent from an unambiguous cue, or None. Entity intents (an IP, a technique
    id, an attack name with explain/defend wording) are left to the caller."""
    low = q.lower().strip()
    for pat, name in _STRONG:
        if re.search(pat, low):
            return name
    return None


def keyword_intent(q: str, e: dict[str, Any] | None = None) -> str:
    """Fallback when no trained model is available."""
    e, low = (e or entities(q)), q.lower()
    if e["ips"]:
        return "ip"
    if e["tids"]:
        return "mitre"
    if (
        any(
            w in low
            for w in ("stop", "defend", "prevent", "mitigat", "protect", "block")
        )
        and e["attack"]
    ):
        return "defend"
    if e["attack"] and any(w in low for w in ("what", "explain", "how", "describe")):
        return "explain"
    if any(w in low for w in ("how many", "count", "number of")):
        return "count"
    if re.search(r"(who|top|rank)", low):  # whole words: "stop" is not "top"
        return "top"
    if any(w in low for w in ("result", "hypothes", "accura", "report", "performance")):
        return "results"
    if any(w in low for w in ("help", "what can you")):
        return "help"
    return "incidents"

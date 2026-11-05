"""Training text for ShieldLM: incident facts -> summary pairs, and knowledge text.

Data-to-text with *delexicalisation*: facts that vary without limit (IP addresses,
technique names, event counts, sensor lists) are replaced by slot tokens
(``<IP>``, ``<T1>``..``<T5>``, ``<N>``, ``<SENSORS>``) in both the prompt and the
summary. The model learns to write about slots; ``fill`` puts the real values back.
So a sentence can be clumsy, but an IP or a technique can never be invented.

Three control tokens drive RLCD (Yang et al., 2023, "RLCD: Reinforcement Learning
from Contrastive Distillation"):
  <good> prompt -> a faithful, complete summary
  <bad>  prompt -> a contrastively corrupted one (vague, missing/out-of-order steps,
                   wrong severity, repetition, mismatched advice)
  <sum>  prompt -> an unaligned mixture of both: the policy RLCD then aligns.
"""

from __future__ import annotations

import random
import re
from typing import Any

from src.soc.correlate import TECHNIQUES, tactic_rank

LEVELS = ["low", "medium", "high", "critical"]
MAX_T = 5
SLOTS = [
    "<IP>",
    "<LEVEL>",
    "<N>",
    "<SENSORS>",
    *[f"<T{i}>" for i in range(1, MAX_T + 1)],
]
CONTROL = ["<good>", "<bad>", "<sum>", "<kb>"]
CORROBORATION = ("independent sensors agree", "more than one sensor confirms")

REC = {
    "reconnaissance": "restrict what is exposed to the internet",
    "discovery": "close unused ports and segment the network",
    "initial-access": "patch the exposed service and put it behind a waf",
    "execution": "isolate the host and review the executed commands",
    "persistence": "remove new accounts and services",
    "privilege-escalation": "remove the gained privileges and patch the escalation path",
    "lateral-movement": "disable remote admin tools between hosts and reset used credentials",
    "collection": "check which data was accessed and tighten its permissions",
    "defense-evasion": "forward logs to a central siem so they cannot be erased",
    "credential-access": "reset the exposed credentials and enforce mfa",
    "command-and-control": "isolate the host and block its outbound traffic",
    "exfiltration": "block the destination and check what data left",
    "impact": "protect availability with upstream rate limiting and offline backups",
}


# --- facts ---------------------------------------------------------------------


def random_facts(rng: random.Random) -> dict[str, Any]:
    """A plausible incident: 1-5 techniques ordered along the kill chain."""
    k = rng.choice([1, 1, 2, 2, 3, 3, 4, 5])
    techs = rng.sample(sorted(TECHNIQUES), k)
    techs.sort(key=lambda t: tactic_rank(TECHNIQUES[t][1]))
    tactics = list(dict.fromkeys(TECHNIQUES[t][1] for t in techs))
    sensors = rng.sample(
        ["detector", "suricata", "zeek", "winsec", "syslog", "firewall"],
        rng.choice([1, 1, 2, 3]),
    )
    level = rng.choice(LEVELS)
    return {
        "ip": f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "level": level,
        "n": rng.randint(1, 500),
        "sensors": sensors,
        "techniques": [f"{TECHNIQUES[t][0]} ({t})" for t in techs],
        "tactics": tactics,
        "corroborated": len(sensors) >= 2,
    }


def facts_from_incident(inc: dict[str, Any]) -> dict[str, Any]:
    """Correlator incident -> facts dict (technique path kept in time order)."""
    techs = inc["techniques"][:MAX_T]
    return {
        "ip": inc["src_ip"],
        "level": inc["level"],
        "n": inc["n_events"],
        "sensors": inc["sources"] or ["logs"],
        "techniques": [f"{t['name']} ({t['id']})" for t in techs],
        "tactics": list(
            dict.fromkeys(t["tactic"] for t in techs if t["tactic"] in REC)
        ),
        "corroborated": inc.get("corroborated", False),
    }


def prompt_text(f: dict[str, Any], control: str = "<sum>") -> str:
    """Delexicalised prompt. Only slots present here may appear in the output."""
    ts = " ".join(f"<T{i}>" for i in range(1, len(f["techniques"]) + 1))
    return (
        f"{control} src <IP> level <LEVEL> path {ts}tactics {' '.join(f['tactics'])} "
        f"sensors <SENSORS> events <N> {'corroborated' if f['corroborated'] else 'single'} <sep>"
    )


# --- reference (good) and contrastive (bad) summaries ----------------------------


def _path(k: int, rng: random.Random) -> str:
    t = [f"<T{i}>" for i in range(1, k + 1)]
    if k == 1:
        return rng.choice(
            [f"the activity matches {t[0]} .", f"it was identified as {t[0]} ."]
        )
    middle = " , then ".join(t[1:-1])
    return rng.choice(
        [
            f"observed attack path : {' , then '.join(t)} .",
            f"the attack started with {t[0]} and continued with {' and '.join(t[1:])} .",
            # mentions stay in time order (the faithfulness check enforces it)
            f"the attacker moved from {t[0]}"
            + (f" through {middle}" if middle else "")
            + f" to {t[-1]} .",
        ]
    )


def good_summary(f: dict[str, Any], rng: random.Random) -> str:
    lvl = "<LEVEL>"
    parts = [
        rng.choice(
            [
                f"{lvl} incident from <IP> .",
                f"<IP> triggered a {lvl} incident .",
                f"a {lvl} severity attack was traced to <IP> .",
            ]
        ),
        _path(len(f["techniques"]), rng),
        rng.choice(
            [
                "<N> events from <SENSORS> support this .",
                "it is backed by <N> events seen by <SENSORS> .",
            ]
        ),
    ]
    if f["corroborated"]:
        parts.append(
            rng.choice(
                [
                    "independent sensors agree , so a false alarm is unlikely .",
                    "more than one sensor confirms it .",
                ]  # keep in sync with CORROBORATION
            )
        )
    recs = [REC[t] for t in f["tactics"][-2:]] or ["investigate the source"]
    parts.append(
        f"recommended action : block <IP> and {' and '.join(dict.fromkeys(recs))} ."
    )
    return " ".join(parts)


def bad_summary(f: dict[str, Any], rng: random.Random) -> str:
    """Contrastive negative: one or two corruptions of a faithful summary."""
    k = len(f["techniques"])
    kinds = ["vague", "drop", "order", "level", "repeat", "advice", "noip"]
    out = good_summary(f, rng)
    for kind in rng.sample(kinds, rng.choice([1, 2])):
        if kind == "vague":
            out = rng.choice(
                [
                    "suspicious activity was seen . please check the logs .",
                    "something happened on the network . it may be an attack .",
                ]
            )
        elif kind == "drop" and k > 1:
            out = re.sub(
                rf"\s*,?\s*(then |and |passing through )?<T{rng.randint(2, k)}>",
                "",
                out,
            )
        elif kind == "order" and k > 1:
            out = (
                out.replace("<T1>", "<TX>")
                .replace(f"<T{k}>", "<T1>")
                .replace("<TX>", f"<T{k}>")
            )
        elif kind == "level":
            wrong = rng.choice([lv for lv in LEVELS if lv != f["level"]])
            out = out.replace("<LEVEL>", wrong)  # a literal, wrong severity
        elif kind == "repeat":
            first = out.split(" . ")[0]
            out = f"{first} . {first} . {out}"
        elif kind == "advice":
            out = re.sub(
                r"recommended action : .*",
                f"recommended action : {rng.choice(list(REC.values()))} .",
                out,
            )
        elif kind == "noip":
            out = out.replace("<IP>", "the host")
    return out


def knowledge_sentences() -> list[str]:
    """General security language from the project's own knowledge bases."""
    from config.defense_kb import DEFENSE_KB
    from src.eval.final_report import ATTACKS
    from src.knowledge.attack_kb import explain, known_names

    out = []
    for name in known_names():
        e = explain(name)
        if e:
            out += [f"{name} : {e['what']}", f"how it works : {e['attacker']}"]
            out += [f"to defend against {name} : {d}" for d in e["defenses"]]
    for cls, kb in DEFENSE_KB.items():
        out.append(f"{cls} : {kb['summary']}")
    for tid, (name, tactic) in TECHNIQUES.items():
        out.append(f"{name} is a {tactic} technique in mitre att&ck ( {tid} ) .")
    for a in ATTACKS.values():
        out.append(f"{a['name']} : {a['what']} {a['measure']}")
    return out


def build_corpus(n_incidents: int, seed: int = 0) -> list[tuple[dict[str, Any], str]]:
    """(facts or {}, full training text) — good / bad / mixed pairs plus knowledge."""
    rng = random.Random(seed)
    data: list[tuple[dict[str, Any], str]] = []
    for _ in range(n_incidents):
        f = random_facts(rng)
        data.append((f, f"{prompt_text(f, '<good>')} {good_summary(f, rng)}"))
        data.append((f, f"{prompt_text(f, '<bad>')} {bad_summary(f, rng)}"))
        mixed = good_summary(f, rng) if rng.random() < 0.5 else bad_summary(f, rng)
        data.append((f, f"{prompt_text(f, '<sum>')} {mixed}"))
    kb = knowledge_sentences()
    for _ in range(max(1, n_incidents // 4)):
        data.append(({}, f"<kb> {rng.choice(kb)}"))
    rng.shuffle(data)
    return data


# --- filling slots + independent faithfulness metric ------------------------------


def fill(text: str, f: dict[str, Any]) -> str:
    sensors = f["sensors"]
    sens = (
        sensors[0]
        if len(sensors) == 1
        else ", ".join(sensors[:-1]) + " and " + sensors[-1]
    )
    from src.soc.correlate import is_unknown

    if is_unknown(f["ip"]):  # host-local incident: nothing to block
        text = text.replace("block <IP> and ", "").replace(
            "block <IP>", "contain the host"
        )
        text = text.replace("<IP>", "an unidentified source")
    out = (
        text.replace("<IP>", f["ip"])
        .replace("<N>", str(f["n"]))
        .replace("<SENSORS>", sens)
    )
    out = out.replace("<LEVEL>", f["level"])
    for i, t in enumerate(f["techniques"], 1):
        out = out.replace(f"<T{i}>", t)
    out = re.sub(r"<T\d>", "", out)  # defensive: never leave an unfilled slot
    out = re.sub(r"\s+([.,:])", r"\1", out)
    sentences = [s.strip() for s in out.split(". ") if s.strip()]
    out = ". ".join(s[0].upper() + s[1:] for s in sentences)
    # display-time capitalisation of acronyms (the model's delexicalised text stays
    # lower-case, so the faithfulness check is unaffected)
    return re.sub(r"\b(mfa|waf|siem|edr|vpn)\b", lambda m: m.group(1).upper(), out)


def faithfulness(text: str, f: dict[str, Any]) -> dict[str, float]:
    """Rule-based check, independent of the reward model (guards reward hacking).

    Every component is 1.0 only when that part of the narrative is factually right:
    coverage (IP + every technique), order (techniques in time order), level (no
    wrong literal severity), no_repeat, advice (the recommended action names the
    IP and ONLY mitigations for this incident's own tactics), corroboration (no
    "several sensors agree" claim for a single-sensor incident)."""
    k = len(f["techniques"])
    need = ["<IP>", *[f"<T{i}>" for i in range(1, k + 1)]]
    coverage = sum(s in text for s in need) / len(need)
    pos = [text.find(f"<T{i}>") for i in range(1, k + 1)]
    seen = [p for p in pos if p >= 0]
    order = float(seen == sorted(seen))
    literal = set(re.findall(r"\b(low|medium|high|critical)\b", text))
    level = float("<LEVEL>" in text and not literal - {f["level"]})
    sents = [s.strip() for s in text.split(" . ") if s.strip()]
    no_repeat = float(len(sents) == len(set(sents)))
    act = text.split("recommended action")[-1] if "recommended action" in text else ""
    said = {t for t, phrase in REC.items() if phrase in act}
    allowed = set(f["tactics"])
    advice = float(
        "<IP>" in act
        and said <= allowed
        and (bool(said) or not allowed or "investigate the source" in act)
    )
    claims = any(c in text for c in CORROBORATION)
    corroboration = float(f["corroborated"] or not claims)
    parts = {
        "coverage": coverage,
        "order": order,
        "level": level,
        "no_repeat": no_repeat,
        "advice": advice,
        "corroboration": corroboration,
    }
    return {**parts, "score": sum(parts.values()) / len(parts)}

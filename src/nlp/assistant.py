"""Grounded SOC assistant: question -> intent + entities -> facts from the SOC store
-> answer. ShieldLM writes incident narratives; every fact (IPs, techniques, counts)
comes from the data, and a narrative that fails the faithfulness check is replaced
by the reference template, so a weak model degrades wording, never correctness.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Any

from src.knowledge.attack_details import (
    DETAILS,
    TACTICS,
    display,
    techniques_for_tactic,
)
from src.nlp.corpus import (
    facts_from_incident,
    faithfulness,
    fill,
    good_summary,
    prompt_text,
)
from src.nlp.intent import (
    entities,
    keyword_intent,
    status_question,
    strong_intent,
    with_context,
)
from src.nlp.status import check_store as store_status
from src.nlp.status import describe, follow_ups
from src.soc.correlate import is_unknown
from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT, RESULTS_DIR

_log = get_logger(__name__)
NLP_DIR = PROJECT_ROOT / "artifacts" / "nlp"
NEEDS = {"ip": "ips", "mitre": "tids", "explain": "attack", "defend": "attack"}

HELP = (
    "💡 Ask me things like:\n"
    "- 📖 *what is a brute force attack?* (types, how it is done, signs, defences)\n"
    "- 🔎 *is there any defense evasion happening?* (checks all your logs)\n"
    "- 🚨 *what attacks are happening today?*\n- 🌐 *what did 203.0.113.45 do?*\n"
    "- 🛡️ *explain port scan* / *how do I stop brute force?*\n- 📊 *how many alerts in the last hour?*\n"
    "- 🎯 *who is attacking us?* / *which incidents used T1110?*\n- 🧪 *what are the project results?*"
)


_EXPLAIN_CUE = re.compile(
    r"^\s*(what\s+(is|are)|explain|describe|tell me about|how\s+(does|do|is))\b", re.I
)


def _title(name: str) -> str:
    """'**Brute force** (BruteForce)'; no repeat when the name is already shown."""
    d = display(name)
    head = f"**{d[:1].upper()}{d[1:]}**"
    return head if name in d else f"{head} ({name})"


def _display_facts(f: dict[str, Any]) -> dict[str, Any]:
    """Incident facts for a narrative shown as Markdown: the values that come from
    logs (source IP / entity, sensor names) are escaped (src/soc/safe.py). The
    facts themselves - used for feedback and training - stay unchanged."""
    from src.soc.safe import md

    ip = f.get("ip")
    return f | {
        "ip": ip if is_unknown(ip) else md(ip),
        "sensors": [md(s) for s in f.get("sensors", [])],
    }


def _dedupe_recs(text: str) -> str:
    """Drop a recommendation the narrative repeats ("... and reset the exposed
    credentials and enforce mfa and reset the exposed credentials and enforce mfa")."""
    from src.nlp.corpus import REC

    for rec in set(REC.values()):
        first = text.find(rec)
        if first < 0:
            continue
        head, tail = text[: first + len(rec)], text[first + len(rec) :]
        while (k := tail.find(" and " + rec)) >= 0:
            tail = tail[:k] + tail[k + len(" and " + rec) :]
        text = head + tail
    return text


class ShieldAssistant:
    def __init__(self, device: str = "cpu") -> None:
        import threading

        self.device = device
        self.clf = self.tok = self.lm = None
        # narratives by incident facts: written once (in the background when possible)
        # and reused, so an incident question is answered in milliseconds
        self._narratives: dict[str, dict[str, Any]] = {}
        self._gen_lock = threading.Lock()
        self._warming = False
        self.lm_name = "template"
        try:
            import joblib

            if (NLP_DIR / "intent.joblib").is_file():
                self.clf = joblib.load(NLP_DIR / "intent.joblib")
            from src.nlp.lm import Tokenizer, load

            if (NLP_DIR / "shieldlm_deployed.pt").is_file() and (
                NLP_DIR / "tokenizer.json"
            ).is_file():
                self.tok = Tokenizer.load(NLP_DIR / "tokenizer.json")
                self.lm = load(NLP_DIR / "shieldlm_deployed.pt", device)
                metrics = RESULTS_DIR / "nlp_metrics.json"
                m = json.loads(metrics.read_text("utf-8")) if metrics.is_file() else {}
                kind = m.get("rlcd", {}).get("deployed", "base")
                shrunk = m.get("compression", {}).get("deployed", "")
                self.lm_name = (
                    f"ShieldLM ({'RLCD-aligned' if kind == 'rlcd' else 'base'}"
                    + (
                        f", compressed: {shrunk}"
                        if shrunk and shrunk != "teacher"
                        else ""
                    )
                    + ")"
                )
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            _log.warning(
                "NLP artifacts unavailable (%s); using templates + keywords", exc
            )

    # --- understanding -------------------------------------------------------------
    def intent(self, q: str, e: dict[str, Any] | None = None) -> str:
        e = e or entities(q)
        # entity questions first (an IP / technique id is unambiguous), then
        # high-precision cues, then the classifier, then keywords
        if e["tids"] and not e["ips"] and status_question(q):
            return "status"  # "any credential dumping?" -> check the logs for T1003
        if e["ips"] or e["tids"]:
            return keyword_intent(q, e)
        strong = strong_intent(q)
        # "is there any defense evasion?" / "is the brute force still going on?"
        # -> check the logs and say ongoing / happened / not seen
        if strong != "count" and (e["attack"] or e["tactic"]) and status_question(q):
            return "status"
        if e["tactic"] and not e["attack"] and strong != "count":
            return "explain"  # "what is lateral movement?"
        if strong and not (e["attack"] and strong in ("help", "incidents")):
            return strong
        guess = None
        if self.clf is not None:
            proba = self.clf.predict_proba([q])[0]
            if proba.max() >= 0.35:
                guess = str(self.clf.classes_[proba.argmax()])
        guess = guess or keyword_intent(q, e)
        need = NEEDS.get(guess)
        if need and not e[need]:  # intent needs an entity the question lacks
            guess = keyword_intent(q, e)
        return guess

    # --- generation ----------------------------------------------------------------
    def summarize(self, inc: dict[str, Any]) -> tuple[str, str]:
        n = self.narrate(inc)
        return n["text"], n["by"]

    def narrate(self, inc: dict[str, Any]) -> dict[str, Any]:
        """Narrative + what produced it, its delexicalised form and the facts (the
        latter two are what analyst feedback stores for self-improvement). The same
        facts get the same narrative back from the cache (the language model samples,
        so a cached faithful narrative is as valid as a new one - and stays stable)."""
        f = facts_from_incident(inc)
        key = json.dumps(f, sort_keys=True, default=str)
        hit = self._narratives.get(key)
        if hit is not None:
            return dict(hit)
        with self._gen_lock:  # one generation at a time (chat + background warm-up)
            hit = self._narratives.get(key)
            if hit is None:
                hit = self._narrate(f, inc)
                if len(self._narratives) > 5000:
                    self._narratives.clear()
                self._narratives[key] = hit
        return dict(hit)

    def prewarm(self, limit: int = 15) -> int:
        """Write the narratives of the most severe incidents ahead of time (called in
        the background by the dashboard). Returns how many were new."""
        from src.soc import engine

        n = 0
        for inc in engine.incidents()[:limit]:
            key = json.dumps(facts_from_incident(inc), sort_keys=True, default=str)
            if key not in self._narratives:
                self.narrate(inc)
                n += 1
        return n

    def start_prewarm(self, every_s: float = 20.0) -> None:
        """Keep the top incidents' narratives ready (daemon thread, once per process)."""
        import threading
        import time

        if self._warming or self.lm is None:
            return
        self._warming = True

        def loop() -> None:
            while True:
                try:
                    self.prewarm()
                except Exception as exc:  # noqa: BLE001 - background helper, never fatal
                    _log.debug("narrative warm-up skipped: %s", exc)
                time.sleep(every_s)

        threading.Thread(target=loop, name="narrative-warmup", daemon=True).start()

    def _narrate(self, f: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
        if not f["techniques"]:
            text = f"Activity from {f['ip']} ({f['level']})."
            return {"text": text, "by": "template", "delex": "", "facts": f}
        if self.lm is not None and self.tok is not None:
            from src.nlp.lm import generate

            prompt = self.tok.encode(prompt_text(f, "<sum>"))
            # an analyst acts on this text: only a narrative with NO factual error
            # (every faithfulness component 1.0) is shown; up to 3 samples, then the
            # template, which is correct by construction
            for cont in generate(
                self.lm, self.tok, [prompt] * 3, self.device, temperature=0.7
            ):
                delex = self.tok.decode(cont)
                delex = _dedupe_recs(delex)  # a model sometimes repeats an action
                if faithfulness(delex, f)["score"] == 1.0:
                    return {"text": fill(delex, _display_facts(f)), "by": self.lm_name,
                            "delex": delex, "facts": f}  # fmt: skip
        delex = good_summary(f, random.Random(inc["id"]))
        return {
            "text": fill(delex, _display_facts(f)),
            "by": "template",
            "delex": delex,
            "facts": f,
        }

    # --- log checks -----------------------------------------------------------------
    def _status(
        self, e: dict[str, Any], lines: list[str], meta: dict[str, Any]
    ) -> list[str]:
        """'Is X happening?' from every ingested log; returns follow-up questions and
        records the evidence in ``meta`` (used for the steps)."""
        from src.soc.correlate import CLASS_TECHNIQUES, TECHNIQUES

        if e["tids"] and not e["attack"]:
            techs = list(e["tids"])
            subject = ", ".join(
                f"{TECHNIQUES[t][0]} ({t})" if t in TECHNIQUES else t for t in techs
            )
            family = None
        elif e["tactic"] and not e["attack"]:
            t = TACTICS[e["tactic"]]
            techs = techniques_for_tactic(e["tactic"])
            subject, family = f"{t['name']} ({t['id']})", None
        else:
            family = e["attack"]
            techs = CLASS_TECHNIQUES.get(family, [])
            subject = display(family)
        res = store_status(
            techs,
            family,
            since=e["since"],
            include_low=e["low_confidence"],
        )
        lines += describe(subject, res)
        if not techs and family is None:
            lines.append(
                "- 📡 None of the connected log sources can show this tactic directly; "
                "process or endpoint telemetry (for example Sysmon or an EDR) would be needed."
            )
        elif techs:
            lines.append("- 🎯 Looked for: " + ", ".join(techs) + ".")
        if family is not None:
            plain = display(family)
        elif e["tids"]:
            plain = TECHNIQUES.get(e["tids"][0], (e["tids"][0], ""))[0].lower()
        else:
            plain = TACTICS[e["tactic"]]["name"].lower()
        meta.update(status=res, subject=subject, plain=plain,
                    ips=res.get("top_ips", []), tids=list(res.get("techniques") or techs),
                    sources=sorted(res.get("by_source", {})))  # fmt: skip
        follow = follow_ups(plain, res, had_window=e["since"] is not None)
        if follow:
            lines.append(
                "🤔 **Nothing conclusive. Shall I look further?** You can ask:"
                if res["status"] == "not_seen"
                else "🔍 **The evidence is thin. Shall I dig further?** You can ask:"
            )
            lines += [f"- *{q}*" for q in follow]
        return follow

    def _explain_tactic(
        self, tactic: str, lines: list[str], meta: dict[str, Any]
    ) -> None:
        from src.soc.correlate import TECHNIQUES

        t = TACTICS[tactic]
        techs = techniques_for_tactic(tactic)
        lines.append(
            f"🧭 **{t['name']} ({t['id']})** is the MITRE ATT&CK tactic for {t['what']}."
        )
        meta.update(
            plain=t["name"].lower(), subject=f"{t['name']} ({t['id']})", tids=techs
        )
        if techs:
            lines.append("🧬 **Techniques of this tactic this SOC can see:**")
            lines += [f"- {x} {TECHNIQUES[x][0]}" for x in techs]
            res = store_status(techs)
            meta.update(status=res, ips=res.get("top_ips", []),
                        sources=sorted(res.get("by_source", {})))  # fmt: skip
            lines.append(
                "📡 **In your logs right now:** "
                + describe(t["name"], res)[0].replace("**", "")
            )
        else:
            lines.append(
                "📡 None of the connected log sources record this tactic directly; "
                "endpoint telemetry (Sysmon or an EDR) would be needed to detect it."
            )

    # --- answering -----------------------------------------------------------------
    @staticmethod
    def _off_topic(q: str, context: dict[str, Any] | None) -> bool:
        """A question with no security entity, no clear cue and mostly unknown words."""
        from src.nlp import typo
        from src.soc import reports, search

        if context or not typo.off_topic(q):
            return False
        e = entities(q)
        if e["ips"] or e["tids"] or e["attack"] or e["tactic"] or strong_intent(q):
            return False
        return not (reports.is_request(q) or search.is_search(q))

    def answer(
        self, q: str, context: dict[str, Any] | None = None, mode: str = "auto"
    ) -> dict[str, Any]:
        """Answer one question. ``context`` is the previous turn's subject (returned
        as ``result["context"]``), so "is it still going on?" or "how do I stop
        it?" refer to the attack / tactic / IP asked about before. ``mode`` is the
        reader's choice: ``auto`` (style follows the situation), ``beginner`` or
        ``expert``. Misspellings are corrected first (``result["corrected"]``)."""
        from src.nlp import style, typo

        raw = (q or "").strip()[:500]
        if raw and self._off_topic(raw, context):
            return {"intent": "off_topic", "context": dict(context or {}), "follow_up": [
                "what is a brute force attack?", "is anything happening now?", "who is attacking us?"],
                "text": "🙂 That looks outside what I cover. I answer questions about security: "
                        "attacks and ATT&CK techniques, what is happening in your logs, incidents, "
                        "IP addresses, how to stop an attack, reports and log searches.\n"
                        + HELP}  # fmt: skip
        q, changes = typo.correct(raw) if raw else (raw, [])
        e = with_context(q, entities(q), context) if q else None
        meta: dict[str, Any] = {}
        res = self._answer(q, e, meta)
        subject = {
            k: e[k] for k in ("attack", "tactic", "ips", "tids") if e and e.get(k)
        }
        res["context"] = subject or dict(context or {})
        it = res["intent"]
        lines = res["text"].split("\n")
        status = meta.get("status")
        urgent = style.is_urgent(it, status, meta.get("incs", []))
        st = style.detect(raw, it, mode if mode in style.MODES else "auto", urgent)
        state = status.get("status") if status else ("ongoing" if urgent else None)
        steps: list[str] = []
        plain = meta.get("plain") or "it"
        if state in ("ongoing", "happened", "possible") or (
            it in ("incidents", "ip", "mitre", "top") and meta.get("incs")
        ):
            steps = style.steps(meta.get("subject") or plain, meta.get("ips", []),
                                meta.get("tids", []), meta.get("sources", []),
                                state or "happened", plain)  # fmt: skip
        extra: list[str] = []
        if style.wants_expert(raw, mode):
            extra += style.expert_lines(meta.get("tids", []), meta.get("ips", []))
        if st == "learning" or mode == "beginner":
            extra += style.glossary_lines(res["text"])
        note = typo.did_you_mean(q, changes)
        if state == "ongoing" and st != "urgent":  # e.g. "what is X?" while X is active
            alert = "🚨 This is happening in your logs right now: see **What to do now** below."
            note = f"{note}\n{alert}" if note else alert
        text = style.compose(lines[0], lines[1:], st, steps, extra, note)
        res["text"] = "\n".join(style.tidy(ln) for ln in text.split("\n"))
        res["headline"] = res["text"].split("\n", 1)[0]
        res["style"] = st
        res["steps"] = steps
        if changes:
            res["corrected"] = q
        follow = res.get("follow_up", []) + style.suggestions(
            it, meta.get("plain"), meta.get("ips", []), meta.get("tids", []), state, q
        )
        res["follow_up"] = list(dict.fromkeys(follow))[:3]
        return res

    def _answer(
        self, q: str, e: dict[str, Any] | None, meta: dict[str, Any]
    ) -> dict[str, Any]:
        from src.knowledge.attack_kb import explain
        from src.soc import engine
        from src.soc.correlate import CLASS_TECHNIQUES, LEVEL_ICON, TECHNIQUES, level_of

        if not q or e is None:
            return {"intent": "help", "text": HEAD_HELP + "\n" + _HELP_BODY}
        # reports and log searches are explicit requests: routed before the
        # question model so its validated accuracy is unchanged (owner 2026-10-15)
        from src.soc import reports, search

        if reports.is_request(q):
            rep = reports.generate_request(q)
            files = rep["files"]
            return {"intent": "report", "report_files": {k: str(v) for k, v in files.items()},
                    "text": f"Report ready ({rep['label']}): {rep['headline']}\n"
                            f"Saved as {', '.join(p.name for p in files.values())} (in "
                            "results/reports). Download them below; what goes in comes "
                            "from the report template on the Reports page.",
                    "follow_up": ["make a report of yesterday", "make a report of this week",
                                  "is anything happening now?"]}  # fmt: skip
        if search.is_search(q):
            res = search.run_store(q)
            return {"intent": "search", "text": search.answer_text(res), "search": res,
                    "follow_up": ["show only alerts from the last 24 hours",
                                  "make a report of today", "who is attacking us?"]}  # fmt: skip
        it = self.intent(q, e)
        # any of the 706 ATT&CK techniques: "what is kerberoasting?", "explain T1548"
        if (
            not e["attack"]
            and not e["tactic"]
            and not e["ips"]
            and (it in ("explain", "defend", "help") or _EXPLAIN_CUE.search(q))
            and (
                it not in ("results", "count", "top")
                # "what is ransomware?": a named technique outranks a weak "results" guess
                or (it == "results" and not strong_intent(q) and _EXPLAIN_CUE.search(q)
                    and _find_technique(q) is not None)
            )
        ):
            from src.knowledge.explain import technique, technique_markdown

            tid = e["tids"][0] if e["tids"] else _find_technique(q)
            info = technique(tid) if tid else None
            if tid and info:
                res = store_status([tid.split(".")[0]])
                status = describe(tid, res)[0].replace("**", "")
                meta.update(status=res, plain=info["name"].lower(), subject=f"{info['name']} ({tid})",
                            tids=[tid], ips=res.get("top_ips", []),
                            sources=sorted(res.get("by_source", {})))  # fmt: skip
                tactic = (
                    ", ".join(x.replace("-", " ") for x in info["tactics"]) or "unknown"
                )
                head = (f"📖 **{info['name']} ({tid})** is a MITRE ATT&CK technique "
                        f"(tactic: {tactic}). {_first(info['what'])}")  # fmt: skip
                body = technique_markdown(tid).split("\n", 1)[1].lstrip("\n")
                return {
                    "intent": "explain_technique",
                    "entities": {"tids": [tid]},
                    "text": f"{head}\n{body}\n\n📡 **In your logs right now:** {status}",
                }
        since = e["since"] or 0
        incs = [i for i in engine.incidents() if i["end"] >= since]
        alerts = [
            a for a in engine.read_jsonl(engine.ALERTS) if a.get("ts", 0) >= since
        ]
        lines: list[str] = []

        if it in ("incidents", "ip", "mitre"):
            if it == "ip":
                incs = [i for i in incs if i["src_ip"] in e["ips"]]
                n_alerts = sum(a.get("src_ip") in e["ips"] for a in alerts)
                lines.append(
                    f"🌐 **{', '.join(e['ips'])}** is involved in {len(incs)} incident(s) "
                    f"and {n_alerts} alert(s)."
                    if incs or n_alerts
                    else f"✅ I found no incidents or alerts for **{', '.join(e['ips'])}**"
                    + (" in that time window." if e["since"] else ".")
                )
            if it == "mitre":
                incs = [
                    i
                    for i in incs
                    if {t["id"] for t in i["techniques"]} & set(e["tids"])
                ]
                lines.append(f"🧬 {len(incs)} incident(s) used {', '.join(e['tids'])}.")
                asked = e["tids"][0]
                name = TECHNIQUES.get(asked.split(".")[0], (asked, ""))[0]
                meta.update(
                    tids=list(e["tids"]),
                    plain=name.lower(),
                    subject=f"{name} ({asked})",
                )
            _incident_meta(incs, meta)
            if it == "mitre":  # the asked technique leads the steps and expert detail
                meta["tids"] = list(dict.fromkeys(e["tids"] + meta["tids"]))
            if not incs:
                if it == "incidents":
                    lines.append(
                        "✅ No matching incidents"
                        + (" in that time window." if e["since"] else ".")
                    )
            else:
                if it == "incidents":
                    lv = Counter(i["level"] for i in incs)
                    lines.append(
                        f"🧩 **{len(incs)} incident(s)**: "
                        + ", ".join(
                            f"{LEVEL_ICON[k]} {lv[k]} {k}"
                            for k in ("critical", "high", "medium", "low")
                            if lv[k]
                        )
                        + "."
                    )
                for inc in incs[:3]:
                    text, by = self.summarize(inc)
                    lines.append(f"- {text}  \n  _({by})_")
        elif it == "status":
            follow = self._status(e, lines, meta)
            if follow:
                return {
                    "intent": it,
                    "entities": {k: v for k, v in e.items() if v},
                    "text": "\n".join(lines),
                    "follow_up": follow,
                }
        elif it == "explain" and e["tactic"] and not e["attack"]:
            self._explain_tactic(e["tactic"], lines, meta)
        elif it in ("explain", "defend"):
            exp = explain(e["attack"]) if e["attack"] else None
            if exp is None:
                lines.append(
                    "❓ Which attack do you mean? For example: *explain port scan*."
                )
            else:
                info = DETAILS.get(exp["name"], {})
                tids = CLASS_TECHNIQUES.get(exp["name"], [])
                meta.update(
                    plain=display(exp["name"]), subject=display(exp["name"]), tids=tids
                )
                if it == "explain":
                    lines.append("📖 " + _title(exp["name"]) + f": {exp['what']}")
                    if info.get("types"):
                        lines.append("🗂️ **Types:**")
                        lines += [f"- **{n}**: {desc}." for n, desc in info["types"]]
                    if info.get("steps"):
                        lines.append("🪜 **How it is carried out:**")
                        lines += [f"{i}. {st}" for i, st in enumerate(info["steps"], 1)]
                    else:
                        lines.append(f"🛠️ **How attackers do it:** {exp['attacker']}")
                    if info.get("signs"):
                        lines.append("🔍 **Signs in your logs:**")
                        lines += [f"- {x}" for x in info["signs"]]
                else:
                    first = exp["defenses"][0] if exp["defenses"] else "see below"
                    lines.append(
                        f"🛡️ To stop **{display(exp['name'])}**, start here: {_lc(first)}"
                    )
                lines.append("🛡️ **How to defend:**")
                lines += [f"- {x}" for x in exp["defenses"]]
                if tids:
                    lines.append(
                        "🧬 **MITRE ATT&CK:** "
                        + ", ".join(f"{t} {TECHNIQUES[t][0]}" for t in tids)
                    )
                if it == "explain" and exp["kind"] == "attack_class":
                    res = store_status(tids, exp["name"])
                    meta.update(status=res, ips=res.get("top_ips", []),
                                sources=sorted(res.get("by_source", {})))  # fmt: skip
                    lines.append(
                        "📡 **In your logs right now:** "
                        + describe(display(exp["name"]), res)[0].replace("**", "")
                    )
        elif it == "count":
            counts = Counter(a.get("label") or a.get("kind") for a in alerts)
            lines.append(
                f"📊 There are **{len(alerts)} alert(s)** and **{len(incs)} incident(s)**"
                + (" in that window." if e["since"] else " in total.")
            )
            lines += [f"- {k}: {v}" for k, v in counts.most_common(5)]
        elif it == "top":
            sev: dict[str, int] = {}
            for i in incs:
                sev[i["src_ip"]] = max(sev.get(i["src_ip"], 0), i["severity"])
            n = Counter(a.get("src_ip") for a in alerts if a.get("src_ip"))
            top = sorted(sev, key=lambda ip: (-sev[ip], -n[ip]))[:5]

            def name(ip: str | None) -> str:
                if not is_unknown(ip):
                    return str(ip)
                return "an unknown source (no IP in the logs)"

            if top:
                lead = top[0]
                lines.append(
                    f"🎯 The top attacking source is **{name(lead)}** "
                    f"(`{level_of(sev[lead]).upper()}`, severity {sev[lead]}/10, {n[lead]} alert(s))."
                )
                lines.append("**All sources, most severe first:**")
                _incident_meta([i for i in incs if i["src_ip"] in top], meta)
                meta["ips"] = [ip for ip in top if not is_unknown(ip)]
            else:
                lines.append("✅ No attacking sources yet.")
            lines += [
                f"- {LEVEL_ICON[level_of(sev[ip])]} {name(ip)}: "
                f"`{level_of(sev[ip]).upper()}` severity {sev[ip]}/10, {n[ip]} alert(s)"
                for ip in top
            ]
        elif it == "results":
            lines.append("🧪 Here are the project's headline results.")
            for fname, label in (
                ("final_verdicts.json", "Hypotheses"),
                ("nlp_metrics.json", "NLP"),
            ):
                p = RESULTS_DIR / fname
                if p.is_file():
                    d = json.loads(p.read_text("utf-8"))
                    if label == "Hypotheses":
                        lines.append(
                            "- **Hypotheses:** "
                            + "; ".join(
                                f"{'❌' if v.startswith('not') else '✅'} {k}: `{v.upper()}`"
                                for k, v in d.items()
                            )
                            + "."
                        )
                    else:
                        lines.append(
                            f"- 🤖 **Assistant:** ShieldLM perplexity {d['shieldlm']['heldout_perplexity']:.2f}; "
                            f"question understanding {d['intent']['intent_accuracy_heldout_templates']:.1%}; "
                            f"RLCD win rate {d['rlcd'].get('win_rate_after_vs_before', 0):.2f}."
                        )
            if len(lines) == 1:
                lines[0] = (
                    "⏳ There are no results yet. The training run writes them when it finishes."
                )
        else:
            lines += [HEAD_HELP, _HELP_BODY]
        return {
            "intent": it,
            "entities": {k: v for k, v in e.items() if v},
            "text": "\n".join(lines),
        }


HEAD_HELP = "💡 I answer questions about attacks, your logs and the project's results."
_HELP_BODY = HELP.replace("💡 Ask me", "Ask me", 1)


def _find_technique(q: str) -> str | None:
    """explain.find(), but a short technique name must appear as a whole word
    (find() also matches substrings: "At" (T1053.002) inside "what")."""
    from src.knowledge.explain import find, kb

    tid = find(q)
    if tid is None:
        return None
    name = kb().get(tid, {}).get("name", "").lower()
    low = q.lower()
    short_substring = (
        name and len(name) <= 4 and name in low
        and not re.search(rf"\b{re.escape(name)}\b", low)
    )  # fmt: skip
    return None if short_substring else tid


def _first(text: str) -> str:
    """First sentence of a description."""
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def _lc(text: str) -> str:
    """'Use MFA.' -> 'use MFA.' (sentence continues after a colon)."""
    t = text.strip()
    t = t if t.endswith((".", "!", "?")) else t + "."
    return t[:1].lower() + t[1:] if t[:2] != t[:2].upper() else t


def _incident_meta(incs: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    """IPs, techniques and sources of the most severe incidents (for the steps)."""
    top = sorted(incs, key=lambda i: -i["severity"])[:3]
    meta["incs"] = incs
    meta.setdefault("ips", [i["src_ip"] for i in top if i.get("src_ip")])
    meta.setdefault(
        "tids", list(dict.fromkeys(t["id"] for i in top for t in i["techniques"]))
    )
    meta.setdefault("sources", sorted({s for i in top for s in i.get("sources", [])}))
    if top and top[0]["techniques"]:
        meta.setdefault("plain", top[0]["techniques"][0]["name"].lower())
        meta.setdefault("subject", top[0]["techniques"][0]["name"])

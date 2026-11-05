"""OWASP CRS: shared transform cache vs the old per-rule decoding (2026-10-25).

20,000 real web requests (AIT-LDS v1.1 access logs): every request is checked both ways
(old = each rule decodes the values itself), the results must be identical, and the time
of both is reported.

    python -m scripts.bench_crs_cache
"""

from __future__ import annotations

import json
import time
from itertools import islice

from utils.paths import RAW_DIR, RESULTS_DIR


def _old_matches(rule, req) -> bool:  # noqa: ANN001 - the pre-2026-10-25 behaviour
    from src.soc import crs

    for col, key in rule.variables:
        for value in crs._values(req, col, key):
            v = value
            for t in rule.transforms:
                v = crs.TRANSFORMS[t](v)
            if crs._op(rule.op, rule.arg, v) != rule.negate:
                return True
    return False


def main() -> None:
    from src.soc import crs, logformats

    reqs = []
    for f in sorted((RAW_DIR / "ait-lds-v1.1" / "data").rglob("*access.log*")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in islice(fh, 0, 200_000, 40):
                m = logformats._ACCESS.match(line)
                if m:
                    reqs.append((m["method"], m["target"], m["proto"] or "", m["ua"] or "", m["ref"] or ""))
        if len(reqs) >= 20_000:
            break
    reqs = reqs[:20_000]
    rules = crs._rules(str(crs.RULES_DIR), crs.PARANOIA)
    t = time.perf_counter()
    old = [[r.id for r in rules if _old_matches(r, crs.request(*q))] for q in reqs]
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    new = []
    for q in reqs:
        req = crs.request(*q)
        new.append([r.id for r in rules if r.matches(req)])
    t_new = time.perf_counter() - t
    res = {"requests": len(reqs), "identical": old == new, "old_s": round(t_old, 2), "new_s": round(t_new, 2),
           "speed_up": round(t_old / max(t_new, 1e-9), 2), "when": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
    (RESULTS_DIR / "extra" / "bench_crs_cache.json").write_text(json.dumps(res, indent=1), "utf-8")
    print(res)


if __name__ == "__main__":
    main()

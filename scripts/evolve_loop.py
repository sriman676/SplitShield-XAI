"""Run the self-evolving detection loop (src/soc/evolve.py).

    python -m scripts.evolve_loop                 # one iteration (autopilot does this)
    python -m scripts.evolve_loop --iterations 5
    python -m scripts.evolve_loop --build-benchmark   # once, while the datasets exist
"""

from __future__ import annotations

import argparse
import json

from src.soc import evolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--build-benchmark", action="store_true")
    a = ap.parse_args()
    if a.build_benchmark or not evolve.BENCH.is_file():
        print("benchmark rows:", evolve.build_benchmark())
    for _ in range(a.iterations):
        d = evolve.iterate()
        c = d.get("champion", {})
        print(
            f"iteration {d.get('iteration')}: kept={d.get('kept')} | steps caught "
            f"{c.get('steps_caught', 0):.3f} | attack lines {c.get('attack_lines_caught', 0):.3f} "
            f"| false alarms {c.get('false_alarm_rate', 0):.4f} | {c.get('events_per_second', 0):,.0f} events/s"
        )
        for t in d.get("tried", []):
            print(f"    tried: {t['challenger'][:90]} -> {'KEPT' if t['kept'] else 'rejected'} ({t['why']})")
        if "status" in d:
            print(d["status"])
    print("state:", json.dumps(evolve.state())[:400])


if __name__ == "__main__":
    main()

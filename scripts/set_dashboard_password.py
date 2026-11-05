"""Reset a dashboard user's password from a terminal on this PC (recovery when the
admin password is forgotten; owner 2026-10-16). Typed hidden; only a salted hash is
stored in users.json, and the reset is written to the audit log as "local_recovery".
Day-to-day password changes are made by the admin on the Users & roles page.

    .venv\\Scripts\\python.exe -m scripts.set_dashboard_password [username]
"""

from __future__ import annotations

import getpass
import sys

from src.soc import audit, users
from src.soc.auth import MIN_LENGTH, make_hash


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    d = users.load()
    if not d["users"]:
        print("No users yet - open the dashboard to create the admin account.")
        return 1
    admins = [n for n, u in d["users"].items() if u["role"] == "admin"]
    name = argv[0] if argv else (admins[0] if admins else "")
    if name not in d["users"]:
        print(f"No user {name!r}. Users: {', '.join(d['users'])}")
        return 1
    pw = getpass.getpass(f"New password for {name} (min {MIN_LENGTH} characters): ")
    if len(pw) < MIN_LENGTH:
        print(f"Too short - use at least {MIN_LENGTH} characters.")
        return 1
    if getpass.getpass("Repeat it: ") != pw:
        print("The two entries differ - nothing changed.")
        return 1
    d["users"][name]["hash"] = make_hash(pw)
    d["users"][name]["disabled"] = False
    users._write(d)
    audit.record(f"local:{getpass.getuser()}", "local_recovery", {"user": name})
    print(f"Saved. {name} can sign in with the new password.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

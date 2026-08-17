"""Compact reader for a face_qc_report.json.

Prints one line per gate FAMILY (the part of the name before '['), with
worst value, threshold, pass/fail counts — so a 160-gate report can be
read at a glance while iterating on the failures.

Usage:  python3 -m tools.dev_qc_summary output/qc_base/face_qc_report.json [--all]
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict


def summarise(path: str, show_all: bool = False) -> None:
    d = json.load(open(path))
    print(f"report: {path}   overall={'PASS' if d.get('passed') else 'FAIL'}")
    for char, cd in (d.get("characters") or {}).items():
        print(f"\n── {char}  {'PASS' if cd.get('passed') else 'FAIL'}"
              f"   (skipped: {cd.get('skipped') or '-'})")
        fam: "OrderedDict[str, dict]" = OrderedDict()
        for g in cd.get("gates") or []:
            key = g["name"].split("[")[0]
            f = fam.setdefault(key, {"n": 0, "fail": 0, "worst": None,
                                     "thr": g.get("threshold"),
                                     "detail": "", "where": ""})
            f["n"] += 1
            v = g.get("value")
            if not g.get("passed"):
                f["fail"] += 1
            if v is not None and (f["worst"] is None
                                  or abs(v) > abs(f["worst"])):
                f["worst"] = v
                f["thr"] = g.get("threshold")
                f["detail"] = (g.get("detail") or "")[:96]
                f["where"] = g["name"]
        for key, f in fam.items():
            if f["fail"] == 0 and not show_all:
                continue
            mark = "FAIL" if f["fail"] else "ok  "
            w = f["worst"]
            thr = f["thr"]
            print(f"  {mark} {key:34s} worst={w if w is None else round(w, 3):>10}"
                  f"  thr={thr if thr is None else round(thr, 3):>9}"
                  f"  {f['fail']}/{f['n']}")
            if f["detail"]:
                print(f"        {f['where']}: {f['detail']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    summarise(args[0] if args else "output/face_qc/face_qc_report.json",
              "--all" in sys.argv)

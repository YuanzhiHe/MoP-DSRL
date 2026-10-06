#!/usr/bin/env python3
"""Verify every reference against Crossref and arXiv before it enters the draft.

AI-written bibliographies carry a high rate of fabricated entries, so nothing
here is written from memory: each query hits a live index, and anything that does
not resolve is reported as unverified rather than guessed at.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

QUERIES = [
    ("BMD", "behavioral mode discovery fine-tuning multimodal generative policies"),
    ("DSRL", "diffusion steering reinforcement learning noise space policy"),
    ("D3IL", "towards diverse behaviors benchmark imitation learning D3IL"),
    ("BESO", "goal-conditioned imitation learning score-based diffusion policies"),
    ("DiffusionPolicy", "diffusion policy visuomotor policy learning action diffusion"),
    ("RectifiedFlow", "rectified flow transport generative modeling straight paths"),
    ("FlowMatching", "flow matching for generative modeling"),
    ("DIAYN", "diversity is all you need learning skills without a reward function"),
    ("LIBERO", "LIBERO benchmarking knowledge transfer lifelong robot learning"),
    ("pi0", "pi-0 vision language action flow model general robot control"),
    ("SAC", "soft actor-critic off-policy maximum entropy deep reinforcement learning"),
]


def crossref(q: str, rows: int = 3):
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(q) + f"&rows={rows}&select=title,author,issued,DOI,container-title,type")
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return json.loads(r.read().decode())["message"]["items"]
    except Exception as e:
        return [{"_error": str(e)}]


def arxiv(q: str, rows: int = 3):
    url = ("http://export.arxiv.org/api/query?search_query=all:"
           + urllib.parse.quote(q) + f"&max_results={rows}")
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            txt = r.read().decode()
    except Exception as e:
        return [{"_error": str(e)}]
    out = []
    for chunk in txt.split("<entry>")[1:]:
        def pick(tag):
            a = chunk.find(f"<{tag}>")
            b = chunk.find(f"</{tag}>")
            return chunk[a + len(tag) + 2:b].strip().replace("\n", " ") if a >= 0 else ""
        out.append({"title": " ".join(pick("title").split()),
                    "id": pick("id"), "published": pick("published")[:10]})
    return out


def main() -> int:
    for key, q in QUERIES:
        print(f"\n################ {key} ################")
        print(f"query: {q}")
        print("--- Crossref ---")
        for it in crossref(q):
            if "_error" in it:
                print(f"  ERROR {it['_error']}")
                continue
            t = (it.get("title") or [""])[0]
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            ct = (it.get("container-title") or [""])[:1]
            au = it.get("author", [])[:2]
            names = ", ".join(f"{a.get('family','')}" for a in au)
            print(f"  [{yr}] {t[:95]}")
            print(f"        {names} | {ct} | DOI {it.get('DOI')}")
        print("--- arXiv ---")
        for it in arxiv(q):
            if "_error" in it:
                print(f"  ERROR {it['_error']}")
                continue
            print(f"  [{it['published']}] {it['title'][:95]}")
            print(f"        {it['id']}")
        time.sleep(1.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

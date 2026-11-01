#!/usr/bin/env python3
"""Fetch BibTeX for a list of papers from arXiv and Crossref.

Nothing here is written from memory. Each entry is looked up by title, the
returned authors and year are printed for checking, and a paper that cannot be
found is reported rather than invented.
"""
import json, re, sys, time, urllib.parse, urllib.request

UA = {"User-Agent": "research/1.0 (mailto:heyuanzhi0728@163.com)"}

WANTED = [
    ("ho2016gail", "Generative Adversarial Imitation Learning"),
    ("fujimoto2021td3bc", "A Minimalist Approach to Offline Reinforcement Learning"),
    ("kumar2020cql", "Conservative Q-Learning for Offline Reinforcement Learning"),
    ("ziegler2019finetune", "Fine-Tuning Language Models from Human Preferences"),
    ("stiennon2020summarize", "Learning to summarize from human feedback"),
    ("williams1992reinforce", "Simple statistical gradient-following algorithms for connectionist reinforcement learning"),
    ("sharma2020dads", "Dynamics-Aware Unsupervised Discovery of Skills"),
    ("gregor2016vic", "Variational Intrinsic Control"),
    ("florence2022ibc", "Implicit Behavioral Cloning"),
    ("shafiullah2022bet", "Behavior Transformers: Cloning k modes with one stone"),
    ("janner2022diffuser", "Planning with Diffusion for Flexible Behavior Synthesis"),
    ("zhao2023act", "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"),
    ("ren2024dppo", "Diffusion Policy Policy Optimization"),
    ("song2021ddim", "Denoising Diffusion Implicit Models"),
    ("ho2020ddpm", "Denoising Diffusion Probabilistic Models"),
]


def arxiv(title):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                     timeout=45).read().decode()
    except Exception:
        return None
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(raw)
    except Exception:
        return None
    for e in root.findall("a:entry", ns):
        t = " ".join(e.find("a:title", ns).text.split())
        if t.lower()[:45] != title.lower()[:45]:
            continue
        aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
        au = " and ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns))
        yr = e.find("a:published", ns).text[:4]
        return dict(kind="arxiv", title=t, author=au, year=yr, eprint=aid)
    return None


def crossref(title):
    url = ("https://api.crossref.org/works?rows=3&query.bibliographic="
           + urllib.parse.quote(title))
    try:
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=45).read())
    except Exception:
        return None
    for it in d["message"]["items"]:
        t = (it.get("title") or [""])[0]
        if t.lower()[:40] != title.lower()[:40]:
            continue
        au = " and ".join(f"{a.get('family','')}, {a.get('given','')}"
                          for a in it.get("author", []))
        return dict(kind="doi", title=t, author=au,
                    year=str(it.get("issued", {}).get("date-parts", [[0]])[0][0]),
                    doi=it.get("DOI"),
                    venue=(it.get("container-title") or [""])[0],
                    pages=it.get("page", ""), volume=it.get("volume", ""))
    return None


import os
os.makedirs("runs/_refs", exist_ok=True)
out, missing = [], []
for key, title in WANTED:
    r = arxiv(title) or crossref(title)
    time.sleep(1.5)
    if not r:
        missing.append((key, title))
        print(f"  NOT FOUND  {title[:60]}", flush=True)
        continue
    print(f"  {r['kind']:6s} [{r['year']}] {r['title'][:58]}", flush=True)
    print(f"           {r['author'][:80]}", flush=True)
    if r["kind"] == "arxiv":
        out.append(f"""@article{{{key},
  title   = {{{r['title']}}},
  author  = {{{r['author']}}},
  journal = {{arXiv preprint arXiv:{r['eprint']}}},
  year    = {{{r['year']}}},
  eprint  = {{{r['eprint']}}},
  archivePrefix = {{arXiv}}
}}""")
    else:
        extra = f",\n  volume  = {{{r['volume']}}}" if r.get("volume") else ""
        extra += f",\n  pages   = {{{r['pages']}}}" if r.get("pages") else ""
        out.append(f"""@article{{{key},
  title   = {{{r['title']}}},
  author  = {{{r['author']}}},
  journal = {{{r['venue']}}},
  year    = {{{r['year']}}}{extra},
  doi     = {{{r['doi']}}}
}}""")

print(f"\n取得 {len(out)} 条，未找到 {len(missing)} 条")
open("runs/_refs/new_refs.bib", "w").write("\n\n".join(out) + "\n")
for k, t in missing:
    print(f"  缺: {k} — {t}")

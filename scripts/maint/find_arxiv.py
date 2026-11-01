#!/usr/bin/env python3
"""Look up arXiv entries by title and print what is there, nothing invented."""
import sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
UA = {"User-Agent": "research/1.0 (mailto:heyuanzhi0728@163.com)"}

TITLES = [
    "Enhancing Deep Reinforcement Learning for Robotic Motion Planning via Task Decomposition",
    "Detector Confidence Signals Presence Rather Than Occlusion in Cluttered Manipulation",
    "Sample-Efficient Low-Level Motion Planning for Robotic Manipulation Tasks via Zero-Shot Transfer Learning",
]

for t in TITLES:
    q = urllib.parse.quote(f'ti:"{t}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"
    print(f"\n=== {t[:62]} ===", flush=True)
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                     timeout=45).read().decode()
        root = ET.fromstring(raw)
    except Exception as exc:
        print(f"  查询失败: {type(exc).__name__}", flush=True)
        continue
    hits = root.findall("a:entry", NS)
    if not hits:
        print("  arXiv 无此条目", flush=True)
    for e in hits:
        got = " ".join(e.find("a:title", NS).text.split())
        au = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        print(f"  {e.find('a:id', NS).text.rsplit('/',1)[-1]}  [{e.find('a:published', NS).text[:10]}]")
        print(f"    {got[:72]}")
        print(f"    {', '.join(au)}")
    time.sleep(2)

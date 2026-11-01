"""Re-check every bibliography entry against the live arXiv API and Crossref."""
import json, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

BIB = "paper/references.bib"
UA = {"User-Agent": "bib-check/1.0 (mailto:HeY65@cardiff.ac.uk)"}
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def arxiv(eid):
    raw = get("http://export.arxiv.org/api/query?id_list=" + eid)
    e = ET.fromstring(raw).find(ATOM + "entry")
    if e is None or e.find(ATOM + "title") is None:
        return None
    jr = e.find(ARX + "journal_ref")
    doi = e.find(ARX + "doi")
    return {
        "title": " ".join(e.find(ATOM + "title").text.split()),
        "authors": [a.find(ATOM + "name").text for a in e.findall(ATOM + "author")],
        "year": e.find(ATOM + "published").text[:4],
        "journal_ref": jr.text.strip() if jr is not None and jr.text else None,
        "doi": doi.text.strip() if doi is not None and doi.text else None,
    }


def crossref(doi):
    try:
        m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    except Exception as exc:
        return {"error": str(exc)}
    au = ["%s %s" % (a.get("given", ""), a.get("family", "")) for a in m.get("author", [])]
    d = m.get("issued", {}).get("date-parts", [[None]])[0][0]
    return {
        "title": (m.get("title") or [""])[0],
        "authors": au,
        "year": d,
        "container": (m.get("container-title") or [""])[0],
        "volume": m.get("volume"), "issue": m.get("issue"),
        "page": m.get("page") or m.get("article-number"),
        "type": m.get("type"),
    }


entries = re.split(r"\n(?=@)", open(BIB).read())
out = []
for raw in entries:
    m = re.match(r"@(\w+)\{([^,]+),", raw.strip())
    if not m:
        continue
    key = m.group(2)
    rec = {"key": key}
    ep = re.search(r"eprint\s*=\s*\{([^}]+)\}", raw)
    do = re.search(r"doi\s*=\s*\{([^}]+)\}", raw)
    nau = len(re.findall(r" and ", re.search(r"author\s*=\s*\{(.*?)\}\s*,\s*\n", raw, re.S).group(1))) + 1
    rec["bib_authors"] = nau
    rec["bib_year"] = re.search(r"year\s*=\s*\{(\d+)\}", raw).group(1)
    if ep:
        rec["arxiv"] = arxiv(ep.group(1).split("v")[0])
        time.sleep(3)
    if do:
        rec["crossref"] = crossref(do.group(1))
        time.sleep(1)
    out.append(rec)
    print("done", key, file=sys.stderr)

json.dump(out, open("scripts/maint/_bibcheck.json", "w"), indent=1)

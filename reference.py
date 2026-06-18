"""Reference Repo Analyzer."""
import os, subprocess, json, re, shutil, urllib.request
from pathlib import Path
TMP = Path("/tmp/neural_refs")

def parse_github_url(url):
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    return (m.group(1), m.group(2)) if m else (None, None)

def clone_repo(url):
    owner, repo = parse_github_url(url)
    if not owner:
        return None, "Invalid URL"
    dest = TMP / f"{owner}_{repo}"
    if dest.exists():
        shutil.rmtree(dest)
    TMP.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git","clone","--depth","1",f"https://github.com/{owner}/{repo}.git",str(dest)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None, f"Clone failed: {r.stderr[:100]}"
    return dest, "OK"
def get_github_meta(owner, repo):
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers={"User-Agent": "Neural"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
            return {"stars": d.get("stargazers_count",0),
                    "forks": d.get("forks_count",0),
                    "license": (d.get("license") or {}).get("spdx_id","?"),
                    "desc": d.get("description",""),
                    "lang": d.get("language","")}
    except:
        return {"stars":0,"forks":0,"license":"?","desc":"","lang":""}

def analyze_structure(path):
    files = list(path.rglob("*"))
    exts = {".py":"python",".js":"js",".ts":"ts",".rs":"rust",".go":"go",".md":"docs"}
    counts = {}
    for f in files:
        if f.suffix in exts and f.is_file():
            lang = exts[f.suffix]
            counts[lang] = counts.get(lang,0)+1
    readme = path/"README.md"
    readme_txt = readme.read_text()[:2000] if readme.exists() else ""
    dirs = sorted(set(f.parent.name for f in files if f.is_file() and f.parent.name))[:10]
    return {"total":len(files),"counts":counts,"readme":readme_txt,"dirs":dirs}

def analyze(url):
    owner, repo = parse_github_url(url)
    if not owner:
        return {"error":"Invalid URL"}
    meta = get_github_meta(owner, repo)
    path, status = clone_repo(url)
    if not path:
        return {"error":status,"meta":meta}
    struct = analyze_structure(path)
    features = ["agent","tool","streaming","planner","rag","mcp",
                "sandbox","approval","edit","web","session","persona"]
    rl = struct["readme"].lower()
    shared = [f for f in features if f in rl]
    return {"owner":owner,"repo":repo,"meta":meta,"struct":struct,"shared":shared}

def report(data):
    if "error" in data:
        return f"[red]Error: {data['error']}[/red]"
    m = data["meta"]; s = data["struct"]
    c = s["counts"]
    counts = ", ".join(f"{k}:{v}" for k,v in sorted(c.items())) if c else "none"
    lines = [
        f"[bold cyan]Reference: {data['owner']}/{data['repo']}[/bold cyan]",
        f"[dim]{m['stars']} stars | {m['license']} | {m['lang']}[/dim]",
        f"{m['desc'][:120]}",
        "",
        f"[bold]Structure:[/bold] {s['total']} files | {counts}",
    ]
    if s["dirs"]:
        lines.append(f"Dirs: {', '.join(s['dirs'])}")
    lines.append("")
    lines.append(f"[bold]Shared: {len(data['shared'])} features[/bold]")
    for f in data["shared"]:
        lines.append(f"  [green]+[/green] {f}")
    try:
        from knowledge import add_fact
        add_fact(f"Ref:{data['owner']}/{data['repo']}",
                 f"{m['stars']}stars {m['lang']} shared:{len(data['shared'])} features",
                 source="reference")
    except:
        pass
    return "\n".join(lines)

"""Terminal context awareness + Agent personas for Neural."""
import os, subprocess
from pathlib import Path

PERSONAS = {
    "coder": {"name":"Coder","instruction":"You are a coding assistant. Focus on code quality, git, and project structure. Use file and git tools."},
    "sysadmin": {"name":"Sysadmin","instruction":"You are a system administrator. Focus on system health, processes, networking. Use shell commands."},
    "research": {"name":"Research","instruction":"You are a research assistant. Focus on finding info, reading, analysis. Use web_fetch and grep."},
    "default": {"name":"Default","instruction":"You are a general-purpose AI assistant with tool access."},
}

def get_terminal_context():
    import datetime
    parts = []
    parts.append("Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    parts.append(f"OS: {os.uname().sysname} {os.uname().release}")
    parts.append(f"Host: {os.uname().nodename}")
    parts.append(f"CWD: {os.getcwd()}")
    return "\n".join(parts)

def get_git_context(cwd=None):
    if cwd is None: cwd = os.getcwd()
    try:
        r = subprocess.run(["git","rev-parse","--abbrev-ref","HEAD"],capture_output=True,text=True,timeout=3,cwd=cwd)
        b = r.stdout.strip()
        if not b or "fatal" in b: return ""
        s = subprocess.run(["git","status","--short"],capture_output=True,text=True,timeout=3,cwd=cwd).stdout.strip()
        lines = [f"Git branch: {b}"]
        if s:
            lines.append("Git status:")
            for line in s.split("\n")[:5]: lines.append(f"  {line}")
        return "\n".join(lines)
    except: return ""

def build_context_block():
    parts = []
    t = get_terminal_context()
    if t: parts.append(t)
    g = get_git_context()
    if g: parts.append(g)
    import os as _os2
    for hist_file in ["~/.zsh_history", "~/.bash_history", "~/.history"]:
        hp = Path(_os2.path.expanduser(hist_file))
        if hp.exists():
            try:
                cmds = hp.read_text().strip().split("\n")[-3:]
                if cmds:
                    parts.append("Recent:\n" + "\n".join(f"  $ {c}" for c in cmds))
            except:
                pass
            break
    return "## Terminal Context\n" + "\n".join(parts) if parts else ""

def persona_instruction(name="default"):
    p = PERSONAS.get(name, PERSONAS["default"])
    return p["instruction"]

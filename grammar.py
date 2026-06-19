"""Grammar-guided JSON parser."""
import re, json

COMMON_FIXES = [
    (r',(\s*[}\]])', r'\1'),
    (r'\s*:', ':'),
    (r'\bTrue\b', 'true'),
    (r'\bFalse\b', 'false'),
    (r'\bNone\b', 'null'),
    (r':\s*:', ':'),
]

def extract_json(text):
    m = re.search(r'```(?:json)?\s*\n?(\{.*?\})\n?\s*```', text, re.DOTALL)
    if m: return m.group(1)
    start = text.find('{"tool"')
    if start < 0: start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0: return text[start:i+1]
    return ''

def auto_fix(text):
    for pat, repl in COMMON_FIXES:
        text = re.sub(pat, repl, text)
    return text

def parse_tool_call(text):
    js = extract_json(text)
    if not js: return None
    # Try normal parse
    try:
        d = json.loads(js)
        if 'tool' in d: return d
    except: pass
    # Try auto-fix
    try:
        d = json.loads(auto_fix(js))
        if 'tool' in d: return d
    except: pass
    # Try regex fallback for malformed JSON
    tm = re.search(r'"tool"\s*:\s*"([^"]+)"', js)
    if tm:
        args = {}
        # Extract args with regex for various formats
        am = re.search(r'"args"\s*:\s*\{(.*?)\}', js, re.DOTALL)
        if am:
            try:
                raw = "{" + am.group(1) + "}"
                args = json.loads(auto_fix(raw))
            except:
                # Extract simple key:value pairs
                for kv in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', am.group(1)):
                    args[kv[0]] = kv[1]
        return {'tool': tm.group(1), 'args': args}
    return None
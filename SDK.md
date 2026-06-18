# RSA Agentic Plugin SDK

Build your own tools. 3 steps.

## Step 1: Create a file in plugins/
~/rsa-agentic/plugins/my_tool.py

## Step 2: Define functions
def get_weather(city: str) -> str:
    import requests
    return requests.get(f"https://wttr.in/{city}?format=%t+%w").text

def search_web(query: str) -> str:
    import requests
    r = requests.get(f"https://lite.duckduckgo.com/lite/?q={query}")
    return r.text[:500]

## Step 3: Register
def register(reg):
    from plugin_loader import PluginTool
    reg(PluginTool("weather", get_weather, "Get weather", {"city": "city name"}))
    reg(PluginTool("web_search", search_web, "Search web", {"query": "query"}))

## Done! /plugins to verify
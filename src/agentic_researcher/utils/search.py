import urllib.parse
import httpx
import re
from html import unescape
from typing import List, Dict
from loguru import logger

async def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo Lite and parse the results using regex.
    Returns a list of dicts with keys: 'title', 'url', 'snippet'.
    """
    url = "https://lite.duckduckgo.com/lite/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    }
    data = {"q": query}
    logger.debug(f"🌎💻 Search query: {query}")
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=10.0) as client:
            resp = await client.post(url, data=data)
            if resp.status_code != 200:
                return []
            html_content = resp.text

            # Regex to find links: <a rel="nofollow" href="..." class='result-link'>...</a>
            link_pattern = re.compile(
                r"<a\s+rel=\"nofollow\"\s+href=\"(?P<url>[^\"]+)\"\s+class='result-link'>(?P<title>.*?)</a>",
                re.DOTALL
            )
            # Regex to find snippet: <td class='result-snippet'>...</td>
            snippet_pattern = re.compile(
                r"<td\s+class='result-snippet'[^>]*>(?P<snippet>.*?)</td>",
                re.DOTALL
            )

            links = list(link_pattern.finditer(html_content))
            snippets = list(snippet_pattern.finditer(html_content))

            results = []
            for l, s in zip(links, snippets):
                # Clean html tags from title and snippet
                title = re.sub(r"<[^>]+>", "", l.group("title")).strip()
                url_str = l.group("url")
                snippet = re.sub(r"<[^>]+>", "", s.group("snippet")).strip()

                results.append({
                    "title": unescape(title),
                    "url": unescape(url_str),
                    "snippet": unescape(snippet)
                })

                if len(results) >= max_results:
                    break
                
            logger.debug(f"🌎💻 Search results: {results}")
            return results
    except Exception as exc:
        # Gracefully return empty list on any network/parsing failure
        logger.error(f"🌎💻 Search failed: {exc}")
        return []

import re
import time
import json
import requests
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def sleep(sec=0.4):
    time.sleep(sec)


def get(url, timeout=30):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def get_html_js(url, timeout=45000):
    """Страница после выполнения JS (Playwright). Возвращает HTML или None."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [Playwright] не установлен. Выполните: pip install playwright && playwright install chromium")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"Accept-Language": "ru-RU,ru;q=0.9"})
            page.goto(url, wait_until="load", timeout=timeout)
            time.sleep(2)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"  [Playwright] ошибка: {e}")
        return None


def uniq(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def clean_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def strip_html(s):
    if not s:
        return ""
    return BeautifulSoup(s, "lxml").get_text(" ", strip=True)


def meta_content(soup, attr_name, attr_value, content_attr="content"):
    tag = soup.find("meta", attrs={attr_name: attr_value})
    if not tag:
        return None
    return (tag.get(content_attr) or "").strip() or None


def jsonld_article(soup):
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for obj in (data if isinstance(data, list) else [data]):
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") in ("NewsArticle", "Article", ["NewsArticle"], ["Article"]):
                return obj
            if obj.get("@type") == "WebPage":
                me = obj.get("mainEntity")
                if isinstance(me, dict) and me.get("@type") in ("NewsArticle", "Article"):
                    return me
    return None


def extract_article(url, source_name, selectors=None, use_js=False):
    html = None
    if use_js:
        html = get_html_js(url)
    if html is None:
        try:
            r = get(url)
            html = r.text
        except Exception as e:
            return {"url": url, "source_name": source_name, "title": None, "published_at": None, "text": None, "error": str(e)}

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    jsonld = jsonld_article(soup)

    title = (
        meta_content(soup, "property", "og:title")
        or meta_content(soup, "name", "twitter:title")
        or (jsonld.get("headline") if jsonld else None)
        or (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else None)
    )
    published_at = (
        meta_content(soup, "property", "article:published_time")
        or meta_content(soup, "name", "article:published_time")
        or (jsonld.get("datePublished") if jsonld else None)
    )
    body = (jsonld.get("articleBody") if jsonld else None) or ""
    if not body and selectors:
        for sel in selectors:
            nodes = soup.select(sel)
            parts = [n.get_text(" ", strip=True) for n in nodes if n.get_text(" ", strip=True)]
            if parts:
                body = " ".join(parts)
                break
    if not body and soup.find("article"):
        ps = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "h2", "li"]) if p.get_text(" ", strip=True)]
        if ps:
            body = " ".join(ps)
    if not body:
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len((p.get_text(" ", strip=True) or "")) > 40]
        if ps:
            body = " ".join(ps[:80])

    return {
        "url": url,
        "source_name": source_name,
        "title": clean_text(title) if title else None,
        "published_at": published_at,
        "text": clean_text(body) if body else None,
        "description": "",
        "author": "",
        "query": "",
        "language": "",
    }


def collect_links(url, allowed_domain, url_regex, limit=300, debug=False, use_js=False):
    html = None
    if use_js:
        html = get_html_js(url)
        if debug and html is not None:
            print(f"  [collect_links] Playwright {url}: len={len(html)}")
    if html is None and use_js:
        try:
            r = get(url)
            html = r.text
            if debug:
                print(f"  [collect_links] fallback GET {url}: status={r.status_code}, len={len(html)}")
        except Exception as e:
            if debug:
                print(f"  [collect_links] GET {url}: ошибка — {e}")
            return []
    elif html is None:
        try:
            r = get(url)
            html = r.text
            if debug:
                print(f"  [collect_links] GET {url}: status={r.status_code}, len={len(html)}")
        except Exception as e:
            if debug:
                print(f"  [collect_links] GET {url}: ошибка — {e}")
            return []
    soup = BeautifulSoup(html, "lxml")
    all_a = soup.select("a[href]")
    links = []
    same_domain = []
    for a in all_a:
        href = (a.get("href") or "").strip()
        if not href:
            continue
        href = urljoin(url, href).split("#")[0]
        if allowed_domain not in urlparse(href).netloc:
            continue
        same_domain.append(href)
        if re.search(url_regex, href):
            links.append(href)
    if debug:
        print(f"  [collect_links] ссылок <a>: {len(all_a)}, с доменом {allowed_domain}: {len(same_domain)}, по regex: {len(links)}")
        if same_domain and not links:
            for h in same_domain[:5]:
                print(f"    пример (не подошёл): {h[:90]}")
    return uniq(links)[:limit]

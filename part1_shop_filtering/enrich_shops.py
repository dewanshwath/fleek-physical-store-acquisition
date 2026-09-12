"""
Part 1 enrichment: find each genuine shop's Instagram presence and factor it
into the visit-priority score.

WHY SERPAPI, NOT DIRECT INSTAGRAM SCRAPING:
Instagram blocks unauthenticated scraping almost immediately (rate limits,
login walls) and has no public API for arbitrary profile lookup since 2020.
The reliable approach is searching Google for "{shop name} {city} instagram"
via a search API — Google's own snippet often contains the follower count
(e.g. "12.3K Followers, 450 Following...") without ever touching Instagram
directly. This is slower and costs API credits, but it actually works at
scale without getting IP-banned.

SETUP:
1. Sign up for a free SerpAPI account (https://serpapi.com) — free tier gives
   ~100 searches/month, enough for this dataset.
2. Set your key: export SERPAPI_KEY="your-key-here"
3. pip install google-search-results requests

COST/RATE-LIMIT NOTE:
This makes one search per shop. For 34 genuine shops that's 34 calls — fine
on a free tier. At 30,000 rows this would need caching (don't re-search a
shop you've already enriched) and a paid tier / batching — noted in the
scaling section of the README.
"""

import os
import re
import time
import pandas as pd
from serpapi import GoogleSearch

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

FOLLOWER_PATTERN = re.compile(
    r"([\d,.]+[KkMm]?)\s*Followers", re.IGNORECASE
)


def parse_follower_count(text: str) -> float:
    """Extract a follower count like '12.3K' or '1,204' from a search snippet
    and convert to a plain number. Returns 0 if not found."""
    if not text:
        return 0
    match = FOLLOWER_PATTERN.search(text)
    if not match:
        return 0
    raw = match.group(1).replace(",", "")
    if raw[-1].lower() == "k":
        return float(raw[:-1]) * 1_000
    if raw[-1].lower() == "m":
        return float(raw[:-1]) * 1_000_000
    try:
        return float(raw)
    except ValueError:
        return 0


def find_instagram_presence(shop_name: str, city: str = "Manchester") -> dict:
    """Search for the shop's Instagram profile and pull whatever signal we can
    get from the snippet — handle, follower count, or just existence."""
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set — see script docstring for setup.")

    query = f"{shop_name} {city} instagram"
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 3,
    })
    results = search.get_dict()

    organic = results.get("organic_results", [])
    for result in organic:
        link = result.get("link", "")
        if "instagram.com" in link:
            snippet = result.get("snippet", "")
            handle_match = re.search(r"instagram\.com/([^/?]+)", link)
            return {
                "instagram_handle": handle_match.group(1) if handle_match else None,
                "instagram_followers_est": parse_follower_count(snippet),
                "instagram_found": True,
            }

    return {"instagram_handle": None, "instagram_followers_est": 0, "instagram_found": False}


def enrich_shops(genuine_shops_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a dataframe of genuine shops (from filter_shops.py) with
    Instagram signal. Rate-limited with a short delay to be a polite API citizen."""
    enriched_rows = []
    for _, row in genuine_shops_df.iterrows():
        ig_data = find_instagram_presence(row["place_name"], row.get("city", "Manchester"))
        enriched_rows.append({**row.to_dict(), **ig_data})
        time.sleep(1)  # be polite to the API, avoid hammering rate limits

    return pd.DataFrame(enriched_rows)


def instagram_bonus(row) -> float:
    """Small additive bonus for the visit-priority score — an active, sizeable
    Instagram presence suggests a more serious, higher-appetite operation."""
    if not row.get("instagram_found"):
        return 0.0
    followers = row.get("instagram_followers_est", 0)
    if followers >= 10_000:
        return 1.5
    if followers >= 1_000:
        return 0.8
    return 0.3  # found a profile, but small/unclear following


if __name__ == "__main__":
    genuine = pd.read_csv("../output/manchester_visit_list.csv")
    enriched = enrich_shops(genuine)
    enriched["instagram_score_bonus"] = enriched.apply(instagram_bonus, axis=1)
    enriched["visit_priority_score"] = (
        enriched["visit_priority_score"] + enriched["instagram_score_bonus"]
    )
    enriched = enriched.sort_values("visit_priority_score", ascending=False)
    enriched.to_csv("../output/manchester_visit_list_enriched.csv", index=False)

    print(f"Enriched {len(enriched)} shops.")
    print(enriched[["place_name", "instagram_handle", "instagram_followers_est",
                     "visit_priority_score"]].head(10).to_string(index=False))

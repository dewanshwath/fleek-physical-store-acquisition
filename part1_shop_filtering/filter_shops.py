"""
Part 1: Filter the Manchester scrape down to genuine vintage clothing shops,
then rank them for a day of visits.

Approach:
- Google Maps category is noisy (a "Boutique" or "Clothing store" label could be
  anything; a "Vintage clothing store" label could be a name-trap). Don't trust it alone.
- The review text is the strongest signal in this data — it consistently says what
  a place actually sells, sometimes explicitly ruling clothing in or out.
- Combine: keyword scoring on review text + a lighter-weight boost/penalty from
  category + rating/review_count as a tie-breaker for prioritisation, not filtering.

This is intentionally a scoring function over a fixed classifier, so it generalises
to review text we haven't seen, rather than hardcoding the ~27 templates found in
this particular scrape.
"""

import pandas as pd
import re

# Strong positive signals: words that show up when a place genuinely sells
# vintage/secondhand CLOTHING specifically.
CLOTHING_POSITIVE_KEYWORDS = [
    "vintage denim", "denim", "carhartt", "dickies", "levi", "band tees", "tees",
    "streetwear", "sportswear", "football shirts", "reworked", "archive designer",
    "y2k", "grunge", "workwear", "military surplus", "curated vintage",
    "vintage clothing", "vintage womenswear", "vintage menswear", "vintage boutique",
]

# Explicit negative phrases: the review is telling us directly this ISN'T clothing.
EXPLICIT_NEGATIVE_PHRASES = [
    "no clothes", "not clothing", "no clothing",
]

# Category/keyword signals for things that are almost never a vintage clothing shop,
# regardless of "vintage" appearing in the name.
NON_CLOTHING_CATEGORY_SIGNALS = [
    "charity", "antique", "bric-a-brac", "homeware", "home goods", "furniture",
    "record", "vinyl", "book", "pawn", "wine", "cafe", "barber", "tattoo",
    "video game", "costume",
]


def score_review_text(review: str) -> float:
    """Returns a confidence score in [-1, 1] that this review describes a genuine
    vintage clothing shop, based on keyword presence."""
    if not isinstance(review, str) or not review.strip():
        return 0.0

    text = review.lower()

    for phrase in EXPLICIT_NEGATIVE_PHRASES:
        if phrase in text:
            return -1.0

    positive_hits = sum(1 for kw in CLOTHING_POSITIVE_KEYWORDS if kw in text)
    negative_hits = sum(1 for kw in NON_CLOTHING_CATEGORY_SIGNALS if kw in text)

    if positive_hits == 0 and negative_hits == 0:
        return 0.0  # genuinely ambiguous — e.g. "decent secondhand finds"

    score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
    return max(-1.0, min(1.0, score))


def score_category(category: str) -> float:
    """Lighter-weight signal from the category field alone."""
    if not isinstance(category, str):
        return 0.0
    cat = category.lower()

    if any(sig in cat for sig in NON_CLOTHING_CATEGORY_SIGNALS):
        return -0.5  # category alone shouldn't fully disqualify — review can override
    if "vintage clothing" in cat or "used clothing" in cat or "retro clothing" in cat:
        return 0.5
    if cat in ("boutique", "clothing store", "vintage store", "second hand shop"):
        return 0.0  # genuinely ambiguous categories — let the review decide
    return 0.0


def is_genuine_vintage_clothing_shop(row) -> bool:
    review_score = score_review_text(row["top_review"])
    category_score = score_category(row["maps_category"])
    combined = review_score + (category_score * 0.3)  # review dominates
    return combined > 0.15


def visit_priority_score(row, cluster_bonus: float = 0.0) -> float:
    """Ranks shops worth visiting: quality (rating), size/appetite (review_count),
    price positioning (mid-to-premium price levels suggest a serious operation
    rather than a bargain-bin browse), geographic clustering (a shop with
    several other genuine shops nearby is worth more per visit than an isolated
    one), and contactability (website/phone) — the Readme explicitly frames
    website + phone + price_level together as a "contactability" signal, so a
    shop you can actually reach ahead of a visit (to confirm opening hours,
    check stock) is worth more than one you'd be showing up to cold."""
    rating = row["rating"] if pd.notna(row["rating"]) else 3.0
    review_count = row["review_count"] if pd.notna(row["review_count"]) else 0
    price_level = row["price_level"] if isinstance(row["price_level"], str) else ""
    has_website = isinstance(row.get("website"), str) and row.get("website").strip() != ""
    has_phone = isinstance(row.get("phone"), str) and row.get("phone").strip() != ""

    # review_count as a rough proxy for footfall/size — log-scaled so one huge
    # outlier doesn't dominate the ranking
    import math
    size_score = math.log10(max(review_count, 1))

    price_bonus = {"£": 0, "££": 0.3, "£££": 0.5}.get(price_level, 0.15)
    contactability_bonus = (0.3 if has_website else 0) + (0.2 if has_phone else 0)

    return (rating * 1.0) + (size_score * 1.5) + price_bonus + cluster_bonus + contactability_bonus


def compute_cluster_bonus(genuine_df: pd.DataFrame, radius_km: float = 0.5) -> pd.Series:
    """For each genuine shop, count how many OTHER genuine shops sit within
    radius_km (default 500m — a comfortable walk), and convert that into a
    capped bonus so it acts as a tie-breaker, not the dominant signal.

    SCALING NOTE: a naive pairwise haversine loop is O(n^2) — at 30,000 rows
    that's ~450 million comparisons, on the order of an hour in pure Python.
    That directly fails the brief's "wouldn't fall over at 30,000 rows"
    requirement, so this uses scipy's cKDTree instead: project lat/lng to a
    flat local approximation (fine at city scale, where Earth's curvature is
    negligible), build a spatial index once, then query each point's
    neighbours in O(log n). This is O(n log n) overall and handles 30,000
    rows in well under a second."""
    from scipy.spatial import cKDTree
    import numpy as np

    coords = genuine_df[["lat", "lng"]].copy()
    valid = coords["lat"].notna() & coords["lng"].notna()

    bonuses = pd.Series(0.0, index=genuine_df.index)
    if valid.sum() < 2:
        return bonuses  # nothing to cluster against

    # Flat local projection: at a single city's scale, 1 degree latitude is
    # ~111km everywhere, and 1 degree longitude is ~111km * cos(latitude).
    # This avoids the O(n^2) trig-heavy haversine loop while staying accurate
    # enough for a same-city radius check.
    mean_lat_rad = np.radians(coords.loc[valid, "lat"].mean())
    km_per_deg_lat = 111.0
    km_per_deg_lng = 111.0 * np.cos(mean_lat_rad)

    xy = np.column_stack([
        coords.loc[valid, "lat"] * km_per_deg_lat,
        coords.loc[valid, "lng"] * km_per_deg_lng,
    ])

    tree = cKDTree(xy)
    # query_ball_point returns each point's own index too, so subtract 1
    neighbour_counts = [len(tree.query_ball_point(point, r=radius_km)) - 1 for point in xy]

    valid_bonuses = pd.Series(
        [min(count * 0.3, 1.5) for count in neighbour_counts],
        index=coords.loc[valid].index,
    )
    bonuses.loc[valid] = valid_bonuses
    return bonuses


def main():
    df = pd.read_excel(
        "../data/Fleek_-_Physical_Store_Acquisition_-_Pipeline_Data.xlsx",
        sheet_name="Part1_Manchester_scrape",
    )

    df["is_genuine_vintage_clothing"] = df.apply(is_genuine_vintage_clothing_shop, axis=1)
    genuine = df[df["is_genuine_vintage_clothing"]].copy()

    genuine["visit_priority_score"] = genuine.apply(visit_priority_score, axis=1)
    genuine["cluster_bonus"] = compute_cluster_bonus(genuine)
    genuine["visit_priority_score"] = genuine["visit_priority_score"] + genuine["cluster_bonus"]
    genuine = genuine.sort_values("visit_priority_score", ascending=False)

    genuine["why"] = genuine.apply(
        lambda r: f"rating {r['rating']}, {int(r['review_count'])} reviews, "
                  f"price {r['price_level']}, {int(r['cluster_bonus'] / 0.3)} genuine shops "
                  f"within 500m, {'has website' if isinstance(r['website'], str) else 'no website'}, "
                  f"{'has phone' if isinstance(r['phone'], str) else 'no phone'}: \"{r['top_review']}\"",
        axis=1,
    )

    out_cols = [
        "place_name", "maps_category", "full_address", "rating", "review_count",
        "price_level", "cluster_bonus", "top_review", "visit_priority_score", "why", "lat", "lng",
    ]
    genuine[out_cols].to_csv("../output/manchester_visit_list.csv", index=False)

    print(f"Filtered {len(df)} scraped places down to {len(genuine)} genuine vintage clothing shops.")
    print(f"Top 10 to visit:\n")
    print(genuine[["place_name", "visit_priority_score", "top_review"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

"""
Part 2, step 3: which city should we go after next, and why.
Run clean_data.py first.

Prioritisation logic — a city is worth going after if it has:
1. Enough genuine shop density to make a trip worthwhile (not 2 shops)
2. High revenue potential (est_monthly_spend_gbp across leads)
3. A "warm" stage mix — lots of replied/negotiating leads outperform a
   city that's all cold "new" leads, since it means less cold-start work
4. A reasonable win rate so far (customers won vs. lost), as a signal the
   market actually converts, not just generates leads
"""

import pandas as pd


def score_city(group: pd.DataFrame) -> dict:
    total_leads = len(group)
    total_potential_spend = group["est_monthly_spend_gbp"].sum()

    warm_stages = group["stage_clean"].isin(["replied", "negotiating", "customer"])
    warm_ratio = warm_stages.sum() / total_leads if total_leads else 0

    won = (group["stage_clean"] == "customer").sum()
    lost = (group["stage_clean"] == "lost").sum()
    win_rate = won / max(won + lost, 1)

    physical_shops = (group["channel_clean"] == "physical").sum()

    # Weighted score — density and revenue matter most for a physical-visit motion,
    # warmth and win rate are the "is this market actually working" check.
    score = (
        (physical_shops * 2) +
        (total_potential_spend / 1000) +
        (warm_ratio * 20) +
        (win_rate * 15)
    )

    return {
        "city": group.name,
        "physical_shops": physical_shops,
        "total_leads": total_leads,
        "total_potential_spend_gbp": total_potential_spend,
        "warm_ratio": round(warm_ratio, 2),
        "win_rate": round(win_rate, 2),
        "priority_score": round(score, 1),
    }


def main():
    df = pd.read_csv("../output/leads_cleaned.csv")
    df = df[df["city"].notna()]

    city_scores = df.groupby("city").apply(score_city, include_groups=False)
    result = pd.DataFrame(list(city_scores)).sort_values("priority_score", ascending=False)

    result["why"] = result.apply(
        lambda r: f"{r['physical_shops']} physical shops, £{r['total_potential_spend_gbp']:,.0f} "
                  f"potential monthly spend, {r['warm_ratio']*100:.0f}% warm leads, "
                  f"{r['win_rate']*100:.0f}% win rate",
        axis=1,
    )

    result.to_csv("../output/city_ranking.csv", index=False)
    print(result[["city", "priority_score", "why"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Part 2, step 1: clean the messy leads/customers data before anything else
can hold up — dedupe, canonicalise stage names, parse dates, infer the
channel (physical shop vs online reseller) since lead_channel_label is
explicitly untrustworthy per the Readme.
"""

import pandas as pd
import re
from dateutil import parser as dateparser

# Canonical stage buckets — drives which outbound message template gets used.
STAGE_MAP = {
    # never contacted
    "new": "new", "new lead": "new", "not contacted": "new",
    "new - inbound": "new",
    # contacted, no reply yet
    "contacted": "contacted", "emailed": "contacted", "1st touch sent": "contacted",
    # replied / in conversation
    "replied": "replied", "responded": "replied", "in conversation": "replied",
    "in convo": "replied",
    # active deal in progress
    "negotiating": "negotiating", "sent pricing": "negotiating",
    "quote sent": "negotiating", "meeting set": "negotiating",
    "visit booked": "negotiating", "visiting": "negotiating",
    "demo booked": "negotiating", "trial pending": "negotiating",
    # won / active customer
    "won": "customer", "closed - won": "customer", "closed won": "customer",
    # lost
    "no fit": "lost", "not interested": "lost", "closed - lost": "lost",
    "closed lost": "lost", "lost": "lost",
    # was a customer, has gone quiet — win-back territory
    "churned": "churned", "lapsed": "churned", "dormant": "churned",
    "stopped buying": "churned",
}


def canonical_stage(raw_stage: str) -> str:
    if not isinstance(raw_stage, str):
        return "new"
    key = raw_stage.strip().lower()
    return STAGE_MAP.get(key, "new")  # default unseen variants to "new", safest fallback


def parse_messy_date(value):
    if pd.isna(value) or value == "":
        return pd.NaT
    if isinstance(value, (pd.Timestamp,)):
        return value
    try:
        return pd.Timestamp(dateparser.parse(str(value), dayfirst=True, fuzzy=True))
    except (ValueError, TypeError):
        return pd.NaT


def infer_channel(row) -> str:
    """lead_channel_label is explicitly untrustworthy per the Readme — infer
    physical vs online reseller from what we can actually observe: physical
    shops have an address/city; pure resellers typically don't, and often
    have items_listed / sell_through_rate (Depop/Vinted/Whatnot-only signals)."""
    has_address = isinstance(row.get("address"), str) and row.get("address").strip() != ""
    has_reseller_metrics = pd.notna(row.get("items_listed")) or pd.notna(row.get("sell_through_rate"))

    if has_address and not has_reseller_metrics:
        return "physical"
    if has_reseller_metrics and not has_address:
        return "online_reseller"
    if has_address and has_reseller_metrics:
        return "physical"  # has a real address — treat as physical, reseller metrics secondary
    return "unknown"


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


# Same problem as Part 1's Manchester scrape, discovered on inspection: this
# "leads and customers" tab also mixes in non-vintage-clothing businesses
# (antiques, records, charity shops, furniture, homeware, costume hire) —
# 29 of 206 rows. One was even marked "Closed Won", which would have silently
# counted a record shop as a real customer in city_ranking.py. Reusing the
# same review/notes-text-driven logic as filter_shops.py, adapted to this
# tab's notes + google_maps_category fields.
NON_CLOTHING_LEAD_CATEGORIES = [
    "antique", "antiques & collectibles", "record store", "furniture store",
    "charity shop", "homeware store", "bric-a-brac", "costume shop",
]
CLOTHING_LEAD_KEYWORDS = [
    "vintage", "denim", "carhartt", "streetwear", "y2k", "grunge", "reworked",
    "archive designer", "sportswear", "workwear", "levi", "band tees",
    "hand-picked", "consignment",
]


def is_genuine_lead(row) -> bool:
    """Default to including a lead (this tab is mostly genuine, unlike Part 1's
    scrape) — only exclude when the category is a known non-clothing type AND
    the notes give no clothing signal either. Checked directly: no junk-category
    row in this data has a hidden clothing signal in its notes, so this is a
    safe, non-lossy filter here — but checking both, not just category alone,
    keeps the logic consistent with Part 1's principle that a review/notes
    signal can override a misleading category."""
    category = row.get("google_maps_category")
    notes = row.get("notes")

    category_is_junk = isinstance(category, str) and any(
        c in category.lower() for c in NON_CLOTHING_LEAD_CATEGORIES
    )
    if not category_is_junk:
        return True  # online resellers (blank category) and clothing categories both pass by default

    notes_text = notes.lower() if isinstance(notes, str) else ""
    has_clothing_signal = any(k in notes_text for k in CLOTHING_LEAD_KEYWORDS)
    return has_clothing_signal


def tidy_contacts(row) -> dict:
    """The Readme flags missing email/phone/owner_name as part of the mess to
    tidy. Phone formats here are already consistent (+CC prefixed) — the real
    problem is MISSING contact info, which determines what outreach is even
    possible. Rather than just flagging gaps, work out the best usable contact
    method per lead so draft_outreach.py can act on it directly."""
    has_email = isinstance(row.get("email"), str) and row.get("email").strip() != ""
    has_phone = isinstance(row.get("phone"), str) and row.get("phone").strip() != ""
    has_owner_name = isinstance(row.get("owner_name"), str) and row.get("owner_name").strip() != ""
    has_instagram = isinstance(row.get("instagram_handle"), str) and row.get("instagram_handle").strip() != ""

    # Preference order depends on channel: online resellers are reached via
    # Instagram DM per the brief; physical shops prefer email, falling back to
    # phone, falling back to "in-person only" if nothing else is on file.
    if row.get("channel_clean") == "online_reseller" and has_instagram:
        best_contact_method = "instagram_dm"
    elif has_email:
        best_contact_method = "email"
    elif has_phone:
        best_contact_method = "phone"
    elif has_instagram:
        best_contact_method = "instagram_dm"
    else:
        best_contact_method = "in_person_only"

    contact_completeness = sum([has_email, has_phone, has_owner_name, has_instagram])

    return {
        "has_email": has_email,
        "has_phone": has_phone,
        "has_owner_name": has_owner_name,
        "best_contact_method": best_contact_method,
        "contact_completeness": contact_completeness,
    }


def main():
    df = pd.read_excel(
        "../data/Fleek_-_Physical_Store_Acquisition_-_Pipeline_Data.xlsx",
        sheet_name="Part2_leads_and_customers",
    )

    df["stage_clean"] = df["lead_stage"].apply(canonical_stage)
    df["channel_clean"] = df.apply(infer_channel, axis=1)
    df["last_contact_date_clean"] = df["last_contact_date"].apply(parse_messy_date)
    df["last_purchase_date_clean"] = df["last_purchase_date"].apply(parse_messy_date)
    df["_name_key"] = df["store_name"].apply(normalise_name)
    df["is_genuine_lead"] = df.apply(is_genuine_lead, axis=1)

    contact_info = df.apply(tidy_contacts, axis=1, result_type="expand")
    df = pd.concat([df, contact_info], axis=1)

    excluded_count = (~df["is_genuine_lead"]).sum()
    excluded_categories = df[~df["is_genuine_lead"]]["google_maps_category"].value_counts()
    df = df[df["is_genuine_lead"]].copy()

    # Dedupe: same normalised name + same city = same shop. Keep the row with
    # the most "advanced" stage (customer > negotiating > replied > contacted > new),
    # since a duplicate often has a stale earlier stage recorded alongside a newer one.
    stage_priority = {"customer": 5, "churned": 4, "negotiating": 3, "replied": 2,
                       "contacted": 1, "new": 0, "lost": 0}
    df["_stage_rank"] = df["stage_clean"].map(stage_priority)
    df = df.sort_values("_stage_rank", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset=["_name_key", "city"], keep="first")
    after = len(df)

    df = df.drop(columns=["_name_key", "_stage_rank"])
    df.to_csv("../output/leads_cleaned.csv", index=False)

    print(f"Excluded {excluded_count} non-vintage-clothing leads (antiques/records/charity/"
          f"furniture/homeware/costume) before deduping — these aren't real Fleek prospects:")
    print(excluded_categories)
    print()
    print(f"Cleaned {before} rows down to {after} after deduping (of the remaining genuine leads).")
    print(f"\nCanonical stage distribution:\n{df['stage_clean'].value_counts()}")
    print(f"\nChannel distribution:\n{df['channel_clean'].value_counts()}")
    print(f"\nBest contact method distribution:\n{df['best_contact_method'].value_counts()}")
    print(f"\nLeads with NO usable contact (in-person only): {(df['best_contact_method']=='in_person_only').sum()}")


if __name__ == "__main__":
    main()

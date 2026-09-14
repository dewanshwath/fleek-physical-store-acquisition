# Fleek — Physical Store Acquisition Case Study

## Setup

```bash
pip install pandas openpyxl python-dateutil anthropic scipy python-dotenv
```

**Note:** if you hit a `numpy`/`scipy` import error (e.g. `ModuleNotFoundError: No module named 'numpy._core...'`), this is usually caused by a very new Python version (e.g. 3.14) that these packages don't yet have full stable support for. Fix: create the virtual environment with an older, stable version instead — e.g. `python3.12 -m venv .venv` — then reinstall.

Place the provided workbook in `data/`.

## Part 1 — Which shops do I visit? (Manchester)

```bash
cd part1_shop_filtering
python filter_shops.py
```

Outputs `output/manchester_visit_list.csv` — a ranked list of genuine vintage
clothing shops with a `why` column explaining the ranking.

**Approach:** category alone is too noisy to filter on (a "Boutique" or "Vintage
store" label can be anything, and genuine shops hide under generic categories).
The review text is the strongest signal in this data, so filtering is
keyword-scored off review text (weighted heavily) plus category (lighter weight,
as a tie-breaker) — not a single "vintage" keyword match, which the brief
explicitly warns will fail on name-traps like "Vintage Wines."

Ranking for visit priority combines rating, review count (as a size/footfall
proxy, log-scaled), price level (a proxy for a more serious/premium operation
worth the trip), geographic clustering (a shop with several other genuine
shops within 500m gets a bonus, since it means you can walk between them on
the same trip rather than treating every shop as an isolated stop), and
contactability. The Readme explicitly frames `website`/`phone`/`price_level`
together as a single "contactability and rough price positioning" signal —
checked directly, 19/34 genuine shops have a website and 25/34 have a phone
on file, both previously unused. A shop you can actually reach ahead of a
visit (confirm hours, check stock) is worth more than one you'd be showing up
to cold, so both get a small additive bonus. The clustering bonus is capped
so it acts as a tie-breaker, not the dominant signal over shop quality.

**Scaling note on clustering:** a naive pairwise distance check across every
shop is O(n²) — at 30,000 rows that's ~450 million comparisons, on the order
of an hour in pure Python, which fails the brief's explicit "wouldn't fall
over at 30,000 rows" requirement. `compute_cluster_bonus()` instead projects
lat/lng to a flat local approximation and uses `scipy.spatial.cKDTree` for
O(n log n) neighbour lookups. Tested directly at 30,000 simulated rows: 0.79
seconds, vs. the projected ~75 minutes for the naive approach.

## Part 2 — Outbound + city prioritisation

```bash
cd part2_outbound_and_prioritization
python clean_data.py      # dedupe, canonicalise stages/dates, infer channel
python draft_outreach.py  # personalised message per lead stage
python rank_cities.py     # which city to prioritise next
```

**Cleaning:** `lead_channel_label` is explicitly untrustworthy per the brief, so
channel (physical vs. online reseller) is inferred from whether a real address
exists vs. reseller-only metrics (items_listed, sell_through_rate). Stage names
are canonicalised via an explicit mapping table (not fuzzy string matching,
so it's auditable) into 7 buckets: new, contacted, replied, negotiating,
customer, churned, lost. Dedup keeps the most advanced stage per shop, since
duplicates often carry a stale earlier stage alongside a newer one.

**Non-clothing leads, found on inspection:** this tab has the same problem as
Part 1's Manchester scrape — 29 of 206 rows are non-vintage-clothing
businesses (antique stores, record shops, charity shops, furniture, homeware,
costume hire), identifiable the same way as Part 1: by category combined with
notes text ("Antique furniture & mid-century homeware", "Records, vinyl &
memorabilia", etc — checked directly, none of these hide a genuine clothing
signal in their notes, unlike some of Part 1's category traps). One of these
was marked "Closed Won" in the raw data, which would have silently counted a
record shop as a real Fleek customer. `is_genuine_lead()` excludes these
before anything else runs. This meaningfully changed the output — Amsterdam's
win rate dropped from an inflated 100% to an accurate 0% once its one "won"
deal (a non-clothing lead) was correctly excluded.

**Contacts:** checked directly — phone numbers here are already consistently
formatted (+CC prefixed), so the real mess is missing data, not format:
92/206 rows missing email, 99/206 missing phone, 132/206 missing owner name.
`tidy_contacts()` works out the single best usable contact method per lead
(email > phone > Instagram DM > in-person-only, weighted by channel), rather
than just flagging gaps — this feeds directly into `draft_outreach.py`, which
uses the actual tidied contact method rather than guessing from channel type
alone. 2 of the 192 cleaned leads have no usable remote contact info at all —
these get routed to a rep briefing note for an in-person visit instead of a
broken email/DM draft.

**Outreach:** each stage gets a distinct instruction set for tone and intent —
a cold "new" lead gets a low-commitment intro, a "churned" customer gets an
acknowledgement of the gap plus a concrete reason to return, an active
"customer" gets a relationship check-in rather than a sales pitch. See
`draft_outreach.py`'s `stage_instructions` dict.

Note: `draft_message_via_claude()` calls the real Anthropic API (model `claude-sonnet-4-6`), reading the key from a local `.env` file (excluded from git via `.gitignore`). To run this yourself, create a `.env` file in the project root with `ANTHROPIC_API_KEY=your-key-here`.

**Objection handling:** "skeptical" isn't a pipeline stage in this data —
checked directly, the same objection ("thinks Fleek is for small resellers")
shows up in the notes field across three different stages (New, negotiating,
sent pricing). So it's handled as a layer independent of stage: `detect_objection()`
scans the notes for known objection patterns (small-reseller perception, prior
churn, price perception) and, when found, adds a specific counter-argument
instruction on top of whatever the stage would normally call for — rather than
inventing a separate "skeptical" stage that wouldn't match how the real data
is structured. One resolved edge case worth noting: a "new"/cold-stage lead
with a detected objection creates a contradiction (the stage instruction
assumes zero familiarity with Fleek, but an objection implies they've already
heard of us) — the prompt explicitly resolves this rather than leaving it
silently inconsistent.

City ranking: scores each city on physical shop density, total revenue
potential, ratio of "warm" leads (replied/negotiating/customer vs. cold), and
win rate (won vs. lost) — not just raw lead count, since a city full of cold,
never-converting leads isn't actually a good next bet. Also factors in total
follower reach among online resellers in that city: checked the data
dictionary directly, `followers` is a given column (184/206 leads populated)
that wasn't being used anywhere — a city where resellers have a large
combined following represents brand visibility value beyond direct revenue,
so it's added as a smaller-weighted signal (scaled by /50,000) alongside the
main revenue and conversion factors, not as a replacement for them.

**A note on design tradeoffs, for the debrief:** the brief lists name, category,
review text, and rating as signals to combine for filtering. In practice, review
text combined with category caught every name-trap in this dataset (Vintage
Wines, Vintage Barber Co, etc. — all correctly excluded) without needing to
inspect the shop name directly, since the review text already tells you what's
actually sold regardless of what the name implies. Rating is used for ranking
(quality signal) rather than filtering, since a low rating doesn't mean a shop
isn't genuinely a vintage clothing shop — it just means it's a lower-priority
visit. Verified this holds on the real data rather than assumed.

- **Sourcing new cities**: Google Places API for the initial scrape (same
  source as this data), cross-referenced with Instagram hashtag/location
  search for shops that don't show up cleanly on Maps. Could run on a schedule
  per target city via a simple queue rather than manually.
- **At 30,000 rows**: the clustering step specifically was tested and fixed
  for this scale (see above — cKDTree, 0.79s at 30,000 simulated rows). The
  rest of the pipeline (filtering, cleaning, dedup) was also stress-tested at
  30,000 simulated rows and completed in under 2 seconds. Beyond that, the
  scripts are currently pandas-in-memory — fine at this scale, but the next
  step for real production scale would be a proper DB (Postgres/Supabase)
  with the scoring logic as a batch job, so filtering/ranking runs
  incrementally on new rows rather than reprocessing everything each time.

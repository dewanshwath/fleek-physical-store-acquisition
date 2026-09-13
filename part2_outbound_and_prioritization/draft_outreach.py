"""
Part 2, step 2: personalised outbound. Tone and content change by stage —
not one template with a name swapped in. Run clean_data.py first.

Stage -> intent mapping:
- new: cold intro, no assumed familiarity
- contacted: gentle follow-up, add a reason to reply now
- replied: move the conversation forward concretely (pricing/visit)
- negotiating: remove friction, close the specific open item
- customer (won): light-touch relationship maintenance, not a sales pitch
- churned: win-back — acknowledge the gap, give a concrete reason to return
- lost: low-priority, long-gap re-approach only if genuinely warranted

OBJECTION HANDLING (orthogonal to stage):
"Skeptical" isn't a pipeline stage in this data — it's a trait that shows up
in the notes field regardless of what stage a lead is at (confirmed: this
data has 3 leads across 3 different stages — New, negotiating, sent pricing —
all carrying the same "thinks it's for small resellers" objection). So this
is handled as a layer on top of the stage instruction, not a separate stage:
if the notes signal a known objection, the prompt is told to address that
objection specifically, on top of whatever the stage would normally call for.
"""

import pandas as pd
import re
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Known objection patterns -> the specific counter-argument to include.
# Matched against the notes field, independent of lead_stage.
OBJECTION_PATTERNS = {
    r"for small resellers|too small|small.reseller": (
        "They believe Fleek is only for small resellers / too low-end for them. "
        "Directly and briefly counter this — mention that Fleek works with shops "
        "buying serious volume (100+ units at a time) and that pricing/quality tiers "
        "exist for exactly their kind of operation. Don't be defensive about it, "
        "just correct the misconception factually and move on to the ask."
    ),
    r"tried (us |fleek )?(before|years ago)|used to use|churned|old product": (
        "They tried Fleek before and it didn't work out. Acknowledge that plainly "
        "without over-apologising, and give a concrete, specific reason things are "
        "different now (better product, more supply, whatever fits) rather than a "
        "vague 'we've improved' line."
    ),
    r"too cheap|low.?pricing|cheap on the app": (
        "They've seen low prices on the app and assume it's not for a serious/quality "
        "operation like theirs. Correct this directly — explain that pricing varies by "
        "grade and volume, and that what they'd see sourcing at their scale is different "
        "from the low end visible on the consumer-facing app."
    ),
}


def detect_objection(notes: str) -> str | None:
    """Scan notes for a known objection pattern, independent of pipeline stage."""
    if not isinstance(notes, str):
        return None
    text = notes.lower()
    for pattern, instruction in OBJECTION_PATTERNS.items():
        if re.search(pattern, text):
            return instruction
    return None


def build_prompt(row) -> str:
    stage = row["stage_clean"]
    name = row["store_name"]
    notes = row["notes"] if isinstance(row["notes"], str) else ""

    # Use the tidied contact method from clean_data.py rather than guessing —
    # a physical shop with no email on file still can't be emailed, regardless
    # of channel type.
    contact_method_map = {
        "email": "email", "phone": "phone call", "instagram_dm": "Instagram DM",
        "in_person_only": "in-person visit (no other contact info on file)",
    }
    contact_channel = contact_method_map.get(row.get("best_contact_method"), "email")

    context = f"Shop/reseller: {name}. What we know: {notes or 'no notes on file'}."

    stage_instructions = {
        "new": (
            "This is a COLD first touch — they have never heard from us. "
            "Introduce Fleek briefly, reference something specific about what they sell "
            "(from the notes) to show this isn't a mass blast, and end with a low-commitment ask "
            "(a quick call, or 'worth a look?'). Do not assume any familiarity with Fleek."
        ),
        "contacted": (
            "We reached out before and haven't heard back. Gentle follow-up — don't repeat the "
            "full pitch, add ONE new reason to reply now (e.g. a relevant stock drop, a case study), "
            "and keep it very short."
        ),
        "replied": (
            "They've engaged with us before. Move the conversation forward concretely — "
            "propose a specific next step (a call, a visit, sending pricing) rather than "
            "re-introducing Fleek."
        ),
        "negotiating": (
            "We're mid-deal. Be direct and practical — reference that we're close, "
            "remove friction, and ask for the specific next step to close (e.g. confirm the visit, "
            "send over terms). No re-pitching."
        ),
        "customer": (
            "They are an ACTIVE customer. This is NOT a sales message — it's a relationship "
            "check-in. Ask how things are going, mention anything new that suits what they sell, "
            "and make clear this is about the relationship, not extracting another sale."
        ),
        "churned": (
            "They used to buy from us and have gone quiet. This is a WIN-BACK message. "
            "Acknowledge the gap lightly without being awkward about it, don't apologise excessively, "
            "and give a concrete, specific reason to come back (new stock relevant to what they sell, "
            "or what's changed since they left)."
        ),
        "lost": (
            "This lead was marked not interested or no fit previously. Only worth a very "
            "light, low-pressure re-approach — acknowledge it's been a while, note something may "
            "have changed, and make it easy to ignore if still not relevant. Do not be pushy."
        ),
    }

    base_instruction = stage_instructions.get(stage, stage_instructions["new"])
    objection_instruction = detect_objection(notes)

    full_instruction = base_instruction
    if objection_instruction:
        # An objection implies prior awareness of Fleek, which can contradict a
        # "new"/cold-stage instruction that assumes zero familiarity. Resolve
        # explicitly rather than leaving a contradictory prompt.
        full_instruction += (
            f" IMPORTANT — this lead also has a specific objection to address: "
            f"{objection_instruction} Note: since they already have an opinion about "
            f"Fleek, treat them as aware of us even if the stage instruction above "
            f"assumed no familiarity — don't introduce Fleek as if from scratch."
        )

    if row.get("best_contact_method") == "in_person_only":
        # No usable remote contact on file — this isn't a message to draft,
        # it's a rep instruction for a shop visit.
        return (
            f"This lead ({name}) has no email, phone, or Instagram on file — the only "
            f"way to reach them is an in-person visit. {context} Write a 2-sentence "
            f"briefing note for the rep visiting in person: what to open with and what "
            f"the objection/context to be aware of is, given: {full_instruction}"
        )

    return (
        f"Draft a short {contact_channel} message to a vintage clothing shop/reseller. "
        f"{context} {full_instruction} "
        f"Keep it under 70 words, no corporate tone, sound like a real person who knows "
        f"the secondhand/vintage clothing trade."
    )


def draft_message_via_claude(prompt: str) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    df = pd.read_csv("../output/leads_cleaned.csv")

    # Sample a few from EVERY stage so the Loom can show tone genuinely differing —
    # not just running the whole 192 rows blindly.
    stage_samples = df.groupby("stage_clean").head(2)

    # Also explicitly pull in every lead with a detected objection, regardless of
    # stage, so the Loom can show objection-handling as its own dimension —
    # this is a real, separate signal from stage and deserves its own examples.
    has_objection = df["notes"].apply(lambda n: detect_objection(n) is not None)
    objection_samples = df[has_objection]

    samples = pd.concat([stage_samples, objection_samples]).drop_duplicates(subset=["lead_id"])

    results = []
    for _, row in samples.iterrows():
        prompt = build_prompt(row)
        message = draft_message_via_claude(prompt)
        results.append({
            "store_name": row["store_name"],
            "stage": row["stage_clean"],
            "channel": row["channel_clean"],
            "has_objection": detect_objection(row["notes"]) is not None,
            "prompt_used": prompt,
            "drafted_message": message,
        })

    out = pd.DataFrame(results)
    out.to_csv("../output/outbound_drafts.csv", index=False)
    print(f"Drafted outreach for {len(out)} sample leads across all stages.")
    print(f"Of which {out['has_objection'].sum()} carry a detected objection to address.")
    print(out[["store_name", "stage", "channel", "has_objection"]].to_string(index=False))
    print("\n--- Generated messages ---\n")
    for _, row in out.iterrows():
        print(f"[{row['stage']}] {row['store_name']} ({row['channel']}, objection={row['has_objection']})")
        print(row["drafted_message"])
        print()


if __name__ == "__main__":
    main()

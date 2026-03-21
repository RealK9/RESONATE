"""
RESONATE — Claude AI Analysis Engine.
Interprets sonic fingerprints and creates scoring templates.
"""

import re
import json

from config import claude_client, HAS_CLAUDE
from analysis.genre_profiles import get_genre_profile


def claude_analyze_track(track_profile):
    """Ask Claude to interpret the sonic fingerprint and create a scoring template."""
    if not HAS_CLAUDE:
        return None

    print("  Consulting Claude AI...")

    fp = track_profile
    detected_genre = fp.get("detected_genre", "unknown")
    genre_ref = get_genre_profile(detected_genre)

    prompt = f"""You are an expert mix engineer and music producer. Analyze this track's measured audio data and create a scoring template.

MEASURED DATA (from Essentia audio analysis — these are real measurements, not guesses):
- Key: {fp.get('key')} ({fp.get('key_confidence')}% confidence)
- BPM: {fp.get('bpm')}
- Duration: {fp.get('duration')}s
- Loudness: {fp.get('loudness')}dB
- Frequency band energy distribution (% of total):
{json.dumps(fp.get('frequency_bands', {}), indent=2)}
- Detected instruments: {fp.get('detected_instruments', [])}
- Frequency gaps (bands below threshold): {fp.get('frequency_gaps', [])}
- Heuristic genre from audio: {detected_genre}

REFERENCE: For a professional {detected_genre} track, the ideal frequency balance is:
{json.dumps(genre_ref['freq_balance'], indent=2)}

Respond with ONLY a JSON object. No explanation, no markdown fences:
{{
  "genre": "specific genre/subgenre",
  "mood": "2-3 word mood description",
  "energy": "Low/Medium/High",
  "summary": "One sentence describing what this track sounds like and what it needs",
  "what_track_has": ["list", "of", "elements", "present"],
  "what_track_needs": ["specific", "elements", "missing"],
  "avoid": ["elements", "to", "NOT", "add"],
  "type_priority_scores": {{
    "melody": 0-100,
    "vocals": 0-100,
    "hihat": 0-100,
    "pad": 0-100,
    "strings": 0-100,
    "fx": 0-100,
    "percussion": 0-100,
    "bass": 0-100,
    "kick": 0-100,
    "snare": 0-100,
    "unknown": 0-100
  }},
  "ideal_frequency_balance": {{
    "sub_bass_20_80": 0.XX,
    "bass_80_250": 0.XX,
    "low_mid_250_500": 0.XX,
    "mid_500_2k": 0.XX,
    "upper_mid_2k_6k": 0.XX,
    "presence_6k_12k": 0.XX,
    "air_12k_20k": 0.XX
  }}
}}

RULES for type_priority_scores:
- If the track ALREADY HAS an instrument type (detected_instruments), score it 5-15
- If the track DESPERATELY NEEDS a type (in frequency_gaps), score it 85-100
- Be extreme — big gaps between needed (90+) and not-needed (5-15) types
- The scores directly control sample ranking, so precision matters

RULES for ideal_frequency_balance:
- Values must sum to approximately 1.0
- Base it on what a FINISHED professional {detected_genre} track should measure like
- This track is clearly missing midrange — the ideal should reflect what needs to be added"""

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        result = json.loads(text)
        result["source"] = "claude"

        print(f"  Claude: {result.get('summary', '')[:80]}")
        print(f"  Genre: {result.get('genre', '?')}")
        print(f"  Mood: {result.get('mood', '?')}")
        print(f"  Type priorities: {result.get('type_priority_scores', {})}")
        print(f"  Track has: {result.get('what_track_has', [])}")
        print(f"  Track needs: {result.get('what_track_needs', [])}")
        return result
    except Exception as e:
        print(f"  Claude error: {e}")
        return None

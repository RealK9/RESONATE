"""
RESONATE Production Model (RPM) — Genre Taxonomy Knowledge Corpus

Comprehensive hierarchy of 500+ genres and subgenres used for:
1. Hierarchical softmax classification (predict top-level, then subgenre)
2. CLAP text-audio alignment description generation
3. Defining sonic characteristics per genre

Each Genre dataclass carries BPM ranges, key tendencies, time signatures,
defining sonic characteristics, typical instruments, production style,
energy/danceability/acousticness ranges, and lineage metadata.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Genre:
    name: str
    id: int
    parent: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    bpm_range: Tuple[int, int] = (120, 130)
    key_tendencies: List[str] = field(default_factory=list)
    time_signatures: List[str] = field(default_factory=lambda: ["4/4"])
    defining_characteristics: List[str] = field(default_factory=list)
    typical_instruments: List[str] = field(default_factory=list)
    production_style: str = ""
    era_of_origin: str = ""
    parent_genres: List[str] = field(default_factory=list)
    sibling_genres: List[str] = field(default_factory=list)
    energy_range: Tuple[float, float] = (0.4, 0.7)
    danceability_range: Tuple[float, float] = (0.4, 0.7)
    acousticness_range: Tuple[float, float] = (0.1, 0.4)
    famous_artists: List[str] = field(default_factory=list)
    clap_descriptions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Genre ID counter
# ---------------------------------------------------------------------------

_NEXT_ID = 0


def _id() -> int:
    global _NEXT_ID
    _NEXT_ID += 1
    return _NEXT_ID


# ---------------------------------------------------------------------------
# Build the full registry
# ---------------------------------------------------------------------------

def _build_genres() -> List[Genre]:
    """Return the complete list of Genre objects."""
    genres: List[Genre] = []

    # ======================================================================
    # ELECTRONIC  (~80 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Electronic", id=_id(), parent=None,
        aliases=["EDM", "electronic music", "electronica"],
        bpm_range=(100, 180), key_tendencies=["minor keys", "Dorian mode", "Phrygian mode"],
        time_signatures=["4/4"], defining_characteristics=[
            "synthesizer-driven", "drum machine rhythms", "repetitive structures",
            "build-and-drop dynamics", "heavy use of effects processing"],
        typical_instruments=["synthesizer", "drum machine", "sampler", "sequencer"],
        production_style="Layered digital production with emphasis on sound design, automation, and spatial effects",
        era_of_origin="1970s", parent_genres=[], sibling_genres=["Pop", "Hip-Hop/Rap"],
        energy_range=(0.4, 0.95), danceability_range=(0.5, 0.95), acousticness_range=(0.0, 0.15),
        famous_artists=["Kraftwerk", "Aphex Twin", "Daft Punk", "Skrillex", "Deadmau5"],
        clap_descriptions=[
            "an electronic music track with pulsing synthesizers and programmed drum machine rhythms",
            "a high-energy electronic dance track with layered synth pads and a driving four-on-the-floor kick",
        ],
    ))

    # --- House ---
    genres.append(Genre(
        name="House", id=_id(), parent="Electronic",
        aliases=["house music"], bpm_range=(118, 135),
        key_tendencies=["minor keys", "Dorian mode"], time_signatures=["4/4"],
        defining_characteristics=["four-on-the-floor kick", "off-beat hi-hats", "chord stabs",
                                  "soulful vocal samples", "warm basslines"],
        typical_instruments=["TR-909", "TR-808", "Juno-106", "Rhodes", "sampler"],
        production_style="Warm, groove-oriented production with soulful samples and analog-style synthesis",
        era_of_origin="1980s", parent_genres=["Disco", "Electronic"],
        sibling_genres=["Techno", "Garage"],
        energy_range=(0.5, 0.85), danceability_range=(0.7, 0.95), acousticness_range=(0.0, 0.15),
        famous_artists=["Frankie Knuckles", "Larry Heard", "Kerri Chandler", "Disclosure"],
        clap_descriptions=[
            "a groovy house track at 124 BPM with a four-on-the-floor kick, off-beat hi-hats, and warm Rhodes chords",
            "a soulful house music production with chopped vocal samples, analog bass, and rolling percussion",
        ],
    ))

    genres.append(Genre(
        name="Deep House", id=_id(), parent="House",
        aliases=["deep"], bpm_range=(118, 128),
        key_tendencies=["minor keys", "Dorian mode", "minor 7th chords"],
        defining_characteristics=["lush pads", "deep sub-bass", "jazzy chords", "muted percussion",
                                  "atmospheric textures"],
        typical_instruments=["Juno-106", "Rhodes", "Moog bass", "congas"],
        production_style="Smooth, hypnotic production with warm analog textures and subtle groove",
        era_of_origin="1980s", parent_genres=["House", "Jazz"],
        sibling_genres=["Tech House", "Progressive House"],
        energy_range=(0.3, 0.65), danceability_range=(0.6, 0.85), acousticness_range=(0.05, 0.2),
        famous_artists=["Larry Heard", "Kerri Chandler", "Moodymann", "Ron Trent"],
        clap_descriptions=[
            "a deep house groove at 122 BPM with lush Juno pads, a warm sub-bass, and jazzy Rhodes chords",
            "a late-night deep house track with muted percussion, atmospheric pads, and a hypnotic bassline",
        ],
    ))

    genres.append(Genre(
        name="Tech House", id=_id(), parent="House",
        aliases=["tech-house"], bpm_range=(122, 132),
        key_tendencies=["minor keys", "Phrygian mode"],
        defining_characteristics=["minimalist groove", "percussive loops", "filtered synths",
                                  "subtle acid lines", "tight low-end"],
        typical_instruments=["TR-909", "modular synth", "sampler"],
        production_style="Stripped-back, groove-focused production blending house warmth with techno precision",
        era_of_origin="1990s", parent_genres=["House", "Techno"],
        sibling_genres=["Deep House", "Minimal Techno"],
        energy_range=(0.5, 0.8), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.1),
        famous_artists=["Green Velvet", "Jamie Jones", "Fisher", "Patrick Topping"],
        clap_descriptions=[
            "a tech house groove at 128 BPM with a tight 909 kick, percussive loops, and a filtered acid bassline",
            "a peak-time tech house track with rolling percussion, subtle vocal chops, and a driving groove",
        ],
    ))

    genres.append(Genre(
        name="Progressive House", id=_id(), parent="House",
        aliases=["prog house"], bpm_range=(120, 132),
        key_tendencies=["minor keys", "melodic minor"],
        defining_characteristics=["long build-ups", "evolving textures", "lush pads",
                                  "arpeggiated synths", "emotional breakdowns"],
        typical_instruments=["software synths", "reverb-heavy pads", "pluck synths"],
        production_style="Layered, evolving arrangements with long transitions and emotional dynamics",
        era_of_origin="1990s", parent_genres=["House", "Trance"],
        sibling_genres=["Deep House", "Melodic Techno"],
        energy_range=(0.4, 0.8), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.15),
        famous_artists=["Sasha", "John Digweed", "Eric Prydz", "Yotto"],
        clap_descriptions=[
            "a progressive house track at 124 BPM with evolving pad textures, arpeggiated synths, and a long emotional build-up",
            "a melodic progressive house production with lush reverb-drenched pads and a hypnotic groove",
        ],
    ))

    genres.append(Genre(
        name="Acid House", id=_id(), parent="House",
        aliases=["acid"], bpm_range=(118, 135),
        key_tendencies=["minor keys", "chromatic"],
        defining_characteristics=["TB-303 squelch", "resonant filter sweeps", "hypnotic repetition",
                                  "raw analog sound"],
        typical_instruments=["TB-303", "TR-808", "TR-909"],
        production_style="Raw, hypnotic production centered on the squelching 303 bassline",
        era_of_origin="1980s", parent_genres=["House", "Electronic"],
        sibling_genres=["Acid Techno", "Chicago House"],
        energy_range=(0.5, 0.85), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.05),
        famous_artists=["Phuture", "DJ Pierre", "Adonis", "Armando"],
        clap_descriptions=[
            "an acid house track at 126 BPM with a squelching TB-303 bassline and driving 808 drums",
            "a raw acid house groove with resonant filter sweeps, hypnotic 303 patterns, and a pounding kick",
        ],
    ))

    genres.append(Genre(
        name="Tropical House", id=_id(), parent="House",
        aliases=["trop house"], bpm_range=(100, 118),
        key_tendencies=["major keys", "major 7th chords"],
        defining_characteristics=["steel drums", "marimba", "soft kick", "airy synths",
                                  "island-inspired melodies"],
        typical_instruments=["steel drum samples", "marimba", "soft synths", "acoustic guitar"],
        production_style="Light, breezy production with island-influenced percussion and bright melodic elements",
        era_of_origin="2010s", parent_genres=["House", "Pop"],
        sibling_genres=["Deep House", "Dance-Pop"],
        energy_range=(0.3, 0.6), danceability_range=(0.6, 0.8), acousticness_range=(0.1, 0.35),
        famous_artists=["Kygo", "Thomas Jack", "Sam Feldt"],
        clap_descriptions=[
            "a tropical house track at 110 BPM with steel drum melodies, soft kicks, and airy synth pads",
        ],
    ))

    genres.append(Genre(
        name="Future House", id=_id(), parent="House",
        aliases=["future-house"], bpm_range=(124, 130),
        key_tendencies=["minor keys"],
        defining_characteristics=["metallic bass stabs", "pitch-bent synths", "bouncy groove",
                                  "sharp transients"],
        typical_instruments=["Serum", "Massive", "Sylenth1"],
        production_style="Polished, modern production with distinctive metallic bass sounds and bouncy rhythms",
        era_of_origin="2010s", parent_genres=["House", "UK Garage"],
        sibling_genres=["Bass House", "Deep House"],
        energy_range=(0.6, 0.85), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.05),
        famous_artists=["Oliver Heldens", "Tchami", "Don Diablo"],
        clap_descriptions=[
            "a future house track at 128 BPM with metallic bass stabs, pitch-bent synths, and a bouncy groove",
        ],
    ))

    genres.append(Genre(
        name="Bass House", id=_id(), parent="House",
        aliases=["bassline house"], bpm_range=(124, 132),
        key_tendencies=["minor keys"],
        defining_characteristics=["heavy bass drops", "wobble bass", "aggressive groove",
                                  "UK garage influence"],
        typical_instruments=["Serum", "TR-909", "distorted bass synths"],
        production_style="High-energy production merging house structure with heavy bass design",
        era_of_origin="2010s", parent_genres=["House", "UK Bass"],
        sibling_genres=["Future House", "Bassline"],
        energy_range=(0.7, 0.95), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.05),
        famous_artists=["Jauz", "AC Slater", "Chris Lorenzo", "Habstrakt"],
        clap_descriptions=[
            "a bass house banger at 128 BPM with a heavy wobble bass drop and aggressive four-on-the-floor groove",
        ],
    ))

    genres.append(Genre(
        name="Electro House", id=_id(), parent="House",
        aliases=["electro-house", "dirty electro"], bpm_range=(126, 132),
        key_tendencies=["minor keys"],
        defining_characteristics=["big saw leads", "heavy sidechain compression", "massive drops",
                                  "festival-ready builds"],
        typical_instruments=["Massive", "Sylenth1", "distorted synths"],
        production_style="Aggressive, high-energy production with massive drops and saw-wave leads",
        era_of_origin="2000s", parent_genres=["House", "Electro"],
        sibling_genres=["Big Room", "Progressive House"],
        energy_range=(0.7, 0.95), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.05),
        famous_artists=["Deadmau5", "Wolfgang Gartner", "Feed Me", "Knife Party"],
        clap_descriptions=[
            "a high-energy electro house track at 128 BPM with massive saw-wave leads and heavy sidechain pumping",
        ],
    ))

    genres.append(Genre(
        name="Chicago House", id=_id(), parent="House",
        aliases=["Chicago", "classic house"], bpm_range=(118, 130),
        key_tendencies=["minor keys", "Dorian mode"],
        defining_characteristics=["raw drum machine patterns", "soulful vocals", "piano stabs",
                                  "analog warmth", "stripped-back arrangements"],
        typical_instruments=["TR-909", "TR-808", "Juno-106", "DX7"],
        production_style="Raw, stripped-back production with analog warmth and soulful elements",
        era_of_origin="1980s", parent_genres=["Disco", "Electronic"],
        sibling_genres=["Deep House", "Acid House"],
        energy_range=(0.5, 0.8), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.15),
        famous_artists=["Frankie Knuckles", "Marshall Jefferson", "Ron Hardy"],
        clap_descriptions=[
            "a classic Chicago house track with a raw 909 beat, soulful vocal stabs, and piano chords at 124 BPM",
        ],
    ))

    genres.append(Genre(
        name="UK Garage", id=_id(), parent="House",
        aliases=["UKG", "garage"], bpm_range=(130, 140),
        key_tendencies=["minor keys", "minor 7th chords"],
        defining_characteristics=["shuffled beats", "pitched-up vocals", "2-step rhythms",
                                  "bass-heavy", "syncopated percussion"],
        typical_instruments=["sampler", "sub-bass", "organ stabs"],
        production_style="Syncopated, bass-heavy production with shuffled rhythms and R&B-influenced vocals",
        era_of_origin="1990s", parent_genres=["House", "Jungle"],
        sibling_genres=["2-Step", "Speed Garage", "Bassline"],
        energy_range=(0.5, 0.8), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.1),
        famous_artists=["MJ Cole", "Artful Dodger", "Todd Edwards", "El-B"],
        clap_descriptions=[
            "a UK garage track at 135 BPM with shuffled 2-step beats, pitched-up vocals, and deep sub-bass",
        ],
    ))

    genres.append(Genre(
        name="Speed Garage", id=_id(), parent="House",
        aliases=["speed-garage"], bpm_range=(132, 140),
        key_tendencies=["minor keys"],
        defining_characteristics=["warping bass", "time-stretched vocals", "breakbeat elements",
                                  "heavy sub-bass"],
        typical_instruments=["sampler", "sub-bass synth", "breakbeats"],
        production_style="Bass-heavy UK production with warping basslines and garage rhythms",
        era_of_origin="1990s", parent_genres=["UK Garage", "House"],
        sibling_genres=["2-Step", "Bassline"],
        energy_range=(0.6, 0.85), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.05),
        famous_artists=["Armand Van Helden", "187 Lockdown", "Double 99"],
        clap_descriptions=[
            "a speed garage track at 136 BPM with warping bass, time-stretched vocal samples, and driving rhythms",
        ],
    ))

    genres.append(Genre(
        name="French House", id=_id(), parent="House",
        aliases=["French filter house", "French touch"], bpm_range=(118, 128),
        key_tendencies=["major keys", "disco chords"],
        defining_characteristics=["heavy filtering", "disco samples", "phaser effects",
                                  "compressed drums", "funky basslines"],
        typical_instruments=["sampler", "filters", "phaser", "compressor"],
        production_style="Sample-heavy production with heavy filtering of disco and funk records",
        era_of_origin="1990s", parent_genres=["House", "Disco"],
        sibling_genres=["Nu-Disco", "Electro House"],
        energy_range=(0.5, 0.8), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.15),
        famous_artists=["Daft Punk", "Cassius", "Stardust", "Modjo"],
        clap_descriptions=[
            "a French house track at 122 BPM with filtered disco samples, phaser-drenched chords, and a funky groove",
        ],
    ))

    genres.append(Genre(
        name="Funky House", id=_id(), parent="House",
        aliases=["funky"], bpm_range=(122, 132),
        key_tendencies=["major keys", "Mixolydian mode"],
        defining_characteristics=["funk guitar samples", "horn stabs", "disco-influenced",
                                  "uplifting energy"],
        typical_instruments=["funk guitar", "horns", "TR-909", "clavinet"],
        production_style="Uplifting, sample-driven production blending house beats with funk and disco elements",
        era_of_origin="1990s", parent_genres=["House", "Funk"],
        sibling_genres=["French House", "Disco House"],
        energy_range=(0.6, 0.85), danceability_range=(0.75, 0.95), acousticness_range=(0.05, 0.2),
        famous_artists=["Roger Sanchez", "Bob Sinclar", "Armand Van Helden"],
        clap_descriptions=[
            "a funky house groove at 126 BPM with chopped funk guitar, horn stabs, and a driving 909 beat",
        ],
    ))

    genres.append(Genre(
        name="Afro House", id=_id(), parent="House",
        aliases=["afro-house"], bpm_range=(118, 128),
        key_tendencies=["minor keys", "pentatonic scales"],
        defining_characteristics=["African percussion", "organic textures", "chanting vocals",
                                  "deep bass", "polyrhythmic patterns"],
        typical_instruments=["djembe", "shaker", "congas", "kalimba", "synth bass"],
        production_style="Deep, organic production blending African rhythms with electronic house elements",
        era_of_origin="2010s", parent_genres=["House", "Afrobeat"],
        sibling_genres=["Deep House", "Tribal House"],
        energy_range=(0.4, 0.75), danceability_range=(0.7, 0.9), acousticness_range=(0.1, 0.35),
        famous_artists=["Black Coffee", "Culoe De Song", "Da Capo"],
        clap_descriptions=[
            "an Afro house track at 122 BPM with organic African percussion, kalimba melodies, and deep sub-bass",
        ],
    ))

    # --- Techno ---
    genres.append(Genre(
        name="Techno", id=_id(), parent="Electronic",
        aliases=["techno music"], bpm_range=(125, 150),
        key_tendencies=["minor keys", "Phrygian mode", "atonal"],
        defining_characteristics=["relentless kick drums", "industrial textures", "hypnotic repetition",
                                  "minimalist arrangements", "dark atmospheres"],
        typical_instruments=["TR-909", "modular synth", "distortion units"],
        production_style="Dark, repetitive, machine-driven production emphasizing rhythm and texture over melody",
        era_of_origin="1980s", parent_genres=["Electronic", "Industrial"],
        sibling_genres=["House", "EBM"],
        energy_range=(0.6, 0.95), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.05),
        famous_artists=["Juan Atkins", "Derrick May", "Jeff Mills", "Richie Hawtin"],
        clap_descriptions=[
            "a dark techno track at 135 BPM with a pounding 909 kick, industrial textures, and hypnotic repetition",
            "a driving techno production with relentless percussion, modular synth drones, and a menacing atmosphere",
        ],
    ))

    genres.append(Genre(
        name="Detroit Techno", id=_id(), parent="Techno",
        aliases=["Detroit"], bpm_range=(125, 140),
        key_tendencies=["minor keys", "futuristic chords"],
        defining_characteristics=["futuristic synths", "soulful undertones", "machine funk",
                                  "sci-fi aesthetics"],
        typical_instruments=["TR-909", "Juno-106", "DX7", "SH-101"],
        production_style="Futuristic yet soulful production blending machine rhythms with emotive synthesis",
        era_of_origin="1980s", parent_genres=["Electronic", "Funk"],
        sibling_genres=["Berlin Techno", "Electro"],
        energy_range=(0.5, 0.85), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.1),
        famous_artists=["Juan Atkins", "Derrick May", "Kevin Saunderson", "Carl Craig"],
        clap_descriptions=[
            "a Detroit techno track at 130 BPM with futuristic Juno pads, a driving 909 beat, and soulful machine funk",
        ],
    ))

    genres.append(Genre(
        name="Berlin Techno", id=_id(), parent="Techno",
        aliases=["Berlin", "Berghain techno"], bpm_range=(128, 140),
        key_tendencies=["minor keys", "atonal"],
        defining_characteristics=["dark atmospheres", "cavernous reverb", "industrial sounds",
                                  "pounding kicks", "stripped-back arrangements"],
        typical_instruments=["modular synth", "TR-909", "distortion"],
        production_style="Dark, cavernous production designed for large warehouse and club spaces",
        era_of_origin="1990s", parent_genres=["Techno", "Industrial"],
        sibling_genres=["Detroit Techno", "Industrial Techno"],
        energy_range=(0.6, 0.9), danceability_range=(0.6, 0.8), acousticness_range=(0.0, 0.05),
        famous_artists=["Ben Klock", "Marcel Dettmann", "Surgeon"],
        clap_descriptions=[
            "a Berlin techno track at 132 BPM with cavernous reverb, a pounding kick, and dark industrial textures",
        ],
    ))

    genres.append(Genre(
        name="Minimal Techno", id=_id(), parent="Techno",
        aliases=["minimal", "micro-house"], bpm_range=(122, 135),
        key_tendencies=["minor keys", "atonal"],
        defining_characteristics=["stripped-back grooves", "micro-edits", "glitch elements",
                                  "subtle evolution", "space and silence"],
        typical_instruments=["modular synth", "laptop", "click sounds"],
        production_style="Extremely reduced production focusing on subtle textural changes and micro-rhythms",
        era_of_origin="1990s", parent_genres=["Techno"],
        sibling_genres=["Tech House", "Dub Techno"],
        energy_range=(0.3, 0.65), danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.1),
        famous_artists=["Richie Hawtin", "Ricardo Villalobos", "Monolake"],
        clap_descriptions=[
            "a minimal techno track at 128 BPM with sparse clicks, subtle glitch edits, and a hypnotic micro-groove",
        ],
    ))

    genres.append(Genre(
        name="Industrial Techno", id=_id(), parent="Techno",
        aliases=["industrial"], bpm_range=(130, 150),
        key_tendencies=["atonal", "dissonant"],
        defining_characteristics=["distorted kicks", "metal clangs", "noise textures",
                                  "aggressive energy", "factory sounds"],
        typical_instruments=["distortion pedals", "modular synth", "metal samples"],
        production_style="Harsh, aggressive production using distortion, noise, and industrial sound sources",
        era_of_origin="1990s", parent_genres=["Techno", "Industrial"],
        sibling_genres=["Berlin Techno", "Hard Techno"],
        energy_range=(0.7, 0.95), danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.05),
        famous_artists=["Ansome", "Headless Horseman", "Ancient Methods", "Perc"],
        clap_descriptions=[
            "an industrial techno track at 140 BPM with distorted kicks, metal clangs, and harsh noise textures",
        ],
    ))

    genres.append(Genre(
        name="Acid Techno", id=_id(), parent="Techno",
        aliases=["acid"], bpm_range=(130, 145),
        key_tendencies=["minor keys", "chromatic"],
        defining_characteristics=["TB-303 squelch", "driving rhythms", "relentless energy",
                                  "rave aesthetics"],
        typical_instruments=["TB-303", "TR-909", "modular synth"],
        production_style="Intense, 303-driven production combining acid squelch with techno's pounding rhythms",
        era_of_origin="1980s", parent_genres=["Techno", "Acid House"],
        sibling_genres=["Hard Techno", "Acid House"],
        energy_range=(0.7, 0.95), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.05),
        famous_artists=["Plastikman", "Dave Clarke", "Luke Vibert"],
        clap_descriptions=[
            "an acid techno track at 138 BPM with screaming 303 lines, pounding 909 drums, and relentless energy",
        ],
    ))

    genres.append(Genre(
        name="Hard Techno", id=_id(), parent="Techno",
        aliases=["hard-techno", "schranz"], bpm_range=(140, 160),
        key_tendencies=["atonal", "minor keys"],
        defining_characteristics=["distorted kicks", "aggressive loops", "fast tempo",
                                  "relentless energy", "dark atmosphere"],
        typical_instruments=["distorted synths", "TR-909", "noise generators"],
        production_style="Aggressive, fast-paced production with distorted elements and punishing rhythms",
        era_of_origin="1990s", parent_genres=["Techno"],
        sibling_genres=["Industrial Techno", "Hardcore"],
        energy_range=(0.8, 1.0), danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.05),
        famous_artists=["Chris Liebing", "Alignment", "VTSS"],
        clap_descriptions=[
            "a hard techno track at 150 BPM with distorted kicks, aggressive synth loops, and punishing energy",
        ],
    ))

    genres.append(Genre(
        name="Dub Techno", id=_id(), parent="Techno",
        aliases=["dub-techno"], bpm_range=(118, 132),
        key_tendencies=["minor keys", "suspended chords"],
        defining_characteristics=["heavy reverb", "delay feedback", "dub sirens", "hazy textures",
                                  "echo-drenched chords"],
        typical_instruments=["delay units", "reverb", "analog synths", "dub siren"],
        production_style="Spacious, echo-laden production combining dub reggae techniques with techno structure",
        era_of_origin="1990s", parent_genres=["Techno", "Dub"],
        sibling_genres=["Minimal Techno", "Ambient Techno"],
        energy_range=(0.3, 0.6), danceability_range=(0.5, 0.75), acousticness_range=(0.0, 0.15),
        famous_artists=["Basic Channel", "Deepchord", "Monolake", "Fluxion"],
        clap_descriptions=[
            "a dub techno track at 124 BPM with echo-drenched chords, deep reverb, and hazy delay textures",
        ],
    ))

    genres.append(Genre(
        name="Melodic Techno", id=_id(), parent="Techno",
        aliases=["melodic-techno"], bpm_range=(120, 132),
        key_tendencies=["minor keys", "melodic minor"],
        defining_characteristics=["emotive melodies", "lush pads", "driving beats",
                                  "cinematic builds", "hypnotic arpeggios"],
        typical_instruments=["Prophet", "OB-6", "reverb-heavy synths"],
        production_style="Emotional, expansive production blending techno drive with melodic and cinematic elements",
        era_of_origin="2010s", parent_genres=["Techno", "Progressive House"],
        sibling_genres=["Progressive House", "Deep Techno"],
        energy_range=(0.5, 0.85), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.1),
        famous_artists=["Tale Of Us", "Stephan Bodzin", "Adriatique", "Anyma"],
        clap_descriptions=[
            "a melodic techno track at 126 BPM with emotive synth melodies, lush pads, and a driving kick",
        ],
    ))

    # --- Trance ---
    genres.append(Genre(
        name="Trance", id=_id(), parent="Electronic",
        aliases=["trance music"], bpm_range=(128, 150),
        key_tendencies=["minor keys", "Aeolian mode", "Phrygian mode"],
        defining_characteristics=["arpeggiated synths", "soaring leads", "epic breakdowns",
                                  "build-and-release structure", "ethereal pads"],
        typical_instruments=["supersaw synths", "pluck leads", "reverb-heavy pads"],
        production_style="Euphoric, build-and-release production with soaring melodies and layered synths",
        era_of_origin="1990s", parent_genres=["Electronic", "Techno"],
        sibling_genres=["Progressive House", "Eurodance"],
        energy_range=(0.6, 0.95), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.05),
        famous_artists=["Armin van Buuren", "Paul van Dyk", "Tiesto", "Above & Beyond"],
        clap_descriptions=[
            "an uplifting trance track at 138 BPM with soaring supersaw leads, arpeggiated synths, and an epic breakdown",
            "an energetic trance production with layered pads, a driving bassline, and euphoric build-ups",
        ],
    ))

    for sub in [
        ("Progressive Trance", ["prog trance"], (128, 136), "Subtle, evolving trance with long builds and melodic development", "1990s",
         ["Trance", "Progressive House"], ["Progressive House", "Melodic Techno"],
         (0.5, 0.8), (0.6, 0.8), ["Sasha", "John Digweed", "Airwave"],
         "a progressive trance track at 132 BPM with evolving melodies, layered pads, and subtle build-ups"),
        ("Uplifting Trance", ["uplifting", "anthem trance"], (136, 142), "Euphoric, high-energy trance with massive builds and emotional melodies", "1990s",
         ["Trance"], ["Vocal Trance"],
         (0.7, 0.95), (0.6, 0.85), ["Aly & Fila", "Andrew Rayel", "Giuseppe Ottaviani"],
         "an uplifting trance anthem at 140 BPM with soaring leads, massive breakdowns, and euphoric energy"),
        ("Psytrance", ["psychedelic trance", "psy"], (138, 150), "Psychedelic, layered production with rapid arpeggios and alien textures", "1990s",
         ["Trance", "Goa Trance"], ["Goa Trance", "Full-On"],
         (0.7, 0.95), (0.6, 0.85), ["Infected Mushroom", "Astrix", "Vini Vici"],
         "a psytrance track at 145 BPM with rapid arpeggiated basslines, psychedelic textures, and alien sound design"),
        ("Goa Trance", ["Goa"], (140, 150), "Spiritual, psychedelic trance with Eastern influences and organic textures", "1990s",
         ["Electronic"], ["Psytrance"],
         (0.6, 0.9), (0.6, 0.8), ["Astral Projection", "Hallucinogen", "Juno Reactor"],
         "a Goa trance track at 145 BPM with Eastern melodic elements, layered acid lines, and spiritual atmosphere"),
        ("Vocal Trance", ["vocal"], (132, 140), "Trance centered around powerful vocal performances and emotional lyrics", "1990s",
         ["Trance", "Pop"], ["Uplifting Trance"],
         (0.6, 0.9), (0.6, 0.85), ["ATB", "Dash Berlin", "Gareth Emery"],
         "a vocal trance track at 138 BPM with a soaring female vocal, lush pads, and an emotional breakdown"),
        ("Hard Trance", ["hard-trance"], (140, 155), "Aggressive trance with harder kicks and faster tempo", "1990s",
         ["Trance", "Hard Techno"], ["Hardstyle"],
         (0.8, 0.95), (0.6, 0.8), ["Scot Project", "Yoji Biomehanika"],
         "a hard trance track at 148 BPM with a distorted kick, aggressive acid lines, and relentless energy"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Trance", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"], defining_characteristics=[sub[3].split(",")[0].strip()],
            typical_instruments=["synths", "pads"], production_style=sub[3], era_of_origin=sub[4],
            parent_genres=sub[5], sibling_genres=sub[6],
            energy_range=sub[7], danceability_range=sub[8], acousticness_range=(0.0, 0.05),
            famous_artists=sub[9], clap_descriptions=[sub[10]],
        ))

    # --- Drum & Bass ---
    genres.append(Genre(
        name="Drum and Bass", id=_id(), parent="Electronic",
        aliases=["DnB", "drum & bass", "D&B"], bpm_range=(160, 180),
        key_tendencies=["minor keys", "Aeolian mode"],
        defining_characteristics=["fast breakbeats", "heavy sub-bass", "chopped Amen break",
                                  "rapid rhythms", "complex drum patterns"],
        typical_instruments=["sampler", "sub-bass synth", "breakbeats", "Reese bass"],
        production_style="High-tempo production with complex breakbeat patterns and heavy sub-bass",
        era_of_origin="1990s", parent_genres=["Jungle", "Breakbeat"],
        sibling_genres=["Jungle", "Breakbeat"],
        energy_range=(0.7, 0.95), danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.1),
        famous_artists=["Goldie", "LTJ Bukem", "Andy C", "Noisia"],
        clap_descriptions=[
            "a drum and bass track at 174 BPM with chopped breakbeats, heavy sub-bass, and rapid-fire percussion",
            "a high-energy DnB production with rolling Amen breaks, a deep Reese bass, and intense build-ups",
        ],
    ))

    for sub in [
        ("Liquid DnB", ["liquid drum and bass", "liquid"], (170, 178),
         "Smooth, melodic DnB with lush pads, soulful vocals, and rolling breaks",
         (0.5, 0.75), ["LTJ Bukem", "Calibre", "High Contrast", "Netsky"],
         "a liquid drum and bass track at 174 BPM with lush pads, rolling breakbeats, and deep sub-bass"),
        ("Neurofunk", ["neuro"], (172, 178),
         "Dark, technically complex DnB with aggressive bass design and metallic textures",
         (0.8, 0.95), ["Noisia", "Mefjus", "Phace", "Ivy Lab"],
         "a neurofunk track at 175 BPM with aggressive modulated bass, metallic percussion, and dark atmosphere"),
        ("Jump-Up", ["jump up"], (172, 178),
         "Energetic, dancefloor-focused DnB with bouncy basslines and rave energy",
         (0.7, 0.9), ["Macky Gee", "Tyke", "Original Sin"],
         "a jump-up drum and bass track at 175 BPM with a bouncy bassline, rave stabs, and high energy"),
        ("Darkstep", ["dark DnB"], (172, 178),
         "Dark, aggressive DnB with horror-influenced sound design and punishing rhythms",
         (0.8, 0.95), ["Current Value", "Counterstrike", "Sinister Souls"],
         "a darkstep track at 175 BPM with horror-influenced pads, aggressive breaks, and menacing bass"),
        ("Jungle", ["jungle music", "oldskool jungle"], (155, 170),
         "Breakbeat-heavy proto-DnB with reggae/dub influences, chopped Amen breaks, and ragga vocals",
         (0.7, 0.9), ["Remarc", "Congo Natty", "DJ Hype", "General Levy"],
         "a jungle track at 165 BPM with chopped Amen breaks, dub bass, ragga vocal samples, and heavy reverb"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Drum and Bass", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"], defining_characteristics=[],
            typical_instruments=["breakbeats", "sub-bass", "sampler"], production_style=sub[3],
            era_of_origin="1990s", parent_genres=["Drum and Bass"], sibling_genres=[],
            energy_range=sub[4], danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.1),
            famous_artists=sub[5], clap_descriptions=[sub[6]],
        ))

    # --- Dubstep ---
    genres.append(Genre(
        name="Dubstep", id=_id(), parent="Electronic",
        aliases=["dubstep music"], bpm_range=(138, 142),
        key_tendencies=["minor keys", "Phrygian mode"],
        defining_characteristics=["wobble bass", "half-time rhythm", "heavy sub-bass",
                                  "sparse arrangements", "bass drops"],
        typical_instruments=["Massive", "Serum", "sub-bass synth"],
        production_style="Bass-heavy production with half-time rhythms and aggressive sound design",
        era_of_origin="2000s", parent_genres=["UK Garage", "Dub", "Drum and Bass"],
        sibling_genres=["Grime", "UK Bass"],
        energy_range=(0.5, 0.95), danceability_range=(0.4, 0.7), acousticness_range=(0.0, 0.05),
        famous_artists=["Skream", "Benga", "Digital Mystikz", "Skrillex"],
        clap_descriptions=[
            "a dubstep track at 140 BPM with heavy wobble bass, sparse half-time beats, and deep sub-bass drops",
        ],
    ))

    for sub in [
        ("Brostep", ["bro-step", "American dubstep"], (140, 150),
         "Aggressive, mid-range focused dubstep with heavy drops and complex sound design",
         (0.8, 1.0), ["Skrillex", "Excision", "Zomboy"],
         "an aggressive brostep track at 150 BPM with screeching mid-range bass, massive drops, and intense sound design"),
        ("Riddim", ["riddim dubstep"], (140, 150),
         "Repetitive, minimalist dubstep with triplet patterns and mechanical bass sounds",
         (0.7, 0.95), ["Virtual Riot", "Infekt", "Subtronics"],
         "a riddim dubstep track at 150 BPM with repetitive triplet bass patterns and mechanical wobble sounds"),
        ("Deep Dubstep", ["deep dub"], (138, 142),
         "Dark, spacious dubstep with sub-bass focus, dub influences, and meditative grooves",
         (0.3, 0.6), ["Digital Mystikz", "Mala", "Kode9"],
         "a deep dubstep track at 140 BPM with cavernous sub-bass, sparse beats, and dub-influenced atmospheres"),
        ("Future Bass", ["future-bass"], (130, 170),
         "Bright, synth-heavy electronic music with lush chord stacks, pitched vocals, and emotional drops",
         (0.5, 0.85), ["Flume", "San Holo", "Illenium", "Louis The Child"],
         "a future bass track at 150 BPM with lush supersaw chords, pitched vocal chops, and an emotional drop"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Dubstep", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"], defining_characteristics=[],
            typical_instruments=["Serum", "Massive", "sub-bass synth"], production_style=sub[3],
            era_of_origin="2000s", parent_genres=["Dubstep"], sibling_genres=[],
            energy_range=sub[4], danceability_range=(0.4, 0.7), acousticness_range=(0.0, 0.05),
            famous_artists=sub[5], clap_descriptions=[sub[6]],
        ))

    # --- Ambient ---
    genres.append(Genre(
        name="Ambient", id=_id(), parent="Electronic",
        aliases=["ambient music"], bpm_range=(60, 120),
        key_tendencies=["suspended chords", "open tunings", "major 7th chords"],
        defining_characteristics=["atmospheric textures", "no beat", "evolving drones",
                                  "field recordings", "spatial processing"],
        typical_instruments=["synthesizer", "reverb", "delay", "granular processor"],
        production_style="Textural, non-rhythmic production focused on atmosphere and space",
        era_of_origin="1970s", parent_genres=["Electronic", "Minimalist"],
        sibling_genres=["New Age", "Drone"],
        energy_range=(0.0, 0.3), danceability_range=(0.0, 0.15), acousticness_range=(0.05, 0.5),
        famous_artists=["Brian Eno", "Aphex Twin", "Stars of the Lid", "Tim Hecker"],
        clap_descriptions=[
            "an ambient soundscape with evolving pad textures, gentle drones, and deep reverb tails",
            "a meditative ambient piece with granular synthesis, field recordings, and slowly shifting harmonies",
        ],
    ))

    for sub in [
        ("Dark Ambient", ["dark-ambient"], (0, 80),
         "Ominous, unsettling ambient with dissonant drones and horror-influenced atmospheres",
         (0.1, 0.4), ["Lustmord", "Atrium Carceri", "Raison d'Etre"],
         "a dark ambient piece with ominous drones, dissonant textures, and unsettling low-frequency rumbles"),
        ("Space Ambient", ["space music"], (0, 90),
         "Expansive ambient evoking cosmic vastness with shimmering pads and spacious processing",
         (0.0, 0.2), ["Steve Roach", "Robert Rich", "Jonn Serrie"],
         "a space ambient track with shimmering cosmic pads, distant reverb tails, and a sense of infinite vastness"),
        ("Drone", ["drone music", "drone ambient"], (0, 60),
         "Extended, sustained tones creating immersive sonic environments",
         (0.0, 0.2), ["Sunn O)))", "Earth", "Eliane Radigue"],
         "a drone piece with sustained low-frequency tones, slowly evolving harmonic overtones, and immersive depth"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Ambient", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["atonal", "suspended"], defining_characteristics=[],
            typical_instruments=["synthesizer", "reverb", "granular processor"], production_style=sub[3],
            era_of_origin="1980s", parent_genres=["Ambient"], sibling_genres=[],
            energy_range=sub[4], danceability_range=(0.0, 0.1), acousticness_range=(0.1, 0.5),
            famous_artists=sub[5], clap_descriptions=[sub[6]],
        ))

    # --- IDM ---
    genres.append(Genre(
        name="IDM", id=_id(), parent="Electronic",
        aliases=["intelligent dance music", "braindance"], bpm_range=(80, 160),
        key_tendencies=["atonal", "complex harmony"],
        defining_characteristics=["complex rhythms", "glitch elements", "experimental structures",
                                  "cerebral sound design"],
        typical_instruments=["modular synth", "granular processor", "Max/MSP"],
        production_style="Experimental, cerebral electronic production with complex rhythms and unconventional structures",
        era_of_origin="1990s", parent_genres=["Electronic", "Ambient"],
        sibling_genres=["Glitch", "Ambient"],
        energy_range=(0.2, 0.7), danceability_range=(0.2, 0.6), acousticness_range=(0.0, 0.2),
        famous_artists=["Aphex Twin", "Autechre", "Boards of Canada", "Squarepusher"],
        clap_descriptions=[
            "an IDM track with glitchy, complex rhythms, warped synth textures, and an unpredictable structure",
        ],
    ))

    # --- Downtempo ---
    genres.append(Genre(
        name="Downtempo", id=_id(), parent="Electronic",
        aliases=["chill", "chill-out"], bpm_range=(70, 110),
        key_tendencies=["minor keys", "jazz chords", "major 7th"],
        defining_characteristics=["relaxed grooves", "atmospheric textures", "organic samples",
                                  "slow rhythms"],
        typical_instruments=["Rhodes", "turntable", "soft synths", "acoustic guitar"],
        production_style="Laid-back, textural production blending electronic and organic elements at slower tempos",
        era_of_origin="1990s", parent_genres=["Electronic", "Ambient"],
        sibling_genres=["Trip-Hop", "Lo-fi"],
        energy_range=(0.1, 0.45), danceability_range=(0.3, 0.6), acousticness_range=(0.1, 0.5),
        famous_artists=["Bonobo", "Tycho", "Thievery Corporation", "Zero 7"],
        clap_descriptions=[
            "a downtempo track at 90 BPM with laid-back beats, warm Rhodes chords, and atmospheric textures",
        ],
    ))

    for sub in [
        ("Trip-Hop", ["trip hop", "Bristol sound"], (70, 100),
         "Dark, cinematic downtempo with heavy beats, vinyl crackle, and moody atmospheres",
         (0.2, 0.5), ["Massive Attack", "Portishead", "Tricky", "DJ Shadow"],
         "a trip-hop track at 85 BPM with heavy downtempo beats, dark cinematic strings, and moody vocal samples"),
        ("Chillout", ["chill out", "chill-out"], (80, 110),
         "Relaxing electronic music designed for unwinding with soft textures and gentle rhythms",
         (0.1, 0.35), ["Zero 7", "Air", "Cafe Del Mar artists"],
         "a chillout track at 95 BPM with soft synth pads, gentle beats, and a warm, relaxing atmosphere"),
        ("Lo-fi Electronic", ["lo-fi beats", "lo-fi chill"], (70, 95),
         "Nostalgic, lo-fi production with vinyl crackle, tape hiss, and mellow jazzy samples",
         (0.1, 0.3), ["Nujabes", "J Dilla", "Tomppabeats"],
         "a lo-fi electronic beat at 80 BPM with vinyl crackle, mellow jazz piano samples, and a dusty drum loop"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Downtempo", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "jazz chords"], defining_characteristics=[],
            typical_instruments=["turntable", "Rhodes", "sampler"], production_style=sub[3],
            era_of_origin="1990s", parent_genres=["Downtempo"], sibling_genres=[],
            energy_range=sub[4], danceability_range=(0.3, 0.55), acousticness_range=(0.15, 0.5),
            famous_artists=sub[5], clap_descriptions=[sub[6]],
        ))

    # --- Breakbeat ---
    genres.append(Genre(
        name="Breakbeat", id=_id(), parent="Electronic",
        aliases=["breaks"], bpm_range=(120, 140),
        key_tendencies=["minor keys"],
        defining_characteristics=["syncopated drum breaks", "chopped samples", "funk-influenced rhythms"],
        typical_instruments=["sampler", "breakbeats", "synths"],
        production_style="Sample-based production centered on chopped, syncopated drum breaks",
        era_of_origin="1980s", parent_genres=["Electronic", "Funk"],
        sibling_genres=["Drum and Bass", "House"],
        energy_range=(0.5, 0.85), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.15),
        famous_artists=["The Chemical Brothers", "Fatboy Slim", "The Prodigy"],
        clap_descriptions=[
            "a breakbeat track at 130 BPM with chopped syncopated drums, funky bass, and sample-heavy production",
        ],
    ))

    genres.append(Genre(
        name="Big Beat", id=_id(), parent="Breakbeat",
        aliases=["big-beat"], bpm_range=(110, 140),
        key_tendencies=["minor keys"],
        defining_characteristics=["heavy breakbeats", "rock influence", "distorted bass",
                                  "big hooks", "festival energy"],
        typical_instruments=["sampler", "distorted guitar", "breakbeats", "synths"],
        production_style="Loud, hooky production fusing rock energy with electronic breakbeats",
        era_of_origin="1990s", parent_genres=["Breakbeat", "Rock"],
        sibling_genres=["Electro House"],
        energy_range=(0.7, 0.95), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.15),
        famous_artists=["The Chemical Brothers", "Fatboy Slim", "The Prodigy", "The Crystal Method"],
        clap_descriptions=[
            "a big beat track at 125 BPM with heavy breakbeats, distorted bass, and rock-influenced hooks",
        ],
    ))

    genres.append(Genre(
        name="Nu-Skool Breaks", id=_id(), parent="Breakbeat",
        aliases=["nu skool", "nu-skool"], bpm_range=(125, 140),
        key_tendencies=["minor keys"],
        defining_characteristics=["updated breakbeats", "bass-heavy", "modern production",
                                  "rave influence"],
        typical_instruments=["sampler", "bass synth", "breakbeats"],
        production_style="Modernized breakbeat production with heavier bass and cleaner production",
        era_of_origin="2000s", parent_genres=["Breakbeat"],
        sibling_genres=["UK Bass"],
        energy_range=(0.6, 0.85), danceability_range=(0.6, 0.8), acousticness_range=(0.0, 0.1),
        famous_artists=["Freq Nasty", "Elite Force", "Stanton Warriors"],
        clap_descriptions=[
            "a nu-skool breaks track at 132 BPM with updated breakbeats, heavy bass, and rave-influenced stabs",
        ],
    ))

    # --- Hardcore ---
    genres.append(Genre(
        name="Hardcore Electronic", id=_id(), parent="Electronic",
        aliases=["hardcore", "hardcore dance"], bpm_range=(150, 200),
        key_tendencies=["minor keys", "atonal"],
        defining_characteristics=["extremely fast tempo", "distorted kicks", "rave energy",
                                  "aggressive synths"],
        typical_instruments=["distorted synths", "kick drum", "hoover synth"],
        production_style="Extremely fast, aggressive production with distorted kicks and manic energy",
        era_of_origin="1990s", parent_genres=["Techno", "Rave"],
        sibling_genres=["Hard Techno", "Hardstyle"],
        energy_range=(0.85, 1.0), danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.05),
        famous_artists=["Angerfist", "Headhunterz"],
        clap_descriptions=[
            "a hardcore electronic track at 170 BPM with distorted kicks, aggressive synths, and manic rave energy",
        ],
    ))

    for sub in [
        ("Happy Hardcore", ["happy", "happy hard"], (160, 180),
         "Uplifting, euphoric hardcore with bright synths, pitched-up vocals, and manic energy",
         (0.85, 1.0), ["Hixxy", "Dougal", "Gammer"],
         "a happy hardcore track at 170 BPM with bright piano stabs, pitched-up vocals, and euphoric rave energy"),
        ("Gabber", ["gabber music", "gabba"], (160, 200),
         "Extremely aggressive hardcore with distorted kicks, screeching synths, and punishing tempos",
         (0.9, 1.0), ["The Prophet", "Neophyte", "Paul Elstak"],
         "a gabber track at 180 BPM with a massively distorted kick drum, screeching synths, and extreme energy"),
        ("Speedcore", ["speedcore music"], (250, 1000),
         "Extreme tempo electronic music pushing beyond 300 BPM with noise elements",
         (0.95, 1.0), ["Diabarha", "m1dy"],
         "a speedcore track at 300 BPM with noise-distorted kicks and extreme hyperspeed percussion"),
        ("Frenchcore", ["frenchcore music"], (170, 200),
         "French-style hardcore with rolling bass kicks, melodic elements, and high energy",
         (0.85, 1.0), ["Dr. Peacock", "Sefa", "Billx"],
         "a frenchcore track at 190 BPM with rolling bass kicks, melodic synth lines, and intense energy"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Hardcore Electronic", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"], defining_characteristics=[],
            typical_instruments=["distorted synths", "kick drum"], production_style=sub[3],
            era_of_origin="1990s", parent_genres=["Hardcore Electronic"], sibling_genres=[],
            energy_range=sub[4], danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.05),
            famous_artists=sub[5], clap_descriptions=[sub[6]],
        ))

    # --- More electronic subgenres ---
    genres.append(Genre(
        name="Electro", id=_id(), parent="Electronic",
        aliases=["electro music", "electro-funk"], bpm_range=(110, 135),
        key_tendencies=["minor keys", "Dorian mode"],
        defining_characteristics=["robotic vocals", "TR-808 beats", "synth bass", "funk influence",
                                  "mechanical feel"],
        typical_instruments=["TR-808", "vocoder", "synth bass", "DX7"],
        production_style="Mechanical, funk-influenced production with robotic vocals and 808 drum patterns",
        era_of_origin="1980s", parent_genres=["Electronic", "Funk"],
        sibling_genres=["Hip-Hop/Rap", "Synthwave"],
        energy_range=(0.5, 0.8), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.1),
        famous_artists=["Afrika Bambaataa", "Cybotron", "Egyptian Lover", "Drexciya"],
        clap_descriptions=[
            "an electro track at 125 BPM with a robotic vocoder, 808 beats, and mechanical funk bass",
        ],
    ))

    genres.append(Genre(
        name="Synthwave", id=_id(), parent="Electronic",
        aliases=["retrowave", "outrun", "synthwave music"], bpm_range=(80, 130),
        key_tendencies=["minor keys", "Dorian mode", "Aeolian mode"],
        defining_characteristics=["retro 80s synths", "analog warmth", "neon aesthetics",
                                  "arpeggiated basslines", "gated reverb"],
        typical_instruments=["Juno-106", "Prophet-5", "LinnDrum", "DX7"],
        production_style="Nostalgic 1980s-inspired production with analog synths, gated drums, and retro aesthetics",
        era_of_origin="2000s", parent_genres=["Electronic", "New Wave"],
        sibling_genres=["Electro", "New Wave"],
        energy_range=(0.3, 0.75), danceability_range=(0.4, 0.75), acousticness_range=(0.0, 0.1),
        famous_artists=["Kavinsky", "Perturbator", "Carpenter Brut", "Com Truise", "The Midnight"],
        clap_descriptions=[
            "a synthwave track at 110 BPM with warm analog arpeggios, gated reverb drums, and retro 80s atmosphere",
            "a dark synthwave piece with driving Juno bassline, pulsing LinnDrum beats, and neon-drenched pads",
        ],
    ))

    for sub in [
        ("Darksynth", ["dark synth", "dark synthwave"], (100, 140),
         "Aggressive, horror-influenced synthwave with distorted synths and menacing atmospheres",
         (0.6, 0.9), ["Perturbator", "Carpenter Brut", "Dan Terminus"],
         "a darksynth track at 120 BPM with distorted analog synths, menacing bass, and horror-film atmosphere"),
        ("Vaporwave", ["vaporwave music", "vapor"], (60, 120),
         "Slowed-down, nostalgic production sampling 80s/90s corporate music with surreal aesthetics",
         (0.1, 0.4), ["Macintosh Plus", "Saint Pepsi", "Blank Banshee"],
         "a vaporwave track with slowed-down smooth jazz samples, dreamy reverb, and nostalgic 90s aesthetics"),
    ]:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Synthwave", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"], defining_characteristics=[],
            typical_instruments=["analog synths", "drum machine"], production_style=sub[3],
            era_of_origin="2010s", parent_genres=["Synthwave"], sibling_genres=[],
            energy_range=sub[4], danceability_range=(0.3, 0.65), acousticness_range=(0.0, 0.15),
            famous_artists=sub[5], clap_descriptions=[sub[6]],
        ))

    genres.append(Genre(
        name="Footwork", id=_id(), parent="Electronic",
        aliases=["juke", "footwork/juke"], bpm_range=(155, 165),
        key_tendencies=["minor keys"],
        defining_characteristics=["rapid-fire samples", "polyrhythmic patterns", "chopped vocals",
                                  "battle-dance influence"],
        typical_instruments=["sampler", "TR-808", "vocal chops"],
        production_style="Rapid, sample-heavy production designed for competitive footwork dance",
        era_of_origin="2000s", parent_genres=["Ghetto House", "Juke"],
        sibling_genres=["Jungle", "UK Bass"],
        energy_range=(0.7, 0.95), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.1),
        famous_artists=["DJ Rashad", "RP Boo", "DJ Spinn", "Teklife"],
        clap_descriptions=[
            "a footwork track at 160 BPM with rapid-fire vocal chops, polyrhythmic 808 patterns, and frenetic energy",
        ],
    ))

    genres.append(Genre(
        name="UK Bass", id=_id(), parent="Electronic",
        aliases=["bass music"], bpm_range=(130, 145),
        key_tendencies=["minor keys"],
        defining_characteristics=["heavy bass", "UK influence", "genre-blending", "dark atmosphere"],
        typical_instruments=["sub-bass synth", "sampler"],
        production_style="Bass-heavy UK production blending elements from garage, grime, and dubstep",
        era_of_origin="2000s", parent_genres=["UK Garage", "Dubstep", "Grime"],
        sibling_genres=["Dubstep", "Grime"],
        energy_range=(0.5, 0.85), danceability_range=(0.5, 0.8), acousticness_range=(0.0, 0.1),
        famous_artists=["Burial", "Joy Orbison", "Pearson Sound"],
        clap_descriptions=[
            "a UK bass track at 140 BPM with deep sub-bass, shuffled beats, and dark atmospheric textures",
        ],
    ))

    genres.append(Genre(
        name="Grime", id=_id(), parent="Electronic",
        aliases=["grime music"], bpm_range=(138, 142),
        key_tendencies=["minor keys", "Phrygian mode"],
        defining_characteristics=["aggressive MC vocals", "icy synths", "square-wave bass",
                                  "8-bar patterns"],
        typical_instruments=["square wave synth", "Fruity Loops", "mic"],
        production_style="Raw, aggressive production with icy synths, square-wave bass, and MC-focused arrangements",
        era_of_origin="2000s", parent_genres=["UK Garage", "Jungle", "Hip-Hop/Rap"],
        sibling_genres=["UK Bass", "Dubstep"],
        energy_range=(0.6, 0.9), danceability_range=(0.5, 0.75), acousticness_range=(0.0, 0.05),
        famous_artists=["Wiley", "Dizzee Rascal", "Skepta", "Stormzy"],
        clap_descriptions=[
            "a grime instrumental at 140 BPM with icy square-wave synths, aggressive bass, and an 8-bar loop structure",
        ],
    ))

    genres.append(Genre(
        name="Hardstyle", id=_id(), parent="Electronic",
        aliases=["hardstyle music"], bpm_range=(140, 155),
        key_tendencies=["minor keys"],
        defining_characteristics=["reverse bass kick", "euphoric melodies", "distorted leads",
                                  "hard kicks", "anthem builds"],
        typical_instruments=["distorted kick synth", "supersaw", "reverse bass"],
        production_style="Hard-hitting production with distinctive reverse bass kicks and euphoric melodies",
        era_of_origin="2000s", parent_genres=["Hard Trance", "Hardcore Electronic"],
        sibling_genres=["Hard Trance", "Hardcore Electronic"],
        energy_range=(0.8, 1.0), danceability_range=(0.6, 0.85), acousticness_range=(0.0, 0.05),
        famous_artists=["Headhunterz", "D-Block & S-te-Fan", "Wildstylez"],
        clap_descriptions=[
            "a hardstyle track at 150 BPM with a punishing reverse bass kick, euphoric melodies, and anthem-like builds",
        ],
    ))

    genres.append(Genre(
        name="Eurodance", id=_id(), parent="Electronic",
        aliases=["euro dance", "eurobeat"], bpm_range=(130, 155),
        key_tendencies=["minor keys", "major keys"],
        defining_characteristics=["catchy hooks", "female vocals", "male rap verses",
                                  "synth riffs", "high energy"],
        typical_instruments=["synths", "drum machine", "sampler"],
        production_style="Catchy, hook-driven dance production with characteristic male/female vocal interplay",
        era_of_origin="1990s", parent_genres=["Electronic", "Pop"],
        sibling_genres=["Dance-Pop", "Trance"],
        energy_range=(0.7, 0.9), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.1),
        famous_artists=["2 Unlimited", "La Bouche", "Haddaway", "Snap!"],
        clap_descriptions=[
            "a eurodance track at 140 BPM with a catchy synth hook, female vocals, and high-energy dance beats",
        ],
    ))

    genres.append(Genre(
        name="Electronica", id=_id(), parent="Electronic",
        aliases=["art electronic"], bpm_range=(80, 140),
        key_tendencies=["varied"],
        defining_characteristics=["experimental textures", "diverse influences", "album-oriented",
                                  "home listening focus"],
        typical_instruments=["synthesizer", "sampler", "laptop"],
        production_style="Diverse, album-oriented electronic production bridging experimental and accessible",
        era_of_origin="1990s", parent_genres=["Electronic"],
        sibling_genres=["IDM", "Downtempo"],
        energy_range=(0.2, 0.7), danceability_range=(0.3, 0.7), acousticness_range=(0.05, 0.3),
        famous_artists=["Bjork", "Radiohead", "Four Tet", "Jon Hopkins"],
        clap_descriptions=[
            "an electronica track with textured layers of synthesis, organic samples, and an evolving arrangement",
        ],
    ))

    genres.append(Genre(
        name="2-Step", id=_id(), parent="Electronic",
        aliases=["2-step garage", "two-step"], bpm_range=(130, 140),
        key_tendencies=["minor keys"],
        defining_characteristics=["shuffled rhythm missing beat 2 and 4 kick", "R&B influence",
                                  "syncopated hi-hats", "smooth vocals"],
        typical_instruments=["sampler", "sub-bass", "R&B vocals"],
        production_style="Syncopated garage production with R&B influence and smooth, shuffled rhythms",
        era_of_origin="1990s", parent_genres=["UK Garage"],
        sibling_genres=["UK Garage", "Bassline"],
        energy_range=(0.5, 0.75), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.1),
        famous_artists=["MJ Cole", "Artful Dodger", "Wookie"],
        clap_descriptions=[
            "a 2-step garage track at 135 BPM with shuffled syncopated rhythms, smooth R&B vocals, and deep bass",
        ],
    ))

    genres.append(Genre(
        name="Bassline", id=_id(), parent="Electronic",
        aliases=["bassline house", "niche"], bpm_range=(135, 142),
        key_tendencies=["minor keys"],
        defining_characteristics=["heavy bass", "bouncy grooves", "UK garage influence",
                                  "rave stabs"],
        typical_instruments=["bass synth", "sampler", "rave stabs"],
        production_style="Bass-heavy UK production with bouncy rhythms and garage-influenced grooves",
        era_of_origin="2000s", parent_genres=["UK Garage", "Speed Garage"],
        sibling_genres=["UK Bass", "Bass House"],
        energy_range=(0.6, 0.85), danceability_range=(0.7, 0.9), acousticness_range=(0.0, 0.05),
        famous_artists=["Jamie Duggan", "DJ Q", "T2"],
        clap_descriptions=[
            "a bassline track at 138 BPM with a bouncy bass-heavy groove, rave stabs, and UK garage rhythms",
        ],
    ))

    # ======================================================================
    # HIP-HOP / RAP  (~50 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Hip-Hop/Rap", id=_id(), parent=None,
        aliases=["hip hop", "rap", "hip-hop"], bpm_range=(60, 170),
        key_tendencies=["minor keys", "Dorian mode", "blues scale"],
        defining_characteristics=["rhythmic vocals", "sampled beats", "808 bass",
                                  "drum breaks", "lyrical focus"],
        typical_instruments=["turntable", "MPC", "TR-808", "sampler", "microphone"],
        production_style="Sample-based or synthesized beat production with emphasis on rhythm and vocal delivery",
        era_of_origin="1970s", parent_genres=["Funk", "Soul", "Disco"],
        sibling_genres=["R&B/Soul", "Electronic"],
        energy_range=(0.4, 0.9), danceability_range=(0.5, 0.9), acousticness_range=(0.0, 0.3),
        famous_artists=["Tupac", "Notorious B.I.G.", "Jay-Z", "Kendrick Lamar", "Kanye West"],
        clap_descriptions=[
            "a hip-hop beat with hard-hitting drums, sampled loops, and a heavy 808 bass",
            "a rap instrumental with chopped soul samples, crisp snares, and a head-nodding groove",
        ],
    ))

    _hiphop_subs = [
        ("Boom Bap", ["boom-bap"], (80, 100), "Raw, sample-based production with dusty drums, vinyl crackle, and MPC swing",
         (0.5, 0.75), (0.5, 0.75), ["DJ Premier", "Pete Rock", "9th Wonder"], "1980s",
         "a classic boom bap beat at 90 BPM with dusty vinyl samples, hard-hitting MPC drums, and a jazzy loop"),
        ("Trap", ["trap music", "Atlanta trap"], (120, 170), "808-heavy production with rolling hi-hats, deep sub-bass, and dark atmospheres",
         (0.6, 0.9), (0.6, 0.85), ["Metro Boomin", "Zaytoven", "Southside", "Future"], "2000s",
         "a trap beat at 140 BPM with booming 808 bass, rapid-fire hi-hat rolls, and dark synth pads"),
        ("Melodic Trap", ["melodic rap"], (130, 160), "Emotional trap production with lush melodies, auto-tuned vocals, and atmospheric pads",
         (0.5, 0.8), (0.5, 0.8), ["Gunna", "Lil Baby", "Young Thug"], "2010s",
         "a melodic trap beat at 145 BPM with lush guitar loops, atmospheric pads, and emotional 808 patterns"),
        ("Drill", ["UK drill", "Chicago drill"], (135, 145), "Dark, menacing trap variant with sliding 808 bass, minor key melodies, and aggressive bounce",
         (0.6, 0.85), (0.5, 0.75), ["Pop Smoke", "Chief Keef", "Headie One"], "2010s",
         "a drill beat at 140 BPM with sliding 808 bass, dark piano chords, and aggressive bouncing hi-hats"),
        ("UK Drill", ["British drill"], (138, 142), "London variant of drill with distinctive sliding bass, dark melodies, and unique rhythmic patterns",
         (0.6, 0.85), (0.5, 0.75), ["Pop Smoke", "Headie One", "Central Cee"], "2010s",
         "a UK drill beat at 140 BPM with dark piano melodies, sliding 808 bass, and syncopated drill patterns"),
        ("Brooklyn Drill", ["NY drill"], (138, 145), "New York variant of drill blending UK drill elements with NYC rap energy",
         (0.6, 0.9), (0.5, 0.75), ["Pop Smoke", "Fivio Foreign", "Sheff G"], "2010s",
         "a Brooklyn drill beat at 142 BPM with heavy 808 slides, dark piano, and aggressive NYC energy"),
        ("Lo-fi Hip-Hop", ["lo-fi hip hop", "lofi", "lo-fi beats"], (70, 90), "Nostalgic, mellow production with vinyl crackle, tape hiss, and jazzy samples",
         (0.1, 0.35), (0.4, 0.6), ["Nujabes", "J Dilla", "Tomppabeats"], "2000s",
         "a lo-fi hip-hop beat at 78 BPM with vinyl crackle, mellow jazz piano, and a tape-saturated drum loop"),
        ("Cloud Rap", ["cloud", "cloud-rap"], (60, 130), "Dreamy, ethereal rap production with spacey synths, heavy reverb, and slow tempos",
         (0.3, 0.6), (0.4, 0.65), ["Lil B", "Yung Lean", "A$AP Rocky"], "2010s",
         "a cloud rap beat at 75 BPM with dreamy reverb-drenched synths, slow 808 bass, and ethereal atmosphere"),
        ("Conscious Hip-Hop", ["conscious rap", "woke rap"], (80, 110), "Lyric-focused production with socially conscious themes and jazz/soul samples",
         (0.4, 0.7), (0.5, 0.7), ["Kendrick Lamar", "J. Cole", "Common", "Talib Kweli"], "1980s",
         "a conscious hip-hop beat at 95 BPM with soulful samples, crisp drums, and a thoughtful groove"),
        ("Gangsta Rap", ["gangsta"], (80, 110), "Hard-hitting production with aggressive energy, G-funk synths, and street-oriented themes",
         (0.6, 0.85), (0.5, 0.75), ["Dr. Dre", "Snoop Dogg", "Ice Cube", "50 Cent"], "1980s",
         "a gangsta rap beat at 95 BPM with menacing synths, hard-hitting drums, and a heavy bass groove"),
        ("G-Funk", ["g funk"], (90, 108), "Smooth West Coast production with whining synths, deep bass, and Parliament-Funkadelic influence",
         (0.5, 0.7), (0.6, 0.8), ["Dr. Dre", "Warren G", "DJ Quik", "Snoop Dogg"], "1990s",
         "a G-funk beat at 98 BPM with whining Moog synths, deep funky bass, and a smooth West Coast groove"),
        ("Crunk", ["crunk music"], (70, 80), "High-energy Southern production with heavy 808 bass, chant vocals, and aggressive energy",
         (0.7, 0.9), (0.7, 0.85), ["Lil Jon", "Three 6 Mafia", "Ying Yang Twins"], "2000s",
         "a crunk beat at 75 BPM with booming 808 kicks, aggressive chant vocals, and a heavy Southern bounce"),
        ("Snap Music", ["snap"], (75, 85), "Minimalist Southern production with finger snaps, catchy hooks, and a bouncy feel",
         (0.5, 0.7), (0.7, 0.85), ["D4L", "Dem Franchize Boyz"], "2000s",
         "a snap music beat at 78 BPM with finger-snap percussion, a catchy hook, and a bouncy Atlanta groove"),
        ("Emo Rap", ["emo-rap", "SoundCloud rap"], (130, 170), "Emotionally raw production blending emo/punk elements with trap beats",
         (0.5, 0.8), (0.4, 0.65), ["Juice WRLD", "Lil Peep", "XXXTentacion"], "2010s",
         "an emo rap beat at 150 BPM with minor-key guitar melodies, trap hi-hats, and emotionally raw atmosphere"),
        ("Jazz Rap", ["jazz-rap", "jazz hip-hop"], (80, 105), "Jazz-influenced hip-hop with complex harmony, live instrumentation, and sophisticated samples",
         (0.3, 0.6), (0.5, 0.7), ["A Tribe Called Quest", "Guru", "The Roots"], "1990s",
         "a jazz rap beat at 90 BPM with live upright bass, jazzy piano chords, and sophisticated drum programming"),
        ("Abstract Hip-Hop", ["abstract rap", "experimental hip-hop"], (70, 120), "Avant-garde production with unconventional structures, dense lyricism, and experimental beats",
         (0.3, 0.6), (0.3, 0.6), ["MF DOOM", "Aesop Rock", "Madlib"], "1990s",
         "an abstract hip-hop beat at 85 BPM with unconventional samples, lo-fi textures, and experimental arrangement"),
        ("Horrorcore", ["horror rap"], (70, 110), "Dark, horror-themed production with eerie samples, minor keys, and menacing atmosphere",
         (0.5, 0.8), (0.4, 0.65), ["Three 6 Mafia", "Gravediggaz", "Brotha Lynch Hung"], "1990s",
         "a horrorcore beat at 80 BPM with eerie synth samples, dark piano, and a menacing horror-film atmosphere"),
        ("Chopped and Screwed", ["chopped & screwed", "screwed"], (30, 70), "Slowed-down, pitch-shifted production creating a hypnotic, syrupy feel",
         (0.2, 0.45), (0.3, 0.5), ["DJ Screw", "Swisha House", "OG Ron C"], "1990s",
         "a chopped and screwed beat at 45 BPM with slowed-down vocals, pitched-down bass, and syrupy, hypnotic groove"),
        ("Phonk", ["phonk music", "drift phonk"], (120, 145), "Memphis-inspired production with heavy bass, cowbell loops, distorted vocals, and dark aesthetics",
         (0.6, 0.85), (0.6, 0.8), ["DJ Smokey", "Soudiere", "Kordhell"], "2010s",
         "a phonk beat at 130 BPM with Memphis-style cowbell loops, distorted vocal samples, and heavy bass"),
        ("Southern Hip-Hop", ["dirty south", "southern rap"], (65, 100), "Bass-heavy production with bouncy rhythms, regional slang, and diverse Southern styles",
         (0.5, 0.8), (0.6, 0.85), ["OutKast", "UGK", "Lil Wayne", "T.I."], "1990s",
         "a Southern hip-hop beat at 85 BPM with bouncy bass, snapping percussion, and a warm Southern groove"),
        ("East Coast Hip-Hop", ["east coast", "NYC hip-hop"], (80, 100), "Hard-hitting, lyric-focused production with boom-bap roots and jazz/soul samples",
         (0.5, 0.75), (0.5, 0.7), ["Nas", "Jay-Z", "Wu-Tang Clan", "Notorious B.I.G."], "1980s",
         "an East Coast hip-hop beat at 92 BPM with hard-hitting drums, jazz piano samples, and a gritty New York groove"),
        ("West Coast Hip-Hop", ["west coast"], (85, 105), "Smooth, funk-influenced production with deep bass, layered synths, and laid-back grooves",
         (0.5, 0.75), (0.6, 0.8), ["Dr. Dre", "Snoop Dogg", "Ice Cube", "Kendrick Lamar"], "1980s",
         "a West Coast hip-hop beat at 95 BPM with smooth G-funk synths, deep bass, and a laid-back California groove"),
        ("Midwest Hip-Hop", ["midwest rap"], (80, 110), "Diverse production styles from the Midwest blending multiple regional influences",
         (0.5, 0.75), (0.5, 0.75), ["Kanye West", "Eminem", "Bone Thugs-N-Harmony", "Tech N9ne"], "1990s",
         "a Midwest hip-hop beat at 90 BPM with soulful vocal chops, crisp drums, and a versatile groove"),
        ("Underground Hip-Hop", ["underground rap"], (70, 110), "Independent, uncompromising production with raw aesthetics and experimental elements",
         (0.4, 0.7), (0.4, 0.7), ["MF DOOM", "El-P", "Aesop Rock", "Atmosphere"], "1990s",
         "an underground hip-hop beat at 88 BPM with raw, unpolished drums, obscure samples, and experimental textures"),
        ("Instrumental Hip-Hop", ["instrumental rap", "beat tape"], (70, 110), "Beat-focused production without vocals, emphasizing groove and sonic texture",
         (0.3, 0.6), (0.5, 0.7), ["J Dilla", "Madlib", "Flying Lotus", "Knxwledge"], "1990s",
         "an instrumental hip-hop beat at 82 BPM with chopped soul samples, MPC-style drums, and a head-nodding groove"),
        ("Turntablism", ["scratching", "DJ battle"], (80, 120), "Turntable-focused production emphasizing scratching, beat juggling, and vinyl manipulation",
         (0.5, 0.8), (0.5, 0.75), ["DJ Qbert", "DJ Shadow", "Mix Master Mike", "Cut Chemist"], "1980s",
         "a turntablist composition with intricate scratch patterns, beat juggling, and vinyl manipulation techniques"),
        ("Old School Hip-Hop", ["old school", "old-school rap"], (90, 115), "Early hip-hop production with simple drum machines, funk samples, and party-oriented energy",
         (0.5, 0.75), (0.6, 0.8), ["Grandmaster Flash", "Run-DMC", "LL Cool J"], "1970s",
         "an old school hip-hop beat at 100 BPM with a simple drum machine pattern, funk bass sample, and party energy"),
        ("Alternative Hip-Hop", ["alt hip-hop"], (70, 120), "Genre-blending production incorporating rock, electronic, and experimental elements",
         (0.3, 0.7), (0.4, 0.7), ["Tyler, the Creator", "Kid Cudi", "Anderson .Paak"], "1990s",
         "an alternative hip-hop beat at 95 BPM with eclectic instrumentation, live drums, and genre-blending production"),
        ("Industrial Hip-Hop", ["industrial rap"], (80, 140), "Harsh, noise-influenced production with distorted elements and aggressive textures",
         (0.6, 0.9), (0.4, 0.65), ["Death Grips", "clipping.", "JPEGMAFIA"], "2010s",
         "an industrial hip-hop beat at 120 BPM with distorted noise, harsh textures, and aggressive, abrasive production"),
        ("Political Hip-Hop", ["political rap"], (80, 110), "Message-driven production with socially conscious themes and powerful vocal delivery",
         (0.5, 0.75), (0.5, 0.7), ["Public Enemy", "Dead Prez", "Immortal Technique"], "1980s",
         "a political hip-hop beat at 95 BPM with hard-hitting drums, powerful vocal samples, and socially charged energy"),
    ]

    for sub in _hiphop_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Hip-Hop/Rap", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "blues scale"],
            defining_characteristics=[], typical_instruments=["MPC", "TR-808", "sampler"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Hip-Hop/Rap"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.2),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # R&B / SOUL  (~30 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="R&B/Soul", id=_id(), parent=None,
        aliases=["R&B", "soul", "rhythm and blues"], bpm_range=(60, 130),
        key_tendencies=["minor keys", "major 7th chords", "blues scale", "pentatonic"],
        defining_characteristics=["soulful vocals", "groove-oriented", "emotional delivery",
                                  "rich harmony", "call-and-response"],
        typical_instruments=["vocals", "Rhodes", "bass guitar", "drums", "horns"],
        production_style="Vocal-centered production with rich harmony, groove, and emotional depth",
        era_of_origin="1940s", parent_genres=["Blues", "Gospel"],
        sibling_genres=["Hip-Hop/Rap", "Pop", "Funk"],
        energy_range=(0.3, 0.75), danceability_range=(0.5, 0.85), acousticness_range=(0.1, 0.6),
        famous_artists=["Stevie Wonder", "Aretha Franklin", "Marvin Gaye", "Whitney Houston", "Beyonce"],
        clap_descriptions=[
            "a soulful R&B track with warm Rhodes chords, a groovy bassline, and emotive vocal harmonies",
            "a smooth R&B production with lush vocal layers, live drums, and rich jazz-influenced harmony",
        ],
    ))

    _rnb_subs = [
        ("Contemporary R&B", ["modern R&B"], (70, 120), "Polished modern production blending R&B vocals with electronic and hip-hop elements",
         (0.3, 0.65), (0.5, 0.8), ["The Weeknd", "SZA", "Frank Ocean", "H.E.R."], "1980s",
         "a contemporary R&B track at 95 BPM with smooth synth pads, programmed drums, and airy vocal production"),
        ("Neo-Soul", ["neo soul"], (70, 105), "Organic, jazz-influenced soul with live instrumentation and socially conscious lyrics",
         (0.3, 0.6), (0.5, 0.75), ["Erykah Badu", "D'Angelo", "Lauryn Hill", "Jill Scott"], "1990s",
         "a neo-soul track at 85 BPM with live Rhodes, warm analog bass, and a jazzy, organic groove"),
        ("Classic Soul", ["soul music", "60s soul"], (90, 130), "Rich, full-band production with powerful vocals, horn sections, and gospel influence",
         (0.5, 0.8), (0.6, 0.85), ["Otis Redding", "Sam Cooke", "Aretha Franklin"], "1960s",
         "a classic soul track with a powerful vocal performance, a full horn section, and a driving rhythm section"),
        ("Motown", ["Motown sound"], (100, 130), "Polished, hit-making production with catchy melodies, tambourine, and tight arrangements",
         (0.5, 0.8), (0.7, 0.9), ["The Supremes", "Stevie Wonder", "Marvin Gaye", "The Temptations"], "1960s",
         "a Motown track with a driving tambourine beat, tight horn arrangements, and an infectious melody"),
        ("Northern Soul", ["northern"], (110, 140), "Uptempo, dance-oriented soul with stomping beats and rare record aesthetics",
         (0.6, 0.85), (0.75, 0.95), ["Frank Wilson", "Gloria Jones"], "1960s",
         "a Northern soul track with a stomping uptempo beat, powerful vocals, and an energetic dance groove"),
        ("Funk", ["funk music"], (90, 120), "Groove-driven production with syncopated bass, rhythmic guitar, and horn sections",
         (0.6, 0.85), (0.7, 0.9), ["James Brown", "Parliament-Funkadelic", "Sly and the Family Stone", "Prince"], "1960s",
         "a funk track at 105 BPM with a syncopated slap bass, wah-wah guitar, and a tight horn section"),
        ("P-Funk", ["Parliament-Funkadelic"], (90, 115), "Cosmic, psychedelic funk with expansive arrangements, synths, and Afro-futurist themes",
         (0.6, 0.85), (0.7, 0.9), ["Parliament", "Funkadelic", "George Clinton", "Bootsy Collins"], "1970s",
         "a P-funk track with a cosmic synth bass, psychedelic guitar, and an expansive, Afro-futurist arrangement"),
        ("Minneapolis Sound", ["Minneapolis funk"], (100, 130), "Prince-inspired synth-funk with drum machines, synths, and genre-blending production",
         (0.6, 0.85), (0.7, 0.9), ["Prince", "The Time", "Janet Jackson", "Jimmy Jam & Terry Lewis"], "1980s",
         "a Minneapolis sound track with a LinnDrum beat, synth bass, and Prince-influenced funk guitar"),
        ("Quiet Storm", ["quiet-storm"], (65, 90), "Smooth, romantic R&B with gentle rhythms, lush arrangements, and intimate vocals",
         (0.2, 0.45), (0.4, 0.6), ["Luther Vandross", "Anita Baker", "Sade"], "1970s",
         "a quiet storm ballad at 72 BPM with smooth Rhodes chords, gentle drums, and intimate vocal delivery"),
        ("New Jack Swing", ["new jack", "NJS"], (95, 120), "Hip-hop-influenced R&B with programmed beats, heavy bass, and pop hooks",
         (0.6, 0.8), (0.7, 0.85), ["Teddy Riley", "Bobby Brown", "Bell Biv DeVoe"], "1980s",
         "a new jack swing track at 110 BPM with a programmed drum machine beat, heavy bass, and R&B vocal hooks"),
        ("Alternative R&B", ["alt-R&B", "indie R&B"], (60, 110), "Experimental, genre-blending R&B with electronic production and atmospheric textures",
         (0.2, 0.55), (0.4, 0.7), ["Frank Ocean", "The Weeknd", "FKA Twigs", "Kelela"], "2010s",
         "an alternative R&B track at 80 BPM with ethereal synths, glitchy percussion, and atmospheric vocal processing"),
        ("PBR&B", ["hipster R&B"], (60, 100), "Lo-fi, indie-influenced R&B with hazy production and understated vocals",
         (0.2, 0.5), (0.4, 0.65), ["How to Dress Well", "Blood Orange", "Toro y Moi"], "2010s",
         "a PBR&B track at 75 BPM with hazy lo-fi textures, understated vocals, and an indie-influenced groove"),
        ("Gospel", ["gospel music"], (70, 140), "Spiritual, church-rooted production with powerful group vocals, organ, and call-and-response",
         (0.4, 0.9), (0.5, 0.85), ["Mahalia Jackson", "Kirk Franklin", "The Clark Sisters"], "1920s",
         "a gospel track with powerful choir vocals, Hammond organ, and an uplifting, spiritual energy"),
        ("Doo-Wop", ["doo wop"], (70, 130), "Vocal group harmony music with tight harmonies, minimal instrumentation, and romantic themes",
         (0.4, 0.65), (0.5, 0.7), ["The Platters", "The Drifters", "Frankie Lymon"], "1950s",
         "a doo-wop track with tight four-part vocal harmonies, a simple bass pattern, and 1950s charm"),
        ("Blue-Eyed Soul", ["blue eyed soul"], (80, 120), "Soul music performed by white artists with authentic soul vocal style and production",
         (0.4, 0.75), (0.5, 0.8), ["Hall & Oates", "Amy Winehouse", "Adele", "Sam Smith"], "1960s",
         "a blue-eyed soul track with a powerful vocal performance, soulful chord progressions, and retro production"),
        ("Psychedelic Soul", ["psych soul"], (90, 125), "Soul with psychedelic elements including phaser, wah-wah, and experimental production",
         (0.5, 0.8), (0.6, 0.8), ["Sly and the Family Stone", "The Temptations", "Shuggie Otis"], "1960s",
         "a psychedelic soul track with phaser-drenched guitars, wah-wah bass, and a trippy, soulful groove"),
        ("Deep Funk", ["rare funk"], (90, 120), "Raw, stripped-down funk with heavy drums, simple riffs, and a gritty feel",
         (0.6, 0.8), (0.7, 0.9), ["The Meters", "Sharon Jones", "Lee Fields"], "1960s",
         "a deep funk track with a raw, heavy drum break, gritty guitar riff, and stripped-down groove"),
        ("Boogie", ["boogie music", "post-disco"], (110, 130), "Danceable, synth-influenced funk-soul bridging disco and electro",
         (0.5, 0.8), (0.7, 0.9), ["Zapp", "Mtume", "DeBarge"], "1980s",
         "a boogie track at 118 BPM with a synth bass groove, drum machine beat, and dancefloor-ready energy"),
        ("Modern Funk", ["modern-funk"], (100, 125), "Contemporary production reviving 1980s boogie and funk aesthetics with modern tools",
         (0.5, 0.75), (0.7, 0.85), ["Dam-Funk", "Thundercat", "Channel Tres"], "2010s",
         "a modern funk track at 112 BPM with retro synth bass, vocoder harmonies, and a drum machine groove"),
        ("Afrobeats", ["Afrobeats music", "Afro-pop"], (95, 120), "West African pop production with infectious rhythms, log drums, and dance-oriented grooves",
         (0.5, 0.8), (0.7, 0.9), ["Wizkid", "Burna Boy", "Davido", "Tiwa Savage"], "2000s",
         "an Afrobeats track at 108 BPM with infectious log drum patterns, shaker percussion, and a dance-ready groove"),
        ("Afroswing", ["Afro swing"], (90, 115), "UK-born blend of Afrobeats, dancehall, and R&B with a laid-back, melodic feel",
         (0.5, 0.75), (0.7, 0.85), ["J Hus", "Not3s", "Yxng Bane"], "2010s",
         "an Afroswing track at 100 BPM with Caribbean-influenced rhythms, melodic hooks, and Afrobeat percussion"),
    ]

    for sub in _rnb_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="R&B/Soul", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "major 7th chords"],
            defining_characteristics=[], typical_instruments=["vocals", "Rhodes", "bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["R&B/Soul"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.1, 0.5),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # POP  (~40 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Pop", id=_id(), parent=None,
        aliases=["pop music"], bpm_range=(80, 140),
        key_tendencies=["major keys", "minor keys", "I-V-vi-IV"],
        defining_characteristics=["catchy melodies", "verse-chorus structure", "hook-driven",
                                  "polished production", "wide appeal"],
        typical_instruments=["vocals", "synthesizer", "guitar", "drums", "bass"],
        production_style="Highly polished, hook-driven production designed for maximum accessibility",
        era_of_origin="1950s", parent_genres=["Rock and Roll", "Tin Pan Alley"],
        sibling_genres=["R&B/Soul", "Rock", "Electronic"],
        energy_range=(0.4, 0.85), danceability_range=(0.5, 0.85), acousticness_range=(0.05, 0.4),
        famous_artists=["Michael Jackson", "Madonna", "Taylor Swift", "Beyonce", "The Beatles"],
        clap_descriptions=[
            "a polished pop track with a catchy vocal melody, crisp production, and an infectious chorus hook",
        ],
    ))

    _pop_subs = [
        ("Synth-Pop", ["synthpop", "synth pop"], (110, 135), "Synthesizer-driven pop with electronic drums, catchy hooks, and a futuristic feel",
         (0.5, 0.8), (0.6, 0.85), ["Depeche Mode", "Pet Shop Boys", "New Order", "Chvrches"], "1980s",
         "a synth-pop track at 120 BPM with pulsing analog synths, electronic drums, and a catchy vocal hook"),
        ("Electropop", ["electro-pop"], (115, 135), "Pop blended with electronic production, bright synths, and processed vocals",
         (0.6, 0.85), (0.6, 0.85), ["Lady Gaga", "Robyn", "Charli XCX", "Grimes"], "2000s",
         "an electropop track at 125 BPM with bright synth leads, processed vocals, and a danceable electronic beat"),
        ("Indie Pop", ["indie-pop"], (100, 140), "Lo-fi or DIY-ethos pop with jangly guitars, sincere vocals, and a handcrafted feel",
         (0.3, 0.65), (0.4, 0.7), ["Belle and Sebastian", "Vampire Weekend", "The Smiths"], "1980s",
         "an indie pop track with jangly guitar arpeggios, sincere vocals, and a lo-fi, handcrafted feel"),
        ("Dream Pop", ["dreampop"], (80, 120), "Hazy, reverb-drenched pop with ethereal vocals, shimmering guitars, and atmospheric textures",
         (0.2, 0.5), (0.3, 0.55), ["Cocteau Twins", "Beach House", "Mazzy Star"], "1980s",
         "a dream pop track with shimmering reverb-drenched guitars, ethereal vocals, and hazy atmospheric textures"),
        ("Chamber Pop", ["chamber-pop"], (80, 120), "Ornate pop with orchestral arrangements, strings, and rich harmonic complexity",
         (0.3, 0.6), (0.3, 0.6), ["Sufjan Stevens", "Arcade Fire", "Andrew Bird"], "1990s",
         "a chamber pop arrangement with string quartet, French horn, and a richly orchestrated vocal arrangement"),
        ("Art Pop", ["art-pop"], (80, 140), "Experimental, boundary-pushing pop drawing from avant-garde and fine art influences",
         (0.3, 0.7), (0.4, 0.7), ["Kate Bush", "Bjork", "St. Vincent", "David Bowie"], "1970s",
         "an art pop track with experimental synth textures, unconventional song structure, and avant-garde production"),
        ("K-Pop", ["Korean pop"], (90, 140), "Korean pop production with precise choreography focus, genre-blending, and high-gloss production",
         (0.6, 0.9), (0.6, 0.9), ["BTS", "BLACKPINK", "EXO", "TWICE"], "1990s",
         "a K-pop track with a high-energy beat, polished vocal production, and a genre-blending hook section"),
        ("J-Pop", ["Japanese pop"], (100, 145), "Japanese pop with diverse stylistic range, catchy melodies, and anime/media tie-ins",
         (0.5, 0.8), (0.5, 0.8), ["Hikaru Utada", "Kenshi Yonezu", "Perfume"], "1990s",
         "a J-pop track with a bright, catchy melody, electronic production, and energetic arrangement"),
        ("Latin Pop", ["latin-pop"], (85, 120), "Spanish-language pop blending Latin rhythms with contemporary pop production",
         (0.5, 0.8), (0.6, 0.85), ["Shakira", "Ricky Martin", "Enrique Iglesias", "Bad Bunny"], "1980s",
         "a Latin pop track at 100 BPM with Latin percussion, Spanish vocals, and a catchy pop melody"),
        ("Euro Pop", ["Europop"], (120, 140), "European pop with dance-oriented beats, catchy hooks, and polished production",
         (0.6, 0.85), (0.7, 0.9), ["ABBA", "Ace of Base", "Aqua"], "1970s",
         "a Euro pop track at 130 BPM with a catchy dance beat, bright synths, and an infectious hook"),
        ("Dance-Pop", ["dance pop"], (110, 135), "Pop designed for dancing with four-on-the-floor beats and synth-driven production",
         (0.6, 0.85), (0.7, 0.9), ["Madonna", "Dua Lipa", "Kylie Minogue"], "1980s",
         "a dance-pop track at 120 BPM with a four-on-the-floor beat, synth hooks, and an infectious chorus"),
        ("Bubblegum Pop", ["bubblegum"], (100, 130), "Ultra-catchy, simple pop aimed at young audiences with bright, cheerful production",
         (0.6, 0.8), (0.6, 0.8), ["The Archies", "Hanson", "Britney Spears"], "1960s",
         "a bubblegum pop track with an ultra-catchy melody, bright production, and cheerful, youthful energy"),
        ("Power Pop", ["power-pop"], (110, 140), "Guitar-driven pop with strong melodies, crunchy guitars, and big vocal harmonies",
         (0.6, 0.8), (0.5, 0.75), ["Cheap Trick", "Big Star", "Weezer"], "1970s",
         "a power pop track with crunchy guitar chords, a strong vocal melody, and driving rock drums"),
        ("Baroque Pop", ["baroque-pop"], (80, 120), "Ornate pop with classical instrumentation, complex arrangements, and lush orchestration",
         (0.3, 0.6), (0.3, 0.6), ["The Beach Boys", "The Beatles", "Lana Del Rey"], "1960s",
         "a baroque pop arrangement with harpsichord, strings, and a lush, orchestral vocal production"),
        ("Noise Pop", ["noise-pop"], (110, 140), "Pop songs buried under layers of guitar feedback, distortion, and sonic texture",
         (0.5, 0.8), (0.4, 0.65), ["Jesus and Mary Chain", "Yo La Tengo", "My Bloody Valentine"], "1980s",
         "a noise pop track with distorted guitar layers, a buried pop melody, and walls of feedback"),
        ("Twee Pop", ["twee"], (100, 135), "Gentle, lo-fi indie pop with childlike simplicity and innocent aesthetics",
         (0.2, 0.5), (0.4, 0.65), ["Beat Happening", "Belle and Sebastian", "Camera Obscura"], "1980s",
         "a twee pop track with a gentle strummed guitar, soft vocals, and a lo-fi, innocent charm"),
        ("Hyperpop", ["hyper-pop", "hyper pop"], (130, 170), "Maximalist, glitchy pop with pitch-shifted vocals, extreme processing, and chaotic energy",
         (0.7, 1.0), (0.5, 0.8), ["100 gecs", "SOPHIE", "Charli XCX", "AG Cook"], "2010s",
         "a hyperpop track at 155 BPM with pitch-shifted vocals, glitchy bass, and maximalist chaotic production"),
        ("PC Music", ["PC music"], (120, 150), "Hyper-glossy, deconstructed pop with extreme vocal processing and ironic commercial aesthetics",
         (0.6, 0.9), (0.5, 0.8), ["SOPHIE", "AG Cook", "Hannah Diamond"], "2010s",
         "a PC Music track with hyper-glossy synths, extreme vocal pitch-shifting, and ironic pop construction"),
        ("Teen Pop", ["teen-pop"], (100, 130), "Market-targeted pop for teenagers with catchy hooks and relatable themes",
         (0.5, 0.8), (0.6, 0.8), ["Britney Spears", "Justin Bieber", "Olivia Rodrigo"], "1990s",
         "a teen pop track with a catchy vocal hook, polished pop production, and youthful energy"),
        ("Adult Contemporary", ["AC", "soft rock"], (70, 115), "Smooth, radio-friendly pop/soft rock for adult audiences with polished production",
         (0.2, 0.5), (0.3, 0.6), ["Michael Buble", "Celine Dion", "John Legend"], "1960s",
         "an adult contemporary ballad with smooth vocals, gentle piano, and polished, radio-friendly production"),
        ("City Pop", ["city-pop", "Japanese city pop"], (95, 128), "Japanese urban pop from the 80s with jazz-funk influence, lush production, and summer vibes",
         (0.4, 0.7), (0.6, 0.8), ["Tatsuro Yamashita", "Mariya Takeuchi", "Anri"], "1980s",
         "a city pop track at 108 BPM with jazz-influenced chords, bright synths, and a breezy summer groove"),
        ("Sophisti-Pop", ["sophistipop"], (90, 120), "Polished, jazz-influenced pop with sophisticated harmony and slick production",
         (0.4, 0.65), (0.5, 0.75), ["Sade", "Swing Out Sister", "Everything but the Girl"], "1980s",
         "a sophisti-pop track with jazz-influenced chords, a smooth vocal, and polished 80s production"),
        ("Jangle Pop", ["jangle-pop"], (110, 140), "Guitar-driven pop with bright, ringing Rickenbacker guitars and melodic songwriting",
         (0.4, 0.7), (0.5, 0.7), ["R.E.M.", "The Byrds", "The Smiths"], "1980s",
         "a jangle pop track with bright Rickenbacker guitar arpeggios, melodic vocals, and a driving beat"),
        ("C-Pop", ["Chinese pop", "Mandopop"], (80, 130), "Chinese-language pop with diverse styles blending Western and Chinese musical elements",
         (0.4, 0.7), (0.5, 0.75), ["Jay Chou", "Eason Chan", "JJ Lin"], "1990s",
         "a C-pop ballad with Mandarin vocals, lush string arrangement, and polished pop production"),
        ("Cantopop", ["Cantonese pop"], (80, 130), "Cantonese-language pop from Hong Kong with dramatic ballads and polished production",
         (0.3, 0.7), (0.4, 0.7), ["Leslie Cheung", "Anita Mui", "Eason Chan"], "1970s",
         "a Cantopop ballad with dramatic Cantonese vocals, orchestral backing, and emotional arrangement"),
        ("Sunshine Pop", ["sunshine-pop"], (100, 130), "Bright, cheerful pop with lush vocal harmonies and sunny California vibes",
         (0.4, 0.7), (0.5, 0.7), ["The Beach Boys", "The Mamas and the Papas", "The 5th Dimension"], "1960s",
         "a sunshine pop track with lush multi-part vocal harmonies, bright acoustic guitar, and cheerful arrangement"),
    ]

    for sub in _pop_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Pop", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "minor keys"],
            defining_characteristics=[], typical_instruments=["vocals", "synths", "guitar"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Pop"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.05, 0.35),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # ROCK  (~60 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Rock", id=_id(), parent=None,
        aliases=["rock music"], bpm_range=(70, 200),
        key_tendencies=["minor keys", "major keys", "blues scale", "power chords"],
        defining_characteristics=["electric guitar", "bass guitar", "drums", "verse-chorus structure",
                                  "guitar solos", "band format"],
        typical_instruments=["electric guitar", "bass guitar", "drum kit", "vocals"],
        production_style="Band-centric production emphasizing live performance energy and guitar-driven arrangements",
        era_of_origin="1950s", parent_genres=["Blues", "Country", "Rock and Roll"],
        sibling_genres=["Pop", "Blues"],
        energy_range=(0.5, 0.95), danceability_range=(0.3, 0.7), acousticness_range=(0.05, 0.4),
        famous_artists=["The Beatles", "Led Zeppelin", "Pink Floyd", "Nirvana", "Radiohead"],
        clap_descriptions=[
            "a rock track with driving electric guitar riffs, a powerful drum beat, and a full band arrangement",
        ],
    ))

    _rock_subs = [
        ("Classic Rock", ["classic"], (100, 140), "Guitar-driven rock from the 60s-80s with iconic riffs, solos, and anthemic songwriting",
         (0.5, 0.85), (0.4, 0.7), ["Led Zeppelin", "The Rolling Stones", "The Who", "AC/DC"], "1960s",
         "a classic rock track with iconic guitar riffs, a powerful drum groove, and an anthemic chorus"),
        ("Hard Rock", ["hard-rock"], (110, 150), "Aggressive, loud rock with heavy distortion, powerful vocals, and big riffs",
         (0.7, 0.95), (0.4, 0.65), ["AC/DC", "Guns N' Roses", "Aerosmith", "Deep Purple"], "1970s",
         "a hard rock track with heavy distorted guitar riffs, a pounding drum beat, and powerful rock vocals"),
        ("Punk Rock", ["punk"], (150, 200), "Fast, stripped-down rock with short songs, simple chords, and rebellious energy",
         (0.8, 1.0), (0.4, 0.65), ["Ramones", "Sex Pistols", "The Clash", "Dead Kennedys"], "1970s",
         "a punk rock track at 180 BPM with fast power chords, aggressive vocals, and raw three-chord energy"),
        ("Post-Punk", ["post punk"], (100, 150), "Angular, atmospheric rock with bass-driven arrangements and dark, experimental aesthetics",
         (0.4, 0.75), (0.4, 0.65), ["Joy Division", "Siouxsie and the Banshees", "Bauhaus", "The Cure"], "1970s",
         "a post-punk track with angular guitar lines, a driving bass riff, and dark, atmospheric production"),
        ("Pop-Punk", ["pop punk"], (140, 190), "Catchy, high-energy punk with melodic hooks and youthful themes",
         (0.7, 0.95), (0.5, 0.7), ["Green Day", "Blink-182", "Paramore", "Sum 41"], "1990s",
         "a pop-punk track at 170 BPM with fast power chords, a catchy vocal melody, and high-energy drums"),
        ("Hardcore Punk", ["hardcore"], (140, 220), "Extremely fast, aggressive punk with shouted vocals and intense energy",
         (0.9, 1.0), (0.3, 0.55), ["Black Flag", "Minor Threat", "Bad Brains"], "1980s",
         "a hardcore punk track at 200 BPM with blazing fast drums, screamed vocals, and aggressive guitar distortion"),
        ("Heavy Metal", ["metal"], (100, 160), "Heavy, distorted guitar-driven music with complex riffs, solos, and powerful vocals",
         (0.7, 0.95), (0.3, 0.6), ["Black Sabbath", "Iron Maiden", "Judas Priest", "Metallica"], "1970s",
         "a heavy metal track with heavy guitar riffs, a galloping drum beat, and a powerful vocal performance"),
        ("Thrash Metal", ["thrash"], (140, 220), "Fast, aggressive metal with complex riffs, rapid drumming, and intense energy",
         (0.85, 1.0), (0.3, 0.55), ["Metallica", "Slayer", "Megadeth", "Anthrax"], "1980s",
         "a thrash metal track at 200 BPM with blazing fast riffs, double-bass drumming, and aggressive vocals"),
        ("Death Metal", ["death"], (120, 200), "Extreme metal with growled vocals, blast beats, and heavily distorted, tuned-down guitars",
         (0.85, 1.0), (0.2, 0.45), ["Death", "Cannibal Corpse", "Morbid Angel"], "1980s",
         "a death metal track with guttural growled vocals, blast beat drums, and brutally heavy down-tuned guitars"),
        ("Black Metal", ["black"], (120, 220), "Extreme metal with shrieked vocals, tremolo picking, blast beats, and lo-fi production",
         (0.8, 1.0), (0.2, 0.4), ["Mayhem", "Burzum", "Emperor", "Darkthrone"], "1980s",
         "a black metal track with shrieked vocals, tremolo-picked guitars, blast beat drums, and raw lo-fi production"),
        ("Doom Metal", ["doom"], (40, 80), "Slow, heavy metal with down-tuned guitars, crushing riffs, and dark, oppressive atmosphere",
         (0.4, 0.7), (0.2, 0.4), ["Black Sabbath", "Electric Wizard", "Candlemass", "Sleep"], "1970s",
         "a doom metal track at 55 BPM with crushingly slow down-tuned riffs, heavy bass, and an oppressive atmosphere"),
        ("Progressive Metal", ["prog metal"], (80, 180), "Technically complex metal with odd time signatures, extended compositions, and virtuosity",
         (0.6, 0.9), (0.3, 0.55), ["Dream Theater", "Tool", "Opeth", "Meshuggah"], "1980s",
         "a progressive metal track with complex time signature changes, technical guitar solos, and dynamic arrangements"),
        ("Nu-Metal", ["nu metal", "nü-metal"], (80, 140), "Metal blending rap, turntablism, and groove-oriented riffs with angst-driven lyrics",
         (0.7, 0.9), (0.4, 0.65), ["Linkin Park", "Korn", "Limp Bizkit", "System of a Down"], "1990s",
         "a nu-metal track with down-tuned groove riffs, rap-influenced vocals, and turntable scratches"),
        ("Metalcore", ["metal-core"], (110, 180), "Metal-punk hybrid with breakdowns, screamed/sung vocal contrast, and heavy riffs",
         (0.8, 1.0), (0.3, 0.55), ["Killswitch Engage", "As I Lay Dying", "Parkway Drive"], "2000s",
         "a metalcore track with heavy breakdowns, alternating screamed and clean vocals, and aggressive riffs"),
        ("Deathcore", ["death-core"], (80, 180), "Extreme blend of death metal and hardcore with ultra-heavy breakdowns",
         (0.85, 1.0), (0.2, 0.45), ["Suicide Silence", "Whitechapel", "Thy Art Is Murder"], "2000s",
         "a deathcore track with guttural vocals, crushing breakdowns, and blast beat sections"),
        ("Djent", ["djent music"], (80, 160), "Progressive metal subgenre defined by palm-muted polyrhythmic guitar tones",
         (0.6, 0.9), (0.3, 0.55), ["Meshuggah", "Periphery", "Animals as Leaders", "TesseracT"], "2000s",
         "a djent track with palm-muted polyrhythmic guitar patterns, complex grooves, and clean production"),
        ("Power Metal", ["power-metal"], (130, 180), "Melodic, epic metal with soaring vocals, fast tempos, and fantasy themes",
         (0.7, 0.9), (0.4, 0.65), ["Helloween", "Blind Guardian", "DragonForce", "Stratovarius"], "1980s",
         "a power metal track at 160 BPM with soaring vocals, rapid double-bass drumming, and epic melodic guitar leads"),
        ("Symphonic Metal", ["symphonic-metal"], (100, 160), "Metal blended with orchestral elements, operatic vocals, and cinematic arrangements",
         (0.6, 0.9), (0.3, 0.55), ["Nightwish", "Epica", "Within Temptation"], "1990s",
         "a symphonic metal track with full orchestral backing, operatic soprano vocals, and heavy guitar riffs"),
        ("Sludge Metal", ["sludge"], (50, 100), "Harsh blend of doom and hardcore punk with abrasive vocals and thick, sludgy tones",
         (0.6, 0.85), (0.2, 0.4), ["Eyehategod", "Crowbar", "Acid Bath", "Melvins"], "1980s",
         "a sludge metal track at 70 BPM with thick down-tuned guitars, abrasive screamed vocals, and crushing heaviness"),
        ("Stoner Metal", ["stoner-metal", "stoner doom"], (50, 100), "Heavy, fuzz-drenched metal with psychedelic influence and repetitive hypnotic riffs",
         (0.5, 0.8), (0.3, 0.5), ["Sleep", "Electric Wizard", "Bongripper"], "1990s",
         "a stoner metal track at 60 BPM with massive fuzz guitar riffs, a pounding slow tempo, and psychedelic atmosphere"),
        ("Grunge", ["grunge music", "Seattle sound"], (90, 150), "Raw, distortion-heavy rock with angst-driven lyrics and a sludgy, dynamic sound",
         (0.5, 0.85), (0.4, 0.6), ["Nirvana", "Pearl Jam", "Soundgarden", "Alice in Chains"], "1980s",
         "a grunge track with raw distorted guitar, dynamic quiet-loud shifts, and angst-driven vocals"),
        ("Alternative Rock", ["alt-rock", "alt rock"], (90, 150), "Diverse, non-mainstream rock encompassing many styles outside the commercial mainstream",
         (0.4, 0.8), (0.4, 0.65), ["R.E.M.", "Radiohead", "The Smashing Pumpkins", "Pixies"], "1980s",
         "an alternative rock track with dynamic guitar textures, melodic vocals, and unconventional song structure"),
        ("Shoegaze", ["shoe-gaze", "shoegazing"], (80, 130), "Dense, effect-laden guitar rock with walls of sound, buried vocals, and dreamy textures",
         (0.3, 0.6), (0.3, 0.5), ["My Bloody Valentine", "Slowdive", "Ride", "Lush"], "1980s",
         "a shoegaze track with dense layers of reverb-drenched guitar, buried whispered vocals, and a wall of sound"),
        ("Britpop", ["brit pop", "brit-pop"], (100, 140), "Guitar-based British pop-rock with witty lyrics, melodic hooks, and a confident swagger",
         (0.5, 0.8), (0.5, 0.7), ["Oasis", "Blur", "Pulp", "Suede"], "1990s",
         "a Britpop track with jangly British guitar, a catchy vocal melody, and a confident, swaggering attitude"),
        ("Indie Rock", ["indie-rock"], (100, 150), "Guitar-based rock from independent labels with DIY ethos and diverse styles",
         (0.4, 0.75), (0.4, 0.65), ["Arctic Monkeys", "The Strokes", "Modest Mouse", "Interpol"], "1980s",
         "an indie rock track with angular guitar riffs, a driving rhythm section, and an independent, raw aesthetic"),
        ("Post-Rock", ["post rock"], (60, 140), "Instrumental, cinematic rock with gradual builds, dynamic contrasts, and orchestral textures",
         (0.2, 0.8), (0.2, 0.4), ["Godspeed You! Black Emperor", "Mogwai", "Explosions in the Sky", "Sigur Ros"], "1990s",
         "a post-rock track with a slow crescendo build, tremolo guitars, and cinematic dynamic contrast"),
        ("Math Rock", ["math-rock"], (100, 170), "Technically complex rock with irregular time signatures, angular riffs, and precision",
         (0.5, 0.8), (0.3, 0.55), ["Battles", "Hella", "toe", "TTNG"], "1990s",
         "a math rock track with complex odd-time signatures, tapping guitar techniques, and intricate drum patterns"),
        ("Progressive Rock", ["prog rock", "prog"], (60, 160), "Epic, concept-driven rock with complex compositions, virtuosity, and diverse influences",
         (0.3, 0.8), (0.3, 0.55), ["Pink Floyd", "Yes", "Genesis", "King Crimson"], "1960s",
         "a progressive rock track with shifting time signatures, extended instrumental sections, and a concept narrative"),
        ("Psychedelic Rock", ["psych rock"], (80, 140), "Mind-expanding rock with phaser, delay, and experimental production evoking altered states",
         (0.4, 0.75), (0.4, 0.65), ["Pink Floyd", "Jimi Hendrix", "Tame Impala", "The Doors"], "1960s",
         "a psychedelic rock track with phaser-drenched guitars, trippy delay effects, and a mind-expanding arrangement"),
        ("Garage Rock", ["garage"], (120, 170), "Raw, lo-fi rock with a stripped-down, energetic sound and minimal production",
         (0.6, 0.9), (0.4, 0.65), ["The Stooges", "The White Stripes", "The Black Keys"], "1960s",
         "a garage rock track with raw distorted guitar, a simple driving beat, and lo-fi energy"),
        ("Surf Rock", ["surf"], (120, 160), "Reverb-drenched guitar instrumentals with a beach aesthetic and tremolo-picked melodies",
         (0.5, 0.8), (0.5, 0.7), ["The Beach Boys", "Dick Dale", "The Ventures"], "1960s",
         "a surf rock track with heavy spring reverb on a tremolo-picked guitar melody and a driving drum beat"),
        ("Southern Rock", ["southern"], (90, 140), "Blues-influenced rock from the American South with dual guitars and extended jams",
         (0.5, 0.8), (0.4, 0.65), ["Lynyrd Skynyrd", "The Allman Brothers", "ZZ Top"], "1970s",
         "a Southern rock track with dual guitar harmonies, a bluesy groove, and a slide guitar solo"),
        ("Blues Rock", ["blues-rock"], (80, 140), "Rock rooted in blues with emotional guitar bending, 12-bar forms, and raw energy",
         (0.5, 0.8), (0.4, 0.65), ["Jimi Hendrix", "Stevie Ray Vaughan", "Gary Clark Jr."], "1960s",
         "a blues rock track with expressive guitar bends, a 12-bar blues form, and a raw, emotional vocal performance"),
        ("Folk Rock", ["folk-rock"], (90, 140), "Rock blended with folk songwriting, acoustic instruments, and narrative lyrics",
         (0.4, 0.7), (0.4, 0.65), ["Bob Dylan", "Simon & Garfunkel", "Crosby, Stills, Nash & Young"], "1960s",
         "a folk rock track with jangly acoustic and electric guitars, harmonica, and narrative vocal delivery"),
        ("Country Rock", ["country-rock"], (100, 140), "Rock blended with country instrumentation, pedal steel, and twangy guitars",
         (0.5, 0.75), (0.4, 0.65), ["Eagles", "Gram Parsons", "Creedence Clearwater Revival"], "1960s",
         "a country rock track with twangy electric guitar, pedal steel, and a driving rock rhythm"),
        ("Emo", ["emo music", "emo rock"], (110, 170), "Emotional rock with confessional lyrics, dynamic shifts, and melodic/screamed vocals",
         (0.5, 0.85), (0.4, 0.6), ["My Chemical Romance", "Jimmy Eat World", "Dashboard Confessional", "American Football"], "1980s",
         "an emo rock track with confessional vocals, twinkly guitar arpeggios, and dynamic emotional shifts"),
        ("Screamo", ["screamo music", "skramz"], (140, 200), "Intense, chaotic emo with screamed vocals, dissonant guitars, and explosive energy",
         (0.8, 1.0), (0.3, 0.5), ["Orchid", "Saetia", "pg.99", "City of Caterpillar"], "1990s",
         "a screamo track with explosive screamed vocals, chaotic dissonant guitar, and abrupt dynamic contrasts"),
        ("Noise Rock", ["noise-rock"], (80, 160), "Abrasive, dissonant rock using feedback, distortion, and noise as core elements",
         (0.6, 0.9), (0.3, 0.5), ["Sonic Youth", "Big Black", "Lightning Bolt"], "1980s",
         "a noise rock track with abrasive distorted guitars, heavy feedback, and a dissonant, angular groove"),
        ("Industrial Rock", ["industrial-rock"], (100, 150), "Aggressive rock blended with industrial electronic elements, samples, and distortion",
         (0.7, 0.95), (0.4, 0.6), ["Nine Inch Nails", "Ministry", "Rammstein", "KMFDM"], "1980s",
         "an industrial rock track with distorted guitars, electronic drum loops, and aggressive synth-driven production"),
        ("Stoner Rock", ["stoner-rock", "desert rock"], (70, 130), "Fuzzy, groove-heavy rock with psychedelic influence and a laid-back, heavy feel",
         (0.5, 0.8), (0.4, 0.6), ["Kyuss", "Queens of the Stone Age", "Fu Manchu"], "1990s",
         "a stoner rock track at 95 BPM with a fuzzy guitar riff, heavy groove, and a psychedelic desert vibe"),
        ("Space Rock", ["space-rock"], (70, 130), "Spacious, cosmic rock with extended jams, reverb, and sci-fi-inspired soundscapes",
         (0.3, 0.7), (0.3, 0.5), ["Hawkwind", "Spiritualized", "Spacemen 3"], "1970s",
         "a space rock track with cosmic reverb-drenched guitars, droning synths, and an expansive, otherworldly atmosphere"),
        ("Krautrock", ["kraut rock", "kosmische"], (70, 140), "German experimental rock with motorik beats, electronic experimentation, and hypnotic repetition",
         (0.3, 0.7), (0.4, 0.65), ["Can", "Kraftwerk", "Neu!", "Tangerine Dream"], "1970s",
         "a krautrock track with a motorik drum beat, hypnotic bass, and evolving electronic textures"),
        ("New Wave", ["new-wave"], (100, 145), "Post-punk-influenced pop-rock with synths, angular guitars, and art-school aesthetics",
         (0.5, 0.8), (0.5, 0.75), ["Talking Heads", "Blondie", "The Cars", "Devo"], "1970s",
         "a new wave track with angular guitar, synth hooks, and an art-school pop sensibility"),
        ("Gothic Rock", ["goth rock", "goth"], (80, 140), "Dark, atmospheric rock with deep vocals, reverb-heavy guitars, and brooding aesthetics",
         (0.3, 0.65), (0.3, 0.55), ["The Cure", "Bauhaus", "Sisters of Mercy", "Siouxsie and the Banshees"], "1970s",
         "a gothic rock track with deep reverb-heavy guitars, a brooding bassline, and dark atmospheric vocals"),
        ("Skate Punk", ["skate-punk"], (160, 220), "Fast, melodic punk associated with skateboarding culture with technical drumming",
         (0.8, 1.0), (0.4, 0.6), ["NOFX", "Pennywise", "Bad Religion", "Propagandhi"], "1980s",
         "a skate punk track at 190 BPM with blazing fast drums, melodic vocal hooks, and shredding guitar"),
        ("Madchester", ["madchester"], (100, 130), "Baggy, dance-influenced indie rock from Manchester with groovy beats and psychedelic guitars",
         (0.5, 0.8), (0.6, 0.8), ["The Stone Roses", "Happy Mondays", "The Charlatans"], "1980s",
         "a Madchester track with a baggy drum groove, wah-wah guitar, and a dance-influenced indie rock feel"),
    ]

    for sub in _rock_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Rock", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "power chords"],
            defining_characteristics=[], typical_instruments=["electric guitar", "bass", "drums"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Rock"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.05, 0.3),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # JAZZ  (~30 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Jazz", id=_id(), parent=None,
        aliases=["jazz music"], bpm_range=(60, 240),
        key_tendencies=["major 7th chords", "dominant 7th chords", "Dorian mode", "Mixolydian mode",
                        "ii-V-I progressions"],
        defining_characteristics=["improvisation", "swing feel", "complex harmony",
                                  "call-and-response", "blue notes"],
        typical_instruments=["saxophone", "trumpet", "piano", "upright bass", "drum kit", "guitar"],
        production_style="Live performance-oriented production emphasizing improvisation, dynamics, and acoustic space",
        era_of_origin="1890s", parent_genres=["Blues", "Ragtime"],
        sibling_genres=["Blues", "R&B/Soul"],
        energy_range=(0.2, 0.8), danceability_range=(0.3, 0.7), acousticness_range=(0.4, 0.95),
        famous_artists=["Miles Davis", "John Coltrane", "Duke Ellington", "Charlie Parker", "Thelonious Monk"],
        clap_descriptions=[
            "a jazz piece with a swinging upright bass, brushed drums, and an improvised saxophone solo",
            "a jazz combo performance with piano voicings, walking bass, and a muted trumpet melody",
        ],
    ))

    _jazz_subs = [
        ("Bebop", ["bop"], (160, 280), "Fast, virtuosic jazz with complex harmony, rapid tempos, and intricate improvisation",
         (0.6, 0.85), (0.3, 0.55), ["Charlie Parker", "Dizzy Gillespie", "Thelonious Monk"], "1940s",
         "a bebop performance at 220 BPM with rapid saxophone lines, complex harmony, and a swinging rhythm section"),
        ("Hard Bop", ["hard-bop"], (100, 200), "Soulful, blues-rooted jazz with a harder edge, gospel influence, and rhythmic drive",
         (0.5, 0.8), (0.3, 0.6), ["Art Blakey", "Horace Silver", "Cannonball Adderley"], "1950s",
         "a hard bop track with a soulful saxophone melody, bluesy piano comping, and a driving drum groove"),
        ("Cool Jazz", ["cool", "West Coast jazz"], (80, 160), "Relaxed, restrained jazz with lighter tones, smooth arrangements, and cerebral approach",
         (0.2, 0.5), (0.3, 0.55), ["Chet Baker", "Dave Brubeck", "Stan Getz", "Miles Davis"], "1950s",
         "a cool jazz piece with a breathy trumpet melody, vibraphone chords, and a relaxed brushed-drum groove"),
        ("Free Jazz", ["free"], (0, 300), "Atonal, structurally free improvisation breaking from conventional jazz harmony and form",
         (0.3, 0.9), (0.1, 0.35), ["Ornette Coleman", "Cecil Taylor", "Albert Ayler", "Sun Ra"], "1960s",
         "a free jazz performance with atonal saxophone, dissonant piano clusters, and free improvised drumming"),
        ("Modal Jazz", ["modal"], (80, 160), "Jazz based on modes rather than chord progressions, creating spacious harmonic landscapes",
         (0.3, 0.6), (0.3, 0.55), ["Miles Davis", "John Coltrane", "McCoy Tyner", "Bill Evans"], "1950s",
         "a modal jazz piece with a Dorian-mode saxophone solo over a sparse piano accompaniment and walking bass"),
        ("Jazz Fusion", ["fusion", "jazz-rock"], (80, 180), "Jazz blended with rock, funk, and electronic elements with electric instruments",
         (0.5, 0.85), (0.4, 0.7), ["Miles Davis", "Weather Report", "Herbie Hancock", "Mahavishnu Orchestra"], "1960s",
         "a jazz fusion track with an electric guitar solo, synth pads, and a funky bass and drums groove"),
        ("Smooth Jazz", ["smooth"], (70, 110), "Polished, radio-friendly jazz with a focus on melody, production, and accessibility",
         (0.2, 0.45), (0.4, 0.65), ["Kenny G", "George Benson", "Grover Washington Jr."], "1970s",
         "a smooth jazz track with a polished soprano saxophone melody, soft electric piano, and a gentle groove"),
        ("Acid Jazz", ["acid-jazz"], (90, 120), "Jazz blended with funk, soul, and electronic beats for a dancefloor-oriented sound",
         (0.5, 0.75), (0.6, 0.8), ["Jamiroquai", "The Brand New Heavies", "US3", "Incognito"], "1980s",
         "an acid jazz track at 110 BPM with a funky drum loop, jazzy organ, and a groovy bass riff"),
        ("Nu Jazz", ["nu-jazz"], (80, 130), "Modern jazz blending electronic production with acoustic jazz elements",
         (0.3, 0.6), (0.4, 0.7), ["Jaga Jazzist", "The Cinematic Orchestra", "Bugge Wesseltoft"], "1990s",
         "a nu jazz track with live saxophone over electronic beats, glitchy textures, and deep bass"),
        ("Latin Jazz", ["latin-jazz"], (100, 200), "Jazz incorporating Latin American rhythms, clave patterns, and Caribbean percussion",
         (0.5, 0.8), (0.6, 0.85), ["Tito Puente", "Machito", "Eddie Palmieri"], "1940s",
         "a Latin jazz track with Afro-Cuban percussion, a swinging clave pattern, and a fiery piano solo"),
        ("Afro-Cuban Jazz", ["Cubop"], (100, 200), "Jazz fused with Cuban musical traditions, congas, and Afro-Caribbean rhythms",
         (0.5, 0.85), (0.6, 0.85), ["Dizzy Gillespie", "Chano Pozo", "Chucho Valdes"], "1940s",
         "an Afro-Cuban jazz track with conga drums, a clave rhythm, and a trumpet improvisation over montuno piano"),
        ("Bossa Nova", ["bossa"], (100, 140), "Brazilian jazz with gentle rhythms, nylon guitar, and intimate vocal delivery",
         (0.2, 0.45), (0.5, 0.7), ["Antonio Carlos Jobim", "Joao Gilberto", "Stan Getz"], "1950s",
         "a bossa nova track with nylon guitar, gentle brush drums, and a soft vocal over flowing chord changes"),
        ("Gypsy Jazz", ["manouche", "Django jazz"], (120, 240), "Acoustic jazz with Romani influence, rapid guitar picking, and hot-swing energy",
         (0.5, 0.85), (0.5, 0.75), ["Django Reinhardt", "Stephane Grappelli", "Bireli Lagrene"], "1930s",
         "a gypsy jazz performance with rapid acoustic guitar picking, violin solo, and an energetic swing rhythm"),
        ("Swing", ["swing jazz", "swing music"], (120, 200), "Big band era jazz designed for dancing with strong rhythm sections and horn arrangements",
         (0.5, 0.85), (0.7, 0.9), ["Duke Ellington", "Count Basie", "Benny Goodman", "Glenn Miller"], "1930s",
         "a swing jazz big band arrangement with driving rhythm, brass section hits, and an infectious dance groove"),
        ("Big Band", ["big-band"], (100, 200), "Large jazz ensemble with written arrangements for brass, reeds, and rhythm sections",
         (0.5, 0.85), (0.5, 0.8), ["Duke Ellington", "Count Basie", "Stan Kenton"], "1920s",
         "a big band arrangement with a powerful brass section, flowing reed harmonies, and a swinging drum groove"),
        ("Dixieland", ["traditional jazz", "New Orleans jazz"], (100, 180), "Early jazz with collective improvisation, marching band influence, and a joyful feel",
         (0.5, 0.8), (0.5, 0.75), ["Louis Armstrong", "King Oliver", "Jelly Roll Morton"], "1900s",
         "a Dixieland jazz performance with a trumpet lead, clarinet obbligato, and a marching-style drum beat"),
        ("Spiritual Jazz", ["spiritual"], (60, 160), "Transcendent jazz with spiritual and metaphysical themes, drawing from world music",
         (0.3, 0.7), (0.3, 0.6), ["John Coltrane", "Pharoah Sanders", "Alice Coltrane", "Sun Ra"], "1960s",
         "a spiritual jazz piece with meditative saxophone, harp, and an otherworldly, transcendent atmosphere"),
        ("Ethiopian Jazz", ["Ethio-jazz"], (80, 140), "Ethiopian scales and melodies fused with jazz harmony and instrumentation",
         (0.4, 0.7), (0.4, 0.7), ["Mulatu Astatke", "Getatchew Mekurya"], "1960s",
         "an Ethio-jazz track with Ethiopian pentatonic melodies, vibraphone, and a laid-back jazz rhythm section"),
        ("Jazz-Funk", ["jazz funk"], (90, 130), "Groove-oriented jazz with heavy funk bass, rhythmic emphasis, and danceable energy",
         (0.5, 0.8), (0.6, 0.8), ["Herbie Hancock", "Roy Ayers", "The Headhunters"], "1970s",
         "a jazz-funk track at 110 BPM with a slap bass groove, clavinet riff, and a funky drum pattern"),
        ("Post-Bop", ["post bop"], (80, 200), "Adventurous jazz building on hard bop with freer harmony and structural experimentation",
         (0.4, 0.75), (0.3, 0.55), ["Wayne Shorter", "Herbie Hancock", "Andrew Hill"], "1960s",
         "a post-bop piece with adventurous harmony, a lyrical saxophone melody, and a dynamic rhythm section"),
        ("Avant-Garde Jazz", ["avant-garde"], (0, 300), "Boundary-pushing jazz exploring unconventional techniques, extended techniques, and noise",
         (0.3, 0.85), (0.1, 0.35), ["Sun Ra", "Anthony Braxton", "John Zorn"], "1960s",
         "an avant-garde jazz piece with extended techniques, dissonant textures, and unconventional improvisation"),
        ("Contemporary Jazz", ["modern jazz"], (70, 180), "Current jazz blending tradition with modern influences from hip-hop, R&B, and electronic",
         (0.3, 0.7), (0.3, 0.65), ["Robert Glasper", "Kamasi Washington", "Snarky Puppy", "GoGo Penguin"], "2000s",
         "a contemporary jazz track with modern production, hip-hop-influenced beats, and virtuosic acoustic playing"),
        ("Chamber Jazz", ["chamber-jazz"], (60, 140), "Intimate, small-ensemble jazz emphasizing delicate interplay and compositional detail",
         (0.2, 0.5), (0.2, 0.45), ["Jimmy Giuffre", "Paul Motian", "Dave Holland"], "1950s",
         "a chamber jazz piece with intimate acoustic interplay between piano, bass, and drums in a quiet, detailed arrangement"),
    ]

    for sub in _jazz_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Jazz", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major 7th chords", "Dorian mode"],
            defining_characteristics=[], typical_instruments=["saxophone", "piano", "bass", "drums"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Jazz"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.5, 0.95),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # COUNTRY  (~20 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Country", id=_id(), parent=None,
        aliases=["country music"], bpm_range=(70, 160),
        key_tendencies=["major keys", "Mixolydian mode", "pentatonic"],
        defining_characteristics=["twangy guitar", "pedal steel", "fiddle", "narrative lyrics",
                                  "Southern vocal style"],
        typical_instruments=["acoustic guitar", "pedal steel", "fiddle", "banjo", "bass"],
        production_style="Band-oriented production highlighting acoustic instruments, vocal storytelling, and regional character",
        era_of_origin="1920s", parent_genres=["Folk", "Blues", "Gospel"],
        sibling_genres=["Folk/World", "Bluegrass"],
        energy_range=(0.3, 0.75), danceability_range=(0.4, 0.75), acousticness_range=(0.3, 0.8),
        famous_artists=["Johnny Cash", "Dolly Parton", "Hank Williams", "Willie Nelson", "Garth Brooks"],
        clap_descriptions=[
            "a country track with twangy guitar, pedal steel, and a storytelling vocal over a shuffling rhythm",
        ],
    ))

    _country_subs = [
        ("Traditional Country", ["trad country"], (80, 130), "Classic country with fiddle, steel guitar, and honky-tonk storytelling",
         (0.3, 0.6), (0.4, 0.65), ["Hank Williams", "Merle Haggard", "George Jones"], "1920s",
         "a traditional country song with fiddle, pedal steel, and a storytelling vocal over a shuffling beat"),
        ("Honky-Tonk", ["honky tonk"], (100, 160), "Uptempo country rooted in barroom culture with a driving rhythm and twangy guitars",
         (0.5, 0.8), (0.6, 0.8), ["Hank Williams", "Ernest Tubb", "George Jones"], "1940s",
         "a honky-tonk track with a driving rhythm, twangy electric guitar, and a barroom piano"),
        ("Outlaw Country", ["outlaw"], (80, 130), "Rebellious country rejecting Nashville polish with a raw, rock-influenced sound",
         (0.5, 0.75), (0.4, 0.65), ["Willie Nelson", "Waylon Jennings", "Merle Haggard", "Johnny Cash"], "1970s",
         "an outlaw country track with a raw vocal, stripped-down arrangement, and a rock-influenced edge"),
        ("Country Pop", ["country-pop", "Nashville pop"], (90, 130), "Radio-friendly country with pop production and crossover appeal",
         (0.4, 0.7), (0.5, 0.75), ["Shania Twain", "Taylor Swift", "Carrie Underwood"], "1970s",
         "a country pop track with a polished vocal, pop production, and a catchy, radio-friendly chorus"),
        ("Bro-Country", ["bro country"], (100, 140), "Party-oriented country with hip-hop influence, trucks-and-beer themes, and pop hooks",
         (0.5, 0.8), (0.6, 0.8), ["Luke Bryan", "Florida Georgia Line", "Jason Aldean"], "2010s",
         "a bro-country track with a hip-hop-influenced beat, pop hooks, and a party-anthem energy"),
        ("Alt-Country", ["alternative country", "alt country"], (80, 140), "Country blended with punk, indie, and rock aesthetics",
         (0.4, 0.7), (0.4, 0.65), ["Wilco", "Uncle Tupelo", "Ryan Adams", "Lucinda Williams"], "1980s",
         "an alt-country track with distorted guitars, indie-rock energy, and country songwriting sensibility"),
        ("Americana", ["Americana music"], (70, 130), "Roots-oriented blend of country, folk, blues, and rock with authentic storytelling",
         (0.3, 0.65), (0.3, 0.6), ["Jason Isbell", "Sturgill Simpson", "Brandi Carlile"], "1990s",
         "an Americana track with acoustic guitar, warm vocals, and a blend of country, folk, and blues elements"),
        ("Bluegrass", ["bluegrass music"], (100, 180), "Acoustic string-band music with rapid picking, vocal harmonies, and virtuosic improvisation",
         (0.5, 0.85), (0.5, 0.7), ["Bill Monroe", "Earl Scruggs", "Alison Krauss", "Chris Thile"], "1940s",
         "a bluegrass track with rapid banjo picking, fiddle melody, and tight three-part vocal harmonies"),
        ("Country Blues", ["country-blues"], (70, 110), "Early blues style with acoustic guitar, rural themes, and solo performance tradition",
         (0.3, 0.55), (0.3, 0.5), ["Robert Johnson", "Mississippi John Hurt", "Charley Patton"], "1920s",
         "a country blues track with a fingerpicked acoustic guitar, a gravelly vocal, and a Delta-style groove"),
        ("Western Swing", ["western-swing"], (120, 180), "Swing-jazz-influenced country with big band elements and a danceable shuffle",
         (0.5, 0.8), (0.7, 0.85), ["Bob Wills", "Milton Brown", "Asleep at the Wheel"], "1930s",
         "a Western swing track with a fiddle-led melody, steel guitar, and a danceable swing rhythm section"),
        ("Nashville Sound", ["Nashville"], (80, 120), "Polished country with lush string arrangements and pop-oriented production",
         (0.3, 0.55), (0.4, 0.6), ["Patsy Cline", "Jim Reeves", "Eddy Arnold"], "1950s",
         "a Nashville sound track with lush string arrangements, a smooth vocal, and polished pop-country production"),
        ("Country Rap", ["hick-hop", "country rap"], (80, 120), "Country-hip-hop hybrid blending trap beats with country themes and instrumentation",
         (0.5, 0.8), (0.6, 0.8), ["Lil Nas X", "Jelly Roll", "Upchurch"], "2010s",
         "a country rap track with trap hi-hats, country guitar, and a blend of rap and country vocal delivery"),
        ("Red Dirt", ["red dirt country"], (90, 140), "Oklahoma-rooted country with raw, independent spirit and a rock-influenced edge",
         (0.4, 0.7), (0.4, 0.65), ["Cross Canadian Ragweed", "Turnpike Troubadours", "Stoney LaRue"], "1990s",
         "a Red Dirt country track with a raw vocal, rocking guitar, and an independent Oklahoma spirit"),
        ("Texas Country", ["Texas"], (80, 140), "Independently minded Texas country with singer-songwriter roots and honky-tonk energy",
         (0.4, 0.7), (0.4, 0.65), ["George Strait", "Robert Earl Keen", "Pat Green"], "1970s",
         "a Texas country track with a honky-tonk groove, fiddle, and a singer-songwriter vocal style"),
        ("Neotraditional Country", ["neo-traditional"], (80, 130), "Country returning to traditional instrumentation and classic themes in a modern context",
         (0.4, 0.65), (0.4, 0.6), ["George Strait", "Alan Jackson", "Randy Travis"], "1980s",
         "a neotraditional country track with fiddle, steel guitar, and a classic country vocal delivery"),
        ("Bakersfield Sound", ["Bakersfield"], (100, 150), "Hard-edged California country with Telecaster twang and a raw, electric energy",
         (0.5, 0.75), (0.5, 0.7), ["Buck Owens", "Merle Haggard", "Dwight Yoakam"], "1950s",
         "a Bakersfield sound track with a sharp Telecaster twang, a driving beat, and a raw honky-tonk energy"),
    ]

    for sub in _country_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Country", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "Mixolydian mode"],
            defining_characteristics=[], typical_instruments=["acoustic guitar", "pedal steel", "fiddle"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Country"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.3, 0.8),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # LATIN  (~30 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Latin", id=_id(), parent=None,
        aliases=["Latin music", "musica latina"], bpm_range=(70, 180),
        key_tendencies=["major keys", "minor keys", "clave rhythms"],
        defining_characteristics=["clave-based rhythms", "Latin percussion", "syncopation",
                                  "call-and-response", "dance-oriented"],
        typical_instruments=["congas", "timbales", "bongos", "guitar", "piano", "horns"],
        production_style="Rhythm-driven production centered on clave patterns, percussion ensembles, and dance grooves",
        era_of_origin="1900s", parent_genres=["African music", "Spanish music"],
        sibling_genres=["R&B/Soul", "Pop"],
        energy_range=(0.4, 0.9), danceability_range=(0.6, 0.95), acousticness_range=(0.1, 0.6),
        famous_artists=["Celia Cruz", "Carlos Santana", "Bad Bunny", "Shakira", "Daddy Yankee"],
        clap_descriptions=[
            "a Latin music track with conga drums, a clave rhythm, and a dance-oriented arrangement",
        ],
    ))

    _latin_subs = [
        ("Reggaeton", ["reggaeton music", "regueton"], (85, 100), "Urban Latin music with dembow rhythm, heavy bass, and dancehall influence",
         (0.6, 0.85), (0.7, 0.9), ["Daddy Yankee", "Bad Bunny", "J Balvin", "Ozuna"], "1990s",
         "a reggaeton track at 92 BPM with a dembow beat, heavy 808 bass, and a catchy Spanish vocal hook"),
        ("Latin Trap", ["Latin trap music"], (130, 160), "Trap music with Spanish lyrics, Latin melodic influence, and urban production",
         (0.5, 0.8), (0.6, 0.8), ["Bad Bunny", "Anuel AA", "Farruko"], "2010s",
         "a Latin trap beat at 140 BPM with Spanish vocals, trap hi-hats, and heavy 808 bass"),
        ("Salsa", ["salsa music"], (140, 200), "Energetic dance music with Afro-Cuban rhythms, horn sections, and call-and-response vocals",
         (0.6, 0.9), (0.8, 0.95), ["Celia Cruz", "Hector Lavoe", "Marc Anthony", "Fania All-Stars"], "1960s",
         "a salsa track at 180 BPM with a clave rhythm, powerful horn section, and energetic percussion"),
        ("Bachata", ["bachata music"], (120, 140), "Romantic Dominican music with melodic guitar, bongos, and heartfelt vocal delivery",
         (0.4, 0.65), (0.7, 0.85), ["Romeo Santos", "Prince Royce", "Aventura", "Juan Luis Guerra"], "1960s",
         "a bachata track at 130 BPM with a romantic guitar melody, bongo rhythms, and a heartfelt vocal"),
        ("Merengue", ["merengue music"], (120, 160), "Fast Dominican dance music with a driving tambora rhythm and accordion",
         (0.7, 0.9), (0.8, 0.95), ["Juan Luis Guerra", "Wilfrido Vargas", "Olga Tanon"], "1850s",
         "a merengue track at 140 BPM with a driving tambora beat, accordion melody, and high-energy dance groove"),
        ("Cumbia", ["cumbia music"], (80, 110), "Colombian dance music with a distinctive shuffled rhythm, accordion, and cumbia beat",
         (0.5, 0.75), (0.7, 0.9), ["Andres Landero", "Celso Pina", "Los Angeles Azules"], "1940s",
         "a cumbia track at 95 BPM with a shuffled cumbia rhythm, accordion, and a festive dance groove"),
        ("Samba", ["samba music"], (80, 130), "Brazilian carnival music with complex polyrhythmic percussion and infectious energy",
         (0.6, 0.9), (0.7, 0.95), ["Beth Carvalho", "Zeca Pagodinho", "Jorge Ben Jor"], "1910s",
         "a samba track with complex surdo and tamborim rhythms, cavaquinho, and infectious carnival energy"),
        ("MPB", ["Musica Popular Brasileira"], (70, 130), "Sophisticated Brazilian pop blending bossa, samba, rock, and folk influences",
         (0.3, 0.65), (0.4, 0.7), ["Caetano Veloso", "Gilberto Gil", "Chico Buarque", "Elis Regina"], "1960s",
         "an MPB track with sophisticated Brazilian harmony, nylon guitar, and a poetic Portuguese vocal"),
        ("Baile Funk", ["funk carioca", "Brazilian funk"], (120, 145), "Rio de Janeiro club music with heavy bass, MC vocals, and rapid-fire rhythms",
         (0.7, 0.95), (0.7, 0.9), ["MC Kevinho", "Anitta", "Ludmilla"], "1980s",
         "a baile funk track at 130 BPM with a heavy bass-driven beat, MC vocal chants, and Rio carnival energy"),
        ("Dembow", ["dembow music"], (110, 130), "Dominican club music with a rapid, driving rhythm and heavy bass",
         (0.7, 0.9), (0.7, 0.9), ["El Alfa", "Chimbala", "Rochy RD"], "2000s",
         "a dembow track at 118 BPM with a rapid driving rhythm, heavy bass, and energetic Dominican vocal delivery"),
        ("Corridos Tumbados", ["corridos", "corrido"], (80, 110), "Modern Mexican corridos blending traditional narrative with urban trap influence",
         (0.4, 0.7), (0.5, 0.75), ["Natanael Cano", "Junior H", "Peso Pluma"], "2010s",
         "a corridos tumbados track with a requinto guitar, trap hi-hats, and a narrative Mexican vocal"),
        ("Norteno", ["norteño", "norteno music"], (90, 140), "Northern Mexican music with accordion, bajo sexto, and polka-influenced rhythms",
         (0.5, 0.8), (0.6, 0.8), ["Los Tigres del Norte", "Ramon Ayala", "Intocable"], "1920s",
         "a norteno track with accordion, bajo sexto guitar, and a polka-influenced dance rhythm"),
        ("Banda", ["banda music"], (90, 140), "Mexican brass band music with a full wind ensemble and regional Mexican rhythms",
         (0.6, 0.85), (0.6, 0.85), ["Banda MS", "Banda El Recodo", "Julion Alvarez"], "1880s",
         "a banda track with a full brass ensemble, snare drum rhythm, and a festive dance groove"),
        ("Mariachi", ["mariachi music"], (80, 160), "Traditional Mexican ensemble with trumpets, violins, and guitarron",
         (0.4, 0.8), (0.4, 0.7), ["Vicente Fernandez", "Antonio Aguilar", "Mariachi Vargas"], "1890s",
         "a mariachi track with trumpet melody, violin section, and guitarron bass over a traditional rhythm"),
        ("Bolero", ["bolero music"], (60, 90), "Romantic Latin ballad with gentle guitar, lush strings, and passionate vocal delivery",
         (0.2, 0.4), (0.3, 0.5), ["Trio Los Panchos", "Luis Miguel", "Armando Manzanero"], "1880s",
         "a bolero ballad at 70 BPM with a gentle guitar, lush string arrangement, and a passionate vocal"),
        ("Tango", ["tango music"], (60, 80), "Argentine dance music with bandoneon, dramatic pauses, and passionate intensity",
         (0.4, 0.7), (0.6, 0.8), ["Carlos Gardel", "Astor Piazzolla", "Gotan Project"], "1880s",
         "a tango track with a dramatic bandoneon melody, rhythmic staccato piano, and passionate intensity"),
        ("Vallenato", ["vallenato music"], (80, 130), "Colombian folk music with accordion, caja drum, and guacharaca",
         (0.5, 0.75), (0.6, 0.8), ["Carlos Vives", "Diomedes Diaz", "Silvestre Dangond"], "1800s",
         "a vallenato track with an accordion melody, caja drum rhythm, and an energetic Colombian groove"),
        ("Champeta", ["champeta music"], (100, 120), "Colombian Afro-Caribbean dance music with heavy bass and electronic production",
         (0.6, 0.85), (0.7, 0.9), ["Viviano Torres", "Louis Towers"], "1970s",
         "a champeta track at 110 BPM with heavy bass, Caribbean rhythms, and a high-energy dance groove"),
        ("Zouk", ["zouk music"], (80, 120), "French Caribbean dance music with a smooth, sensual rhythm and electronic production",
         (0.4, 0.7), (0.7, 0.85), ["Kassav'", "Jocelyne Beroard"], "1980s",
         "a zouk track at 95 BPM with a smooth sensual groove, synth pads, and Caribbean percussion"),
        ("Soca", ["soca music"], (140, 170), "Trinidad carnival music with high-energy grooves, brass, and uptempo rhythms",
         (0.7, 0.95), (0.8, 0.95), ["Machel Montano", "Bunji Garlin", "Destra Garcia"], "1970s",
         "a soca track at 155 BPM with a high-energy carnival rhythm, brass stabs, and an infectious dance groove"),
        ("Dancehall", ["dancehall music"], (80, 110), "Jamaican electronic dance music with heavy bass, digital riddims, and DJ vocals",
         (0.6, 0.85), (0.7, 0.9), ["Shabba Ranks", "Beenie Man", "Sean Paul", "Vybz Kartel"], "1980s",
         "a dancehall track at 95 BPM with a digital riddim, heavy bass, and energetic Jamaican vocal delivery"),
        ("Reggae", ["reggae music"], (65, 90), "Jamaican music with off-beat guitar skank, heavy bass, and a one-drop rhythm",
         (0.3, 0.6), (0.6, 0.8), ["Bob Marley", "Peter Tosh", "Jimmy Cliff", "Burning Spear"], "1960s",
         "a reggae track at 78 BPM with an off-beat guitar skank, a heavy one-drop bass, and a laid-back groove"),
        ("Dub", ["dub music"], (60, 90), "Remix-oriented Jamaican music with heavy echo, reverb, and bass-driven deconstruction",
         (0.3, 0.55), (0.5, 0.7), ["King Tubby", "Lee Scratch Perry", "Scientist", "Mad Professor"], "1960s",
         "a dub track with heavy echo and reverb, a deep bass drop, and stripped-down deconstructed reggae"),
        ("Roots Reggae", ["roots"], (65, 85), "Spiritually conscious reggae with Rastafarian themes, deep bass, and nyabinghi rhythms",
         (0.3, 0.55), (0.6, 0.75), ["Bob Marley", "Burning Spear", "Culture", "Steel Pulse"], "1970s",
         "a roots reggae track with a deep bass line, nyabinghi-influenced drums, and a spiritual vocal"),
        ("Ska", ["ska music"], (100, 170), "Jamaican music predating reggae with an uptempo off-beat guitar and walking bass",
         (0.6, 0.8), (0.7, 0.85), ["The Skatalites", "Desmond Dekker", "The Specials"], "1950s",
         "a ska track at 140 BPM with a bright off-beat guitar, walking bass, and a brass section melody"),
        ("Rocksteady", ["rock steady"], (70, 90), "Slower Jamaican music bridging ska and reggae with a smooth, laid-back groove",
         (0.3, 0.55), (0.6, 0.75), ["Alton Ellis", "The Paragons", "Toots and the Maytals"], "1960s",
         "a rocksteady track at 80 BPM with a smooth bass groove, gentle off-beat guitar, and sweet vocal harmonies"),
    ]

    for sub in _latin_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Latin", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "minor keys"],
            defining_characteristics=[], typical_instruments=["congas", "guitar", "bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Latin"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.1, 0.5),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # CLASSICAL / ORCHESTRAL  (~25 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Classical", id=_id(), parent=None,
        aliases=["classical music", "orchestral"], bpm_range=(40, 200),
        key_tendencies=["major keys", "minor keys", "modal", "chromatic"],
        defining_characteristics=["composed notation", "orchestral instruments", "dynamic range",
                                  "formal structure", "counterpoint"],
        typical_instruments=["violin", "cello", "piano", "flute", "oboe", "French horn", "timpani"],
        production_style="Notation-based composition performed by acoustic ensembles with emphasis on dynamics and expression",
        era_of_origin="1600s", parent_genres=[],
        sibling_genres=["Jazz", "Folk/World"],
        energy_range=(0.1, 0.9), danceability_range=(0.1, 0.5), acousticness_range=(0.7, 1.0),
        famous_artists=["Ludwig van Beethoven", "Johann Sebastian Bach", "Wolfgang Amadeus Mozart",
                        "Claude Debussy", "Igor Stravinsky"],
        clap_descriptions=[
            "a classical orchestral piece with strings, woodwinds, and brass in a dynamic formal arrangement",
        ],
    ))

    _classical_subs = [
        ("Baroque", ["baroque music"], (60, 140), "Ornate, contrapuntal composition with harpsichord, figured bass, and elaborate ornamentation",
         (0.3, 0.7), (0.2, 0.45), ["J.S. Bach", "Handel", "Vivaldi", "Telemann"], "1600s",
         "a Baroque piece with harpsichord continuo, contrapuntal string lines, and ornate melodic ornamentation"),
        ("Classical Period", ["classical era", "Viennese classical"], (60, 180), "Balanced, elegant composition with clear forms, homophonic texture, and galant style",
         (0.3, 0.7), (0.2, 0.45), ["Mozart", "Haydn", "early Beethoven"], "1750s",
         "a Classical period piece with balanced phrases, an elegant piano melody, and a string quartet accompaniment"),
        ("Romantic", ["romantic period", "romantic music"], (40, 180), "Emotionally expressive composition with expanded orchestration, chromaticism, and virtuosity",
         (0.3, 0.85), (0.1, 0.35), ["Chopin", "Liszt", "Tchaikovsky", "Brahms", "Wagner"], "1800s",
         "a Romantic orchestral piece with sweeping string melodies, dramatic dynamics, and emotional depth"),
        ("Modern Classical", ["20th century classical"], (40, 200), "Experimental, boundary-pushing composition using atonality, serialism, and new techniques",
         (0.2, 0.8), (0.1, 0.3), ["Stravinsky", "Schoenberg", "Bartok", "Shostakovich"], "1900s",
         "a modern classical piece with dissonant harmonies, irregular rhythms, and an experimental structure"),
        ("Contemporary Classical", ["contemporary art music"], (40, 200), "Present-day art music exploring new timbres, electronics, and interdisciplinary approaches",
         (0.2, 0.7), (0.1, 0.3), ["John Adams", "Steve Reich", "Kaija Saariaho", "Thomas Ades"], "1960s",
         "a contemporary classical composition with electronic elements, extended techniques, and innovative sound design"),
        ("Minimalist", ["minimalism", "minimal music"], (60, 140), "Repetitive, gradually evolving composition with phasing patterns and simple harmonic motion",
         (0.2, 0.5), (0.3, 0.55), ["Steve Reich", "Philip Glass", "Terry Riley", "La Monte Young"], "1960s",
         "a minimalist composition with repeating arpeggiated patterns, gradual phasing, and hypnotic evolution"),
        ("Neoclassical", ["neo-classical", "neoclassicism"], (60, 160), "Modern composition drawing on classical forms and aesthetics with contemporary sensibility",
         (0.2, 0.6), (0.2, 0.4), ["Nils Frahm", "Olafur Arnalds", "Max Richter", "Ludovico Einaudi"], "2000s",
         "a neoclassical piano piece with modern minimalist sensibility, intimate dynamics, and classical form"),
        ("Film Score", ["movie soundtrack", "film music"], (40, 200), "Orchestral or hybrid composition designed to accompany visual storytelling on screen",
         (0.1, 0.9), (0.1, 0.3), ["John Williams", "Hans Zimmer", "Ennio Morricone", "Howard Shore"], "1930s",
         "a film score cue with dramatic orchestral swells, emotional string melodies, and cinematic percussion"),
        ("Chamber Music", ["chamber"], (50, 180), "Small ensemble composition for intimate performance spaces with detailed interplay",
         (0.2, 0.6), (0.1, 0.35), ["Beethoven", "Schubert", "Bartok", "Shostakovich"], "1600s",
         "a chamber music piece for string quartet with intricate four-voice interplay and dynamic expression"),
        ("Opera", ["opera music"], (40, 180), "Dramatic staged vocal music with orchestra, combining music, text, and theater",
         (0.3, 0.9), (0.1, 0.3), ["Verdi", "Puccini", "Mozart", "Wagner"], "1600s",
         "an operatic aria with a dramatic soprano voice, full orchestral accompaniment, and emotional intensity"),
        ("Choral", ["choral music"], (40, 140), "Multi-voice vocal ensemble music with rich harmonic textures and dynamic expression",
         (0.2, 0.7), (0.1, 0.3), ["J.S. Bach", "Handel", "Faure", "Eric Whitacre"], "1400s",
         "a choral work with rich four-part vocal harmony, dynamic swells, and resonant cathedral acoustics"),
        ("Solo Piano", ["piano music"], (40, 180), "Music composed for or performed on solo piano spanning all classical eras",
         (0.1, 0.7), (0.1, 0.3), ["Chopin", "Liszt", "Debussy", "Ravel", "Keith Jarrett"], "1700s",
         "a solo piano piece with expressive rubato, dynamic contrasts, and intimate acoustic resonance"),
        ("Impressionist", ["impressionism"], (50, 130), "French late-Romantic style emphasizing color, atmosphere, and non-functional harmony",
         (0.2, 0.5), (0.1, 0.3), ["Debussy", "Ravel", "Satie", "Faure"], "1880s",
         "an impressionist piece with whole-tone harmonies, delicate orchestration, and atmospheric, coloristic textures"),
        ("Concerto", ["concerto form"], (60, 200), "Composition featuring a solo instrument with orchestral accompaniment in three movements",
         (0.3, 0.9), (0.2, 0.35), ["Mozart", "Beethoven", "Rachmaninoff", "Tchaikovsky"], "1600s",
         "a concerto movement with a virtuosic piano solo, orchestral tutti passages, and dramatic cadenza"),
        ("Symphony", ["symphonic"], (40, 200), "Large-scale orchestral composition in multiple movements exploring diverse themes",
         (0.2, 0.9), (0.1, 0.3), ["Beethoven", "Mahler", "Brahms", "Dvorak"], "1700s",
         "a symphonic movement with a full orchestra, thematic development, and powerful dynamic contrasts"),
        ("Electronic Classical", ["electroacoustic"], (40, 160), "Classical composition incorporating electronic sound sources and processing",
         (0.2, 0.6), (0.1, 0.3), ["Karlheinz Stockhausen", "Pierre Henry", "Iannis Xenakis"], "1950s",
         "an electroacoustic composition with electronic synthesis, processed acoustic sounds, and spatial audio"),
        ("Ambient Classical", ["ambient neoclassical"], (40, 100), "Atmospheric classical music blending acoustic instruments with ambient production",
         (0.1, 0.3), (0.1, 0.2), ["Max Richter", "Olafur Arnalds", "Nils Frahm"], "2000s",
         "an ambient classical piece with soft piano, layered string textures, and ethereal electronic atmosphere"),
    ]

    for sub in _classical_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Classical", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "minor keys", "chromatic"],
            defining_characteristics=[], typical_instruments=["violin", "piano", "orchestra"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Classical"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.7, 1.0),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # FOLK / WORLD  (~30 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Folk/World", id=_id(), parent=None,
        aliases=["folk", "world music", "folk music"], bpm_range=(60, 180),
        key_tendencies=["major keys", "modal", "pentatonic", "Dorian mode", "Mixolydian mode"],
        defining_characteristics=["acoustic instruments", "regional traditions", "oral tradition",
                                  "cultural identity", "narrative lyrics"],
        typical_instruments=["acoustic guitar", "fiddle", "banjo", "flute", "percussion"],
        production_style="Acoustic, tradition-rooted production preserving regional character and live performance feel",
        era_of_origin="ancient", parent_genres=[],
        sibling_genres=["Country", "Classical"],
        energy_range=(0.2, 0.8), danceability_range=(0.3, 0.8), acousticness_range=(0.5, 1.0),
        famous_artists=["Bob Dylan", "Woody Guthrie", "Fela Kuti", "Ravi Shankar", "Joni Mitchell"],
        clap_descriptions=[
            "a folk track with acoustic guitar, traditional melodies, and a narrative vocal performance",
        ],
    ))

    _folk_subs = [
        ("American Folk", ["folk revival", "US folk"], (70, 140), "Acoustic singer-songwriter tradition with narrative lyrics and simple instrumentation",
         (0.2, 0.55), (0.3, 0.55), ["Bob Dylan", "Woody Guthrie", "Pete Seeger", "Joan Baez"], "1920s",
         "an American folk song with fingerpicked acoustic guitar, a storytelling vocal, and harmonica"),
        ("British Folk", ["English folk"], (70, 140), "Traditional English, Scottish, and Irish folk with rich ballad tradition",
         (0.2, 0.55), (0.3, 0.6), ["Fairport Convention", "Sandy Denny", "Martin Carthy"], "medieval",
         "a British folk song with a traditional ballad vocal, acoustic guitar, and a fiddle accompaniment"),
        ("Celtic", ["Celtic music", "Irish folk"], (80, 160), "Irish, Scottish, and Welsh traditional music with fiddle, tin whistle, and jigs/reels",
         (0.3, 0.8), (0.5, 0.8), ["The Chieftains", "Planxty", "Clannad", "The Dubliners"], "ancient",
         "a Celtic reel with a driving fiddle melody, tin whistle, and bodhran drum at a lively tempo"),
        ("Flamenco", ["flamenco music"], (60, 180), "Spanish Andalusian art with passionate guitar, rhythmic footwork, and soulful vocals",
         (0.4, 0.9), (0.4, 0.75), ["Paco de Lucia", "Camaron de la Isla", "Rosalia"], "1700s",
         "a flamenco piece with passionate nylon guitar, rhythmic palmas handclaps, and a soulful vocal"),
        ("Afrobeat", ["Afro-beat"], (100, 140), "West African genre blending highlife, funk, and jazz with polyrhythmic percussion and horns",
         (0.6, 0.85), (0.7, 0.9), ["Fela Kuti", "Tony Allen", "Antibalas"], "1960s",
         "an Afrobeat track at 120 BPM with polyrhythmic percussion, a funky bass groove, and a horn section"),
        ("Highlife", ["highlife music"], (90, 130), "West African popular music with jazzy guitar, horns, and a swinging danceable groove",
         (0.5, 0.75), (0.7, 0.85), ["E.T. Mensah", "Osibisa", "Ebo Taylor"], "1920s",
         "a highlife track with a jazzy guitar melody, horn section, and an upbeat West African dance groove"),
        ("Soukous", ["Congolese rumba"], (120, 160), "Central African dance music with rapid guitar picking and infectious rhythm",
         (0.6, 0.85), (0.7, 0.9), ["Franco", "Koffi Olomide", "Pepe Kalle"], "1960s",
         "a soukous track with rapid-fire guitar sebene picking, a driving bass groove, and energetic dance rhythms"),
        ("Gnawa", ["gnawa music"], (70, 120), "Moroccan spiritual music with hypnotic sintir bass, krakeb castanets, and trance-like repetition",
         (0.4, 0.7), (0.5, 0.7), ["Maalem Mahmoud Gania", "Hassan Hakmoun"], "ancient",
         "a gnawa trance piece with hypnotic sintir bass, clacking krakeb, and a chanting, repetitive vocal"),
        ("Qawwali", ["qawwali music"], (80, 160), "Sufi devotional music from South Asia with passionate vocals and rhythmic handclapping",
         (0.4, 0.85), (0.4, 0.7), ["Nusrat Fateh Ali Khan", "Sabri Brothers"], "1200s",
         "a qawwali performance with passionate vocal improvisation, tabla, harmonium, and rhythmic handclapping"),
        ("Hindustani", ["Hindustani classical", "North Indian classical"], (40, 160), "North Indian classical music with raga melodic framework and complex rhythm cycles",
         (0.2, 0.7), (0.2, 0.5), ["Ravi Shankar", "Ali Akbar Khan", "Zakir Hussain"], "ancient",
         "a Hindustani classical performance with sitar, tabla, and a meditative raga improvisation"),
        ("Carnatic", ["Carnatic classical", "South Indian classical"], (60, 200), "South Indian classical music with intricate vocal ornamentation and rhythmic complexity",
         (0.3, 0.75), (0.2, 0.5), ["M.S. Subbulakshmi", "L. Shankar"], "ancient",
         "a Carnatic classical performance with intricate vocal ornamentation, mridangam drums, and veena"),
        ("Gamelan", ["gamelan music"], (60, 120), "Indonesian ensemble music with tuned metallophones, gongs, and interlocking patterns",
         (0.3, 0.6), (0.3, 0.55), ["Javanese gamelan", "Balinese gamelan"], "ancient",
         "a gamelan piece with interlocking metallophone patterns, gong punctuation, and a shimmering metallic texture"),
        ("Klezmer", ["klezmer music"], (80, 200), "Eastern European Jewish music with clarinet, fiddle, and emotional melodic ornamentation",
         (0.4, 0.85), (0.5, 0.8), ["Dave Tarras", "The Klezmatics", "Giora Feidman"], "1500s",
         "a klezmer piece with a wailing clarinet melody, fiddle, and a lively hora dance rhythm"),
        ("Polka", ["polka music"], (100, 160), "Central European dance music with a distinctive two-beat rhythm and accordion",
         (0.6, 0.85), (0.7, 0.9), ["Frankie Yankovic", "Jimmy Sturr"], "1830s",
         "a polka track with a lively accordion, two-beat rhythm, and a cheerful, danceable energy"),
        ("Zydeco", ["zydeco music"], (100, 160), "Louisiana Creole music with accordion, washboard, and a driving dance groove",
         (0.5, 0.8), (0.7, 0.85), ["Clifton Chenier", "Buckwheat Zydeco"], "1950s",
         "a zydeco track with a pumping accordion, washboard rhythm, and a driving Louisiana dance groove"),
        ("Cajun", ["Cajun music"], (90, 150), "French-influenced Louisiana music with fiddle, accordion, and two-step rhythms",
         (0.4, 0.75), (0.6, 0.8), ["Iry LeJeune", "Beausoleil", "Steve Riley"], "1700s",
         "a Cajun track with fiddle, accordion, and a lively two-step dance rhythm"),
        ("Fado", ["fado music"], (50, 90), "Portuguese music expressing saudade with classical guitar and mournful vocal delivery",
         (0.2, 0.45), (0.2, 0.4), ["Amalia Rodrigues", "Mariza", "Ana Moura"], "1820s",
         "a fado piece with a mournful vocal, Portuguese classical guitar, and a deep sense of saudade"),
        ("Tropicalia", ["tropicalia music", "Tropicalism"], (80, 130), "Brazilian avant-garde pop blending bossa nova, psychedelic rock, and cultural critique",
         (0.4, 0.7), (0.4, 0.7), ["Caetano Veloso", "Gilberto Gil", "Os Mutantes", "Tom Ze"], "1960s",
         "a Tropicalia track with psychedelic guitar, bossa nova rhythm, and an experimental Brazilian arrangement"),
        ("Nordic Folk", ["Scandinavian folk", "Viking folk"], (60, 140), "Northern European folk with haunting melodies, hardingfele, and ancient modal scales",
         (0.2, 0.6), (0.3, 0.55), ["Wardruna", "Heilung", "Gjallarhorn"], "ancient",
         "a Nordic folk piece with a haunting hardingfele melody, droning strings, and ancient modal scales"),
        ("Arabic Music", ["Arabic", "Arab music"], (60, 140), "Middle Eastern music with maqam scales, oud, qanun, and complex rhythmic patterns",
         (0.3, 0.7), (0.4, 0.7), ["Umm Kulthum", "Fairuz", "Marcel Khalife"], "ancient",
         "an Arabic music piece with oud melody, maqam scales, and a complex rhythmic pattern on darbuka"),
        ("Turkish Music", ["Turkish folk", "Turkish classical"], (60, 150), "Turkish musical tradition with saz, makam scales, and distinctive microtonal intervals",
         (0.3, 0.7), (0.4, 0.7), ["Baris Manco", "Erkin Koray"], "ancient",
         "a Turkish music piece with a saz melody, makam scales, and distinctive microtonal ornamentation"),
        ("Persian Music", ["Iranian music"], (60, 140), "Iranian classical and folk music with tar, santur, and dastgah modal system",
         (0.3, 0.65), (0.3, 0.55), ["Mohammad Reza Shajarian", "Shahram Nazeri"], "ancient",
         "a Persian classical piece with tar, santur, and a meditative dastgah modal improvisation"),
        ("Enka", ["enka music"], (60, 100), "Japanese popular ballad genre with melismatic vocal style and emotional intensity",
         (0.3, 0.55), (0.2, 0.4), ["Hibari Misora", "Sayuri Ishikawa"], "1960s",
         "an enka ballad with a melismatic Japanese vocal, dramatic string arrangement, and emotional intensity"),
    ]

    for sub in _folk_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Folk/World", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["modal", "pentatonic"],
            defining_characteristics=[], typical_instruments=["acoustic instruments", "percussion"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Folk/World"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.5, 1.0),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ======================================================================
    # OTHER  (~30 subgenres)
    # ======================================================================
    genres.append(Genre(
        name="Other", id=_id(), parent=None,
        aliases=["miscellaneous", "uncategorized"], bpm_range=(0, 300),
        key_tendencies=["varied"],
        defining_characteristics=["genre-defying", "diverse applications"],
        typical_instruments=["varied"],
        production_style="Diverse production approaches spanning functional, experimental, and applied music",
        era_of_origin="ancient", parent_genres=[], sibling_genres=[],
        energy_range=(0.0, 1.0), danceability_range=(0.0, 1.0), acousticness_range=(0.0, 1.0),
        famous_artists=[],
        clap_descriptions=[],
    ))

    _other_subs = [
        ("Soundtrack", ["OST", "movie soundtrack"], (40, 200), "Music composed for film, TV, or media combining orchestral and electronic elements",
         (0.1, 0.9), (0.1, 0.3), ["Hans Zimmer", "John Williams", "Trent Reznor"], "1930s",
         "a cinematic soundtrack cue with orchestral swells, atmospheric synths, and dramatic percussion"),
        ("Video Game Music", ["game music", "VGM", "chiptune"], (80, 180), "Music composed for video games ranging from chiptune to full orchestral scores",
         (0.3, 0.9), (0.3, 0.7), ["Koji Kondo", "Nobuo Uematsu", "Mick Gordon"], "1970s",
         "a video game music track with chiptune arpeggios, energetic drums, and a catchy 8-bit melody"),
        ("Lounge", ["lounge music"], (80, 120), "Relaxed, sophisticated music for upscale environments with smooth arrangements",
         (0.1, 0.35), (0.4, 0.6), ["Esquivel", "Martin Denny", "Burt Bacharach"], "1950s",
         "a lounge music track with smooth vibraphone, bossa nova rhythm, and a relaxed, sophisticated feel"),
        ("Easy Listening", ["easy-listening"], (70, 120), "Gentle, unobtrusive background music with simple melodies and soft arrangements",
         (0.1, 0.3), (0.3, 0.55), ["Mantovani", "Percy Faith", "Andre Kostelanetz"], "1950s",
         "an easy listening piece with gentle strings, a soft melody, and an unobtrusive, relaxing arrangement"),
        ("New Age", ["new-age"], (60, 100), "Meditative, atmospheric music designed for relaxation and spiritual practices",
         (0.05, 0.25), (0.1, 0.3), ["Enya", "Yanni", "Kitaro", "Vangelis"], "1970s",
         "a new age track with shimmering synth pads, gentle nature sounds, and a meditative atmosphere"),
        ("Meditation", ["meditation music", "mindfulness"], (40, 80), "Ultra-calm music designed for meditation, yoga, and deep relaxation",
         (0.0, 0.15), (0.0, 0.1), ["Deuter", "Liquid Mind"], "1990s",
         "a meditation track with slow, evolving drone pads, gentle bells, and a deeply calming atmosphere"),
        ("Worship", ["CCM", "contemporary Christian", "praise"], (70, 140), "Contemporary Christian music designed for congregational singing and worship services",
         (0.3, 0.75), (0.4, 0.7), ["Hillsong", "Chris Tomlin", "Bethel Music", "Elevation Worship"], "1970s",
         "a worship track with acoustic guitar, atmospheric pads, and a congregational vocal melody"),
        ("Spoken Word", ["spoken-word", "poetry"], (0, 120), "Vocal performance art with rhythmic speech, poetry, and minimal musical backing",
         (0.2, 0.5), (0.1, 0.3), ["Gil Scott-Heron", "Saul Williams", "Kate Tempest"], "1950s",
         "a spoken word piece with rhythmic vocal delivery over a sparse, atmospheric musical backing"),
        ("Musical Theater", ["Broadway", "show tunes", "musicals"], (70, 160), "Theatrical vocal music with dramatic arrangements serving narrative storytelling",
         (0.4, 0.85), (0.3, 0.7), ["Stephen Sondheim", "Andrew Lloyd Webber", "Lin-Manuel Miranda"], "1900s",
         "a musical theater number with a dramatic vocal performance, orchestral pit arrangement, and theatrical energy"),
        ("Experimental", ["experimental music", "avant-garde"], (0, 300), "Boundary-pushing music that defies conventional genres, forms, and expectations",
         (0.1, 0.8), (0.0, 0.3), ["John Cage", "Merzbow", "Bjork", "Arca"], "1950s",
         "an experimental music piece with unconventional sound sources, unpredictable structure, and sonic exploration"),
        ("Noise", ["noise music"], (0, 300), "Extreme music using distortion, feedback, and noise as primary compositional elements",
         (0.5, 1.0), (0.0, 0.15), ["Merzbow", "Whitehouse", "The Rita", "Prurient"], "1970s",
         "a noise music piece with walls of harsh distortion, feedback, and extreme sonic intensity"),
        ("Industrial", ["industrial music"], (100, 150), "Aggressive electronic music with factory sounds, distortion, and provocative aesthetics",
         (0.6, 0.9), (0.3, 0.55), ["Throbbing Gristle", "Einsturzende Neubauten", "Cabaret Voltaire"], "1970s",
         "an industrial music track with metallic clanging percussion, distorted synths, and an oppressive atmosphere"),
        ("Musique Concrete", ["musique concrete", "concrete music"], (0, 120), "Composition using recorded sounds as raw material, manipulated through tape techniques",
         (0.1, 0.5), (0.0, 0.15), ["Pierre Schaeffer", "Pierre Henry", "Luc Ferrari"], "1940s",
         "a musique concrete piece with manipulated tape recordings, environmental sounds, and abstract sonic collage"),
        ("Field Recordings", ["field recording", "phonography"], (0, 0), "Documentary audio recordings of natural and urban soundscapes",
         (0.0, 0.2), (0.0, 0.05), ["Chris Watson", "Bernie Krause", "Francisco Lopez"], "1890s",
         "a field recording of a natural environment with birdsong, wind, and flowing water captured in stereo"),
        ("ASMR", ["ASMR music", "tingles"], (0, 80), "Gentle, whispered, or textural audio designed to trigger relaxation response",
         (0.0, 0.1), (0.0, 0.1), ["various ASMR artists"], "2010s",
         "an ASMR audio piece with gentle whispering, soft tapping, and delicate textural sounds"),
        ("Children's Music", ["kids music", "nursery rhymes"], (80, 140), "Music designed for children with simple melodies, repetition, and educational content",
         (0.3, 0.7), (0.5, 0.8), ["Raffi", "The Wiggles", "Sesame Street"], "ancient",
         "a children's music track with a simple sing-along melody, cheerful instrumentation, and a playful feel"),
        ("Marching Band", ["marching music"], (110, 140), "Music for marching ensembles with brass, woodwinds, and percussion in outdoor performance",
         (0.6, 0.85), (0.5, 0.7), ["John Philip Sousa"], "1800s",
         "a marching band piece with brass fanfares, rolling snare drums, and a steady marching tempo"),
        ("Cabaret", ["cabaret music"], (80, 140), "Theatrical entertainment music for intimate venues with witty, dramatic performance",
         (0.4, 0.7), (0.4, 0.65), ["Marlene Dietrich", "Liza Minnelli", "Amanda Palmer"], "1880s",
         "a cabaret number with a theatrical vocal, piano accompaniment, and an intimate nightclub atmosphere"),
        ("Jingle", ["commercial music", "ad music"], (90, 140), "Short, catchy compositions designed for advertising and brand identification",
         (0.5, 0.8), (0.4, 0.7), ["various composers"], "1920s",
         "a jingle with a catchy, memorable melody, bright instrumentation, and an upbeat commercial energy"),
        ("Podcast Music", ["podcast intro", "podcast background"], (80, 120), "Functional music designed for podcast intros, outros, and background beds",
         (0.2, 0.5), (0.2, 0.5), ["various producers"], "2010s",
         "a podcast intro music bed with a clean guitar riff, gentle drums, and an approachable, modern feel"),
    ]

    for sub in _other_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Other", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["varied"],
            defining_characteristics=[], typical_instruments=["varied"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Other"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.1, 0.8),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # ==================================================================
    # ADDITIONAL SUBGENRES — expanding to 500+ total
    # ==================================================================

    # --- More Electronic subgenres ---
    _extra_electronic = [
        ("Tribal House", ["tribal"], (120, 130), "Percussion-heavy house with indigenous rhythms and primal energy",
         (0.6, 0.85), (0.7, 0.9), ["Chus & Ceballos", "Dennis Ferrer"], "1990s",
         "a tribal house track at 125 BPM with heavy percussion patterns, primal drum rhythms, and deep bass"),
        ("Disco House", ["disco-house", "nu-disco house"], (118, 128), "House music heavily sampling or recreating disco aesthetics",
         (0.6, 0.8), (0.75, 0.9), ["Purple Disco Machine", "Dimitri From Paris"], "1990s",
         "a disco house track at 122 BPM with chopped disco strings, a funky bassline, and a four-on-the-floor beat"),
        ("Organic House", ["organic"], (110, 125), "Downtempo house blending natural sounds with electronic textures",
         (0.3, 0.55), (0.5, 0.7), ["Stimming", "Be Svendsen"], "2010s",
         "an organic house track at 118 BPM with field recordings, gentle percussion, and lush natural textures"),
        ("Melodic House", ["melodic house & techno"], (118, 128), "Melodic, emotional house with warm synths and progressive structure",
         (0.4, 0.75), (0.6, 0.8), ["Ben Bohmer", "Jan Blomqvist", "Lane 8"], "2010s",
         "a melodic house track at 122 BPM with warm analog synths, an emotive melody, and a gentle groove"),
        ("Latin House", ["latin-house"], (120, 130), "House music infused with Latin percussion, congas, and salsa influence",
         (0.6, 0.85), (0.75, 0.9), ["Masters At Work", "Erick Morillo", "DJ Sneak"], "1990s",
         "a Latin house track at 125 BPM with conga patterns, Latin piano riffs, and a driving house beat"),
        ("Italo Disco", ["italo", "Italo-disco"], (110, 135), "Italian electronic dance music with melodic synths, vocoders, and pop hooks",
         (0.5, 0.8), (0.7, 0.85), ["Giorgio Moroder", "Baltimora", "Den Harrow"], "1970s",
         "an Italo disco track at 125 BPM with melodic synth arpeggios, vocoder vocals, and a driving drum machine"),
        ("Nu-Disco", ["nu disco", "new disco"], (110, 128), "Modern reinterpretation of disco with indie and electronic production",
         (0.5, 0.8), (0.7, 0.9), ["Todd Terje", "Lindstrom", "Roisin Murphy"], "2000s",
         "a nu-disco track at 118 BPM with funky bass, disco strings, and modern electronic production"),
        ("Rave", ["rave music"], (130, 150), "High-energy electronic music from early rave culture with breakbeats and synth stabs",
         (0.8, 0.95), (0.6, 0.85), ["The Prodigy", "SL2", "Altern-8"], "1980s",
         "a rave track at 140 BPM with breakbeats, hoover synth stabs, and euphoric piano riffs"),
        ("EBM", ["electronic body music"], (110, 140), "Dark, aggressive electronic dance with industrial elements and sequenced bass",
         (0.6, 0.85), (0.5, 0.75), ["Front 242", "Nitzer Ebb", "DAF"], "1980s",
         "an EBM track at 125 BPM with a sequenced bass synth, aggressive vocals, and a pounding drum machine"),
        ("Glitch", ["glitch music", "glitchcore"], (80, 150), "Experimental electronic music using digital errors and artifacts as core aesthetic",
         (0.3, 0.7), (0.2, 0.5), ["Oval", "Alva Noto", "Ryoji Ikeda"], "1990s",
         "a glitch track with digital artifacts, cut-up micro-samples, and an abstract rhythmic structure"),
        ("Witch House", ["witch-house", "drag"], (80, 130), "Dark, occult-themed electronic music with slowed-down beats and eerie textures",
         (0.3, 0.6), (0.3, 0.55), ["Salem", "Crystal Castles", "oOoOO"], "2000s",
         "a witch house track at 100 BPM with slowed-down beats, eerie synths, and a dark occult atmosphere"),
        ("UK Funky", ["funky house UK"], (125, 135), "London-born blend of UK garage with African and Caribbean rhythms",
         (0.6, 0.8), (0.7, 0.9), ["Crazy Cousinz", "Roska", "Champion"], "2000s",
         "a UK funky track at 130 BPM with African-influenced percussion, syncopated bass, and garage rhythms"),
        ("Jersey Club", ["jersey"], (130, 145), "Fast, sample-heavy club music from New Jersey with rapid-fire bed-squeak samples",
         (0.7, 0.9), (0.7, 0.9), ["DJ Sliink", "Nadus", "Uniiqu3"], "2000s",
         "a Jersey club track at 140 BPM with rapid-fire chopped vocal samples, heavy kicks, and a frenetic groove"),
        ("Baltimore Club", ["bmore club", "bmore"], (125, 140), "High-energy dance music from Baltimore with chopped vocals and heavy breakbeats",
         (0.7, 0.9), (0.7, 0.9), ["DJ Technics", "Rod Lee", "DJ K-Swift"], "1990s",
         "a Baltimore club track at 135 BPM with chopped vocal samples, heavy breakbeats, and a manic energy"),
        ("Wonky", ["wonky music"], (100, 140), "Bass-heavy electronic music with detuned synths and off-kilter rhythms",
         (0.4, 0.7), (0.4, 0.65), ["Hudson Mohawke", "Rustie", "Joker"], "2000s",
         "a wonky track at 120 BPM with detuned synth chords, off-kilter rhythms, and heavy bass"),
        ("Ambient Techno", ["ambient-techno"], (110, 130), "Spacious blend of ambient atmospheres with subtle techno rhythms",
         (0.2, 0.5), (0.4, 0.6), ["The Orb", "Global Communication", "Biosphere"], "1990s",
         "an ambient techno track at 120 BPM with spacious pads, subtle kick drums, and ethereal textures"),
        ("Trance Step", ["trancestep"], (140, 150), "Dubstep-trance hybrid with half-time drops and trance melodies",
         (0.6, 0.9), (0.5, 0.75), ["Seven Lions", "Wooli"], "2010s",
         "a trancestep track at 150 BPM with trance arpeggios, a heavy half-time dubstep drop, and emotional melodies"),
        ("Complextro", ["complextro music"], (125, 132), "Electro house with rapid, complex synth patches switching at high speed",
         (0.7, 0.9), (0.5, 0.75), ["Porter Robinson", "Madeon", "Savant"], "2010s",
         "a complextro track at 128 BPM with rapidly switching synth patches, complex bass design, and high energy"),
        ("Big Room", ["big room house"], (126, 132), "Festival-oriented house with massive drops, simple melodies, and crowd-pleasing builds",
         (0.8, 1.0), (0.6, 0.8), ["Martin Garrix", "Dimitri Vegas & Like Mike", "Hardwell"], "2010s",
         "a big room house track at 128 BPM with a massive drop, simple anthemic melody, and festival-ready build"),
        ("Progressive Techno", ["prog techno"], (122, 135), "Evolving techno with progressive structure, melodic elements, and deep grooves",
         (0.5, 0.8), (0.6, 0.8), ["Pryda", "Jeremy Olander", "Patrice Baumel"], "1990s",
         "a progressive techno track at 128 BPM with evolving layers, deep groove, and subtle melodic progression"),
        ("Leftfield", ["leftfield music"], (90, 140), "Experimental electronic music that defies easy categorization",
         (0.3, 0.7), (0.3, 0.65), ["Leftfield", "Amon Tobin", "Clark"], "1990s",
         "a leftfield electronic track with unexpected sound combinations, shifting rhythms, and experimental textures"),
        ("Microhouse", ["micro house"], (118, 130), "Ultra-minimal house with tiny click sounds, subtle textures, and stripped-down grooves",
         (0.2, 0.5), (0.5, 0.75), ["Akufen", "Jan Jelinek", "Luomo"], "2000s",
         "a microhouse track at 124 BPM with tiny click percussion, subtle textural layers, and a minimal groove"),
        ("Full-On Psytrance", ["full-on", "full on"], (140, 148), "Peak-time psytrance with driving basslines, euphoric melodies, and powerful energy",
         (0.8, 0.95), (0.6, 0.8), ["Astrix", "Ace Ventura", "Coming Soon"], "2000s",
         "a full-on psytrance track at 145 BPM with a driving bassline, euphoric melodies, and peak-time energy"),
        ("Dark Psytrance", ["darkpsy", "dark psy"], (148, 160), "Dark, twisted psytrance with horror-influenced textures and fast tempos",
         (0.8, 0.95), (0.5, 0.7), ["Parvati Records artists", "Dark Whisper"], "2000s",
         "a dark psytrance track at 155 BPM with twisted sound design, dark textures, and a relentless bassline"),
        ("Hi-Tech Psytrance", ["hi-tech", "hitech"], (160, 200), "Ultra-fast psytrance with complex sound design and manic energy",
         (0.9, 1.0), (0.5, 0.7), ["Megalopsy", "Kashyyyk"], "2010s",
         "a hi-tech psytrance track at 180 BPM with rapid-fire sound design, glitchy textures, and extreme speed"),
        ("Ambient House", ["ambient-house"], (100, 120), "Blissful blend of house rhythms with ambient pads and dreamy atmospheres",
         (0.2, 0.5), (0.5, 0.7), ["The KLF", "The Orb", "Orbital"], "1990s",
         "an ambient house track at 110 BPM with dreamy pads, a gentle four-on-the-floor beat, and spacious reverb"),
    ]

    for sub in _extra_electronic:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Electronic", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"],
            defining_characteristics=[], typical_instruments=["synthesizer", "drum machine"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Electronic"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.15),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Rock subgenres ---
    _extra_rock = [
        ("Anarcho-Punk", ["anarcho punk"], (150, 200), "Politically charged punk with anarchist themes and raw, aggressive energy",
         (0.8, 1.0), (0.3, 0.55), ["Crass", "Flux of Pink Indians", "Subhumans"], "1970s",
         "an anarcho-punk track with raw aggressive guitars, shouted political lyrics, and a fast punk beat"),
        ("Crust Punk", ["crust", "crustcore"], (140, 200), "Extremely raw punk blending hardcore with metal influence and lo-fi aesthetics",
         (0.85, 1.0), (0.3, 0.5), ["Amebix", "Nausea", "His Hero Is Gone"], "1980s",
         "a crust punk track with heavily distorted guitars, growled vocals, and a raw D-beat rhythm"),
        ("D-Beat", ["d-beat punk", "dis-beat"], (160, 220), "Hardcore punk defined by the distinctive D-beat drum pattern",
         (0.85, 1.0), (0.3, 0.5), ["Discharge", "Anti Cimex", "Wolfbrigade"], "1980s",
         "a D-beat punk track with the signature discharge drum pattern, distorted guitars, and aggressive vocals"),
        ("Street Punk", ["street-punk", "Oi!"], (140, 180), "Working-class punk with anthemic choruses and gang vocal chants",
         (0.7, 0.9), (0.4, 0.6), ["The Exploited", "Sham 69", "Cockney Rejects"], "1970s",
         "a street punk track with anthemic gang vocal choruses, simple power chords, and a driving beat"),
        ("Post-Metal", ["post metal"], (60, 140), "Heavy, atmospheric metal blending post-rock dynamics with crushing heaviness",
         (0.4, 0.85), (0.2, 0.4), ["Isis", "Neurosis", "Cult of Luna", "Russian Circles"], "1990s",
         "a post-metal track with a slow crescendo build from quiet shimmer to crushing heaviness and atmospheric textures"),
        ("Blackgaze", ["blackgaze music", "post-black metal"], (80, 180), "Black metal blended with shoegaze aesthetics, dreamy textures, and tremolo guitar",
         (0.5, 0.85), (0.2, 0.4), ["Deafheaven", "Alcest", "Wolves in the Throne Room"], "2000s",
         "a blackgaze track with tremolo-picked guitars, blast beats, and dreamy shoegaze-inspired atmospheric textures"),
        ("Grindcore", ["grind"], (180, 300), "Extreme fusion of hardcore punk and death metal with micro-songs and blast beats",
         (0.9, 1.0), (0.2, 0.35), ["Napalm Death", "Carcass", "Pig Destroyer"], "1980s",
         "a grindcore track at 250 BPM with blast beat drums, guttural vocals, and a 30-second micro-song structure"),
        ("Mathcore", ["math-core"], (100, 220), "Chaotic, technically complex metalcore with erratic time signatures",
         (0.8, 1.0), (0.2, 0.4), ["The Dillinger Escape Plan", "Converge", "Botch"], "1990s",
         "a mathcore track with erratic time signature shifts, chaotic dissonant riffs, and screamed vocals"),
        ("Groove Metal", ["groove-metal"], (90, 140), "Heavy metal focused on syncopated, headbanging groove riffs",
         (0.7, 0.9), (0.4, 0.6), ["Pantera", "Lamb of God", "Sepultura", "Machine Head"], "1990s",
         "a groove metal track with a syncopated headbanging riff, heavy palm-muted chugging, and a powerful groove"),
        ("Folk Metal", ["folk-metal"], (100, 180), "Metal incorporating folk instruments, melodies, and pagan/Viking themes",
         (0.6, 0.9), (0.4, 0.65), ["Ensiferum", "Finntroll", "Eluveitie", "Korpiklaani"], "1990s",
         "a folk metal track with a fiddle melody, heavy distorted guitar, and a galloping Viking-inspired rhythm"),
        ("Speed Metal", ["speed-metal"], (140, 220), "Fast, technically precise metal bridging NWOBHM and thrash with melodic vocals",
         (0.8, 1.0), (0.4, 0.6), ["Motorhead", "Agent Steel", "Exciter"], "1980s",
         "a speed metal track at 200 BPM with blazing guitar riffs, rapid double-bass drumming, and powerful vocals"),
        ("Gothic Metal", ["gothic-metal"], (80, 140), "Metal blending gothic rock atmosphere with heavy riffs and clean/harsh vocal contrast",
         (0.5, 0.8), (0.3, 0.5), ["Type O Negative", "Paradise Lost", "Lacuna Coil"], "1990s",
         "a gothic metal track with heavy riffs, clean female vocals, and a dark, romantic gothic atmosphere"),
        ("Drone Metal", ["drone-metal"], (20, 60), "Extremely slow, sustained metal tones creating a wall of sound",
         (0.3, 0.7), (0.1, 0.2), ["Sunn O)))", "Earth", "Boris"], "1990s",
         "a drone metal piece with extremely sustained low-frequency guitar tones and a monolithic wall of sound"),
        ("Melodic Death Metal", ["melodeath"], (120, 200), "Death metal with prominent melodic guitar harmonies and technical riffing",
         (0.7, 0.95), (0.3, 0.5), ["At the Gates", "In Flames", "Dark Tranquillity", "Amon Amarth"], "1990s",
         "a melodic death metal track with harmonized guitar leads, blast beats, and growled vocals over melodic riffs"),
        ("Technical Death Metal", ["tech death"], (120, 240), "Virtuosic death metal with extreme complexity, time changes, and technical prowess",
         (0.85, 1.0), (0.2, 0.4), ["Necrophagist", "Obscura", "Archspire", "Beyond Creation"], "1990s",
         "a technical death metal track with blindingly fast fretwork, complex time signatures, and inhuman precision"),
        ("Melodic Hardcore", ["melodic-hardcore", "melodic HC"], (140, 190), "Hardcore punk with melodic elements, singalong choruses, and emotional intensity",
         (0.7, 0.95), (0.4, 0.6), ["Have Heart", "Comeback Kid", "Counterparts"], "1990s",
         "a melodic hardcore track with driving double-time drums, singalong choruses, and emotional screamed vocals"),
        ("Noise Rock", ["noise rock extra"], (80, 160), "Spare entry for noise rock overlap",
         (0.6, 0.85), (0.3, 0.5), ["Sonic Youth", "Swans", "Big Black", "Lightning Bolt"], "1980s",
         "a noise rock track with layers of feedback, aggressive dissonance, and a pounding drum rhythm"),
        ("Slowcore", ["sadcore", "slow-core"], (50, 100), "Quiet, minimalist rock with sparse arrangements and melancholic mood",
         (0.1, 0.35), (0.2, 0.35), ["Low", "Red House Painters", "Codeine", "Duster"], "1990s",
         "a slowcore track at 65 BPM with a sparse guitar, whispered vocals, and a deeply melancholic atmosphere"),
        ("Midwest Emo", ["midwest-emo", "twinkle emo"], (100, 150), "Intricate, guitar-driven emo with tapping, arpeggios, and confessional lyrics",
         (0.4, 0.7), (0.3, 0.55), ["American Football", "Cap'n Jazz", "Algernon Cadwallader"], "1990s",
         "a midwest emo track with twinkling guitar arpeggios, intricate tapping, and heartfelt vocal delivery"),
        ("Post-Hardcore", ["post hardcore"], (110, 170), "Aggressive yet melodic evolution of hardcore punk with dynamic song structures",
         (0.6, 0.9), (0.3, 0.55), ["Fugazi", "At the Drive-In", "Refused", "La Dispute"], "1980s",
         "a post-hardcore track with dynamic shifts between melodic verses and aggressive screamed choruses"),
        ("Riot Grrrl", ["riot grrrl"], (130, 180), "Feminist punk movement with raw energy, political lyrics, and DIY ethos",
         (0.7, 0.9), (0.4, 0.6), ["Bikini Kill", "Sleater-Kinney", "Bratmobile", "L7"], "1990s",
         "a riot grrrl punk track with raw guitar, confrontational vocals, and a driving feminist energy"),
    ]

    for sub in _extra_rock:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Rock", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "power chords"],
            defining_characteristics=[], typical_instruments=["electric guitar", "bass", "drums"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Rock"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.05, 0.25),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Jazz subgenres ---
    _extra_jazz = [
        ("Third Stream", ["third-stream"], (60, 160), "Fusion of jazz improvisation with classical composition techniques",
         (0.3, 0.6), (0.2, 0.45), ["Gunther Schuller", "John Lewis"], "1950s",
         "a third stream composition blending jazz improvisation with classical chamber music instrumentation"),
        ("Electro-Swing", ["electro swing"], (110, 140), "Modern electronic production fused with 1920s-40s swing and jazz samples",
         (0.6, 0.85), (0.7, 0.85), ["Parov Stelar", "Caravan Palace", "Jamie Berry"], "2000s",
         "an electro-swing track at 125 BPM with vintage jazz samples, electronic beats, and a swinging groove"),
        ("Jazz Manouche", ["gypsy jazz extra", "manouche jazz"], (130, 220), "French gypsy jazz with rapid acoustic guitar and Romani flair",
         (0.5, 0.8), (0.5, 0.75), ["Django Reinhardt", "Bireli Lagrene", "Stochelo Rosenberg"], "1930s",
         "a jazz manouche piece with rapid acoustic guitar arpeggios, violin, and a hot-swing rhythm"),
        ("M-Base", ["m-base"], (80, 160), "Brooklyn-born collective approach to jazz with complex odd meters and funk influence",
         (0.5, 0.75), (0.4, 0.6), ["Steve Coleman", "Greg Osby", "Cassandra Wilson"], "1980s",
         "an M-Base jazz track with complex odd-meter rhythms, alto saxophone, and a funk-influenced groove"),
        ("Nordic Jazz", ["Scandinavian jazz", "ECM jazz"], (60, 140), "Spacious Nordic jazz with open harmony, reverb, and contemplative mood",
         (0.2, 0.5), (0.2, 0.45), ["Jan Garbarek", "Esbjorn Svensson Trio", "Nils Petter Molvaer"], "1970s",
         "a Nordic jazz piece with a spacious saxophone tone, crystalline piano, and a contemplative atmosphere"),
        ("Soul Jazz", ["soul-jazz"], (90, 130), "Funky, groove-oriented jazz with organ, blues influence, and danceable rhythms",
         (0.5, 0.75), (0.6, 0.8), ["Jimmy Smith", "Grant Green", "Lou Donaldson"], "1960s",
         "a soul jazz track with a Hammond organ groove, funky drums, and a bluesy saxophone melody"),
    ]

    for sub in _extra_jazz:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Jazz", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major 7th chords", "Dorian mode"],
            defining_characteristics=[], typical_instruments=["saxophone", "piano", "bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Jazz"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.4, 0.9),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More R&B/Soul subgenres ---
    _extra_rnb = [
        ("Trap Soul", ["trap-soul", "trapsoul"], (60, 90), "Dark, moody R&B production with trap beat influence and intimate vocals",
         (0.3, 0.6), (0.5, 0.7), ["Bryson Tiller", "6LACK", "Summer Walker"], "2010s",
         "a trap soul track at 75 BPM with moody trap hi-hats, dark pads, and intimate R&B vocal delivery"),
        ("Disco", ["disco music"], (110, 135), "Dance music with orchestral strings, four-on-the-floor kick, and funky bass",
         (0.6, 0.85), (0.8, 0.95), ["Donna Summer", "Bee Gees", "Chic", "Gloria Gaynor"], "1970s",
         "a disco track at 120 BPM with orchestral strings, a four-on-the-floor kick, and a funky bass groove"),
        ("Southern Soul", ["southern-soul"], (70, 110), "Deep soul from the American South with raw emotion and blues influence",
         (0.4, 0.7), (0.5, 0.7), ["Otis Redding", "Al Green", "Ann Peebles"], "1960s",
         "a Southern soul track with a raw, emotional vocal, Memphis horns, and a deep bluesy groove"),
        ("Philadelphia Soul", ["Philly soul"], (90, 120), "Lush, orchestrated soul from Philadelphia with sweeping strings and polished vocals",
         (0.4, 0.7), (0.6, 0.8), ["The O'Jays", "Harold Melvin", "MFSB"], "1970s",
         "a Philadelphia soul track with lush orchestral strings, a smooth vocal, and a polished disco-soul groove"),
        ("Chicago Soul", ["chi-town soul"], (90, 120), "Smooth, sophisticated soul from Chicago with Curtis Mayfield influence",
         (0.4, 0.7), (0.6, 0.8), ["Curtis Mayfield", "The Impressions", "Minnie Riperton"], "1960s",
         "a Chicago soul track with a falsetto vocal, lush strings, and a sophisticated arrangement"),
    ]

    for sub in _extra_rnb:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="R&B/Soul", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "major 7th chords"],
            defining_characteristics=[], typical_instruments=["vocals", "Rhodes", "bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["R&B/Soul"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.1, 0.5),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Pop subgenres ---
    _extra_pop = [
        ("Indie Folk", ["indie-folk"], (80, 130), "Folk-influenced indie with acoustic instruments, harmonies, and DIY ethos",
         (0.2, 0.55), (0.3, 0.55), ["Bon Iver", "Fleet Foxes", "Iron & Wine", "Sufjan Stevens"], "2000s",
         "an indie folk track with layered vocal harmonies, fingerpicked guitar, and a warm acoustic atmosphere"),
        ("Bedroom Pop", ["bedroom-pop"], (80, 120), "Lo-fi, home-recorded pop with intimate production and dreamy aesthetics",
         (0.2, 0.5), (0.4, 0.65), ["Clairo", "boy pablo", "Gus Dapperton", "mxmtoon"], "2010s",
         "a bedroom pop track with lo-fi home-recorded guitar, soft vocals, and a dreamy indie atmosphere"),
        ("Disco Pop", ["disco-pop"], (110, 130), "Pop music with strong disco influence, four-on-the-floor beats, and funky bass",
         (0.6, 0.8), (0.7, 0.9), ["Dua Lipa", "Doja Cat", "Jessie Ware"], "2010s",
         "a disco pop track at 118 BPM with a funky bassline, four-on-the-floor beat, and a catchy pop vocal"),
        ("Dark Pop", ["dark-pop"], (80, 130), "Pop with darker lyrical themes, moody production, and minor-key melodies",
         (0.3, 0.65), (0.4, 0.7), ["Billie Eilish", "Lorde", "Banks"], "2010s",
         "a dark pop track with moody minor-key synths, whispered vocals, and an atmospheric bass-heavy beat"),
        ("Synthpop Revival", ["synth-pop revival", "modern synthpop"], (110, 135), "Contemporary artists reviving 80s synth-pop aesthetics with modern production",
         (0.5, 0.8), (0.6, 0.8), ["The Weeknd", "Dua Lipa", "Tame Impala"], "2010s",
         "a synthpop revival track at 116 BPM with retro analog synths, gated reverb drums, and a modern pop vocal"),
        ("Progressive Pop", ["art-pop progressive"], (80, 130), "Ambitious pop with complex arrangements, concept albums, and genre-blending",
         (0.4, 0.7), (0.4, 0.65), ["Kate Bush", "Bjork", "Joanna Newsom", "FKA Twigs"], "1980s",
         "a progressive pop track with complex arrangement, unexpected harmonic shifts, and avant-garde production"),
        ("Nu-Gaze", ["nu gaze", "newgaze"], (80, 130), "Modern revival of shoegaze aesthetics with electronic production elements",
         (0.3, 0.6), (0.3, 0.55), ["M83", "Wild Nothing", "DIIV", "Nothing"], "2000s",
         "a nu-gaze track with dense reverb-drenched guitars, electronic beats, and ethereal vocal layers"),
    ]

    for sub in _extra_pop:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Pop", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "minor keys"],
            defining_characteristics=[], typical_instruments=["vocals", "synths", "guitar"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Pop"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.05, 0.4),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Latin subgenres ---
    _extra_latin = [
        ("Punta", ["punta music"], (130, 160), "Garifuna dance music from Central America with rapid percussion and call-and-response",
         (0.7, 0.9), (0.8, 0.95), ["Andy Palacio", "Aurelio Martinez"], "ancient",
         "a punta track at 145 BPM with rapid Garifuna drumming, call-and-response vocals, and a driving dance groove"),
        ("Reggaeton Romantico", ["romantic reggaeton"], (80, 95), "Slower, romantic variant of reggaeton with ballad-like vocals over dembow rhythms",
         (0.4, 0.6), (0.6, 0.8), ["Nicky Jam", "Ozuna", "Prince Royce"], "2000s",
         "a romantic reggaeton track at 88 BPM with a gentle dembow beat, emotional vocals, and romantic lyrics"),
        ("Perreo", ["perreo music"], (90, 100), "Dance-focused reggaeton subgenre with heavy bass and suggestive rhythms",
         (0.7, 0.9), (0.8, 0.95), ["Daddy Yankee", "Don Omar", "Bad Bunny"], "2000s",
         "a perreo track at 95 BPM with a heavy dembow beat, bass-driven production, and dance-focused energy"),
        ("Afro-Latin", ["afro Latin"], (90, 130), "Latin music emphasizing African-derived rhythmic traditions and percussion",
         (0.5, 0.8), (0.7, 0.9), ["Celia Cruz", "Tito Puente", "Cuco Valoy"], "1940s",
         "an Afro-Latin track with Afro-Cuban percussion, clave-based rhythms, and a danceable groove"),
        ("Forró", ["forro"], (100, 140), "Northeastern Brazilian dance music with accordion, triangle, and zabumba drum",
         (0.6, 0.85), (0.7, 0.9), ["Luiz Gonzaga", "Dominguinhos"], "1940s",
         "a forro track with accordion, triangle, and zabumba drum in a driving Northeastern Brazilian dance groove"),
        ("Axé", ["axe music"], (110, 140), "Bahian carnival music blending Afro-Brazilian rhythms with pop elements",
         (0.7, 0.9), (0.8, 0.95), ["Ivete Sangalo", "Olodum", "Daniela Mercury"], "1980s",
         "an axe track with Afro-Brazilian percussion, carnival energy, and an infectious pop melody"),
        ("Boogaloo", ["Latin boogaloo"], (100, 130), "1960s blend of Latin rhythms with soul/R&B popular with NYC Latin youth",
         (0.5, 0.75), (0.7, 0.85), ["Joe Bataan", "Pete Rodriguez", "Ray Barretto"], "1960s",
         "a boogaloo track with Latin percussion, soulful vocals, and a groovy blend of Latin and R&B rhythms"),
        ("Plena", ["plena music"], (100, 130), "Puerto Rican folk music with pandereta drums and call-and-response vocals",
         (0.5, 0.8), (0.7, 0.85), ["Mon Rivera", "Plena Libre"], "1800s",
         "a plena track with pandereta hand drums, call-and-response vocals, and a driving Puerto Rican rhythm"),
        ("Son Cubano", ["son", "Cuban son"], (90, 130), "Foundational Cuban genre blending Spanish guitar with African rhythms",
         (0.4, 0.7), (0.7, 0.85), ["Buena Vista Social Club", "Compay Segundo", "Arsenio Rodriguez"], "1920s",
         "a son cubano track with tres guitar, clave rhythm, and an infectious Cuban dance groove"),
    ]

    for sub in _extra_latin:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Latin", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "minor keys"],
            defining_characteristics=[], typical_instruments=["percussion", "guitar", "bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Latin"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.1, 0.5),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Hip-Hop subgenres ---
    _extra_hiphop = [
        ("Memphis Rap", ["Memphis", "Memphis hip-hop"], (65, 85), "Dark, lo-fi Southern rap with heavy bass, horror samples, and raw production",
         (0.5, 0.8), (0.5, 0.7), ["Three 6 Mafia", "DJ Paul", "Tommy Wright III"], "1990s",
         "a Memphis rap beat at 72 BPM with dark lo-fi production, horror movie samples, and heavy 808 bass"),
        ("Bounce", ["bounce music", "New Orleans bounce"], (95, 115), "New Orleans hip-hop subgenre with call-and-response and Triggerman beat",
         (0.7, 0.9), (0.7, 0.9), ["Juvenile", "Big Freedia", "DJ Jubilee"], "1980s",
         "a bounce beat at 105 BPM with the Triggerman sample, call-and-response chants, and a high-energy groove"),
        ("Hyphy", ["hyphy music"], (80, 110), "Bay Area hip-hop with bouncy beats, slap bass, and energetic party vibes",
         (0.6, 0.85), (0.6, 0.8), ["E-40", "Too Short", "Mac Dre", "Keak Da Sneak"], "2000s",
         "a hyphy beat at 95 BPM with a bouncy kick pattern, slap bass, and a high-energy Bay Area groove"),
        ("Detroit Hip-Hop", ["Detroit rap"], (80, 110), "Raw, aggressive hip-hop from Detroit with complex lyricism and gritty production",
         (0.5, 0.8), (0.5, 0.7), ["Eminem", "J Dilla", "Danny Brown", "Royce da 5'9\""], "1990s",
         "a Detroit hip-hop beat at 90 BPM with gritty drums, raw sample chops, and a hard-hitting groove"),
        ("Plugg", ["plugg music", "plug beats"], (130, 155), "Minimalist, ethereal trap subgenre with sparse 808s, flutes, and dreamy melodies",
         (0.4, 0.65), (0.5, 0.7), ["Pi'erre Bourne", "Playboi Carti", "Lil Yachty"], "2010s",
         "a plugg beat at 140 BPM with ethereal flute melody, sparse 808 bass, and a dreamy, minimalist vibe"),
        ("Rage Beats", ["rage", "rage rap"], (140, 165), "Aggressive, high-energy trap variant with distorted synths and moshpit energy",
         (0.7, 0.95), (0.5, 0.7), ["Trippie Redd", "Yeat", "Ken Carson", "Playboi Carti"], "2020s",
         "a rage beat at 150 BPM with distorted synth leads, aggressive 808 bass, and high-energy moshpit vibes"),
        ("Afro Rap", ["Afro hip-hop"], (90, 120), "African hip-hop blending local musical traditions with rap vocal styles",
         (0.5, 0.8), (0.6, 0.8), ["Sarkodie", "Nasty C", "Falz"], "2000s",
         "an Afro rap beat at 105 BPM with Afrobeat percussion, log drums, and a rap vocal delivery"),
        ("Latin Hip-Hop", ["Spanish rap", "Latin rap"], (80, 110), "Hip-hop with Spanish lyrics and Latin musical influences",
         (0.5, 0.8), (0.5, 0.75), ["Calle 13", "Residente", "Tego Calderon"], "1990s",
         "a Latin hip-hop beat at 90 BPM with Latin percussion, Spanish rap vocals, and a boom-bap-influenced groove"),
    ]

    for sub in _extra_hiphop:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Hip-Hop/Rap", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "blues scale"],
            defining_characteristics=[], typical_instruments=["MPC", "TR-808", "sampler"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Hip-Hop/Rap"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.2),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Country subgenres ---
    _extra_country = [
        ("Cowboy Western", ["western music", "cowboy"], (70, 120), "Romantic Western music with yodeling, prairie themes, and cowboy imagery",
         (0.3, 0.6), (0.3, 0.55), ["Gene Autry", "Roy Rogers", "Marty Robbins"], "1930s",
         "a cowboy Western track with a yodeling vocal, acoustic guitar, and a wide-open prairie atmosphere"),
        ("Progressive Country", ["progressive-country"], (80, 130), "Country pushing boundaries with rock, folk, and experimental elements",
         (0.4, 0.7), (0.4, 0.6), ["Sturgill Simpson", "Tyler Childers", "Orville Peck"], "2010s",
         "a progressive country track with psychedelic guitar effects, country vocal delivery, and genre-blending production"),
        ("Gothic Country", ["gothic americana", "goth country"], (60, 110), "Dark, brooding Americana with Gothic imagery and Southern Gothic themes",
         (0.3, 0.6), (0.3, 0.5), ["16 Horsepower", "Slim Cessna's Auto Club", "Those Poor Bastards"], "1990s",
         "a gothic country track with a dark banjo, brooding baritone vocals, and a Southern Gothic atmosphere"),
        ("Newgrass", ["new grass", "progressive bluegrass"], (100, 180), "Progressive bluegrass with jazz and rock influence and expanded instrumentation",
         (0.5, 0.8), (0.5, 0.7), ["Sam Bush", "Bela Fleck", "Nickel Creek", "Chris Thile"], "1970s",
         "a newgrass track with virtuosic banjo, jazz-influenced improvisation, and a progressive bluegrass arrangement"),
    ]

    for sub in _extra_country:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Country", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["major keys", "Mixolydian mode"],
            defining_characteristics=[], typical_instruments=["acoustic guitar", "fiddle", "banjo"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Country"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.3, 0.8),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Folk/World subgenres ---
    _extra_folk = [
        ("Mbalax", ["mbalax music"], (110, 140), "Senegalese popular music with sabar drumming and Wolof vocal traditions",
         (0.6, 0.85), (0.7, 0.9), ["Youssou N'Dour", "Baaba Maal"], "1970s",
         "a mbalax track with rapid sabar drumming, Wolof vocals, and an infectious Senegalese dance groove"),
        ("Rebetiko", ["rebetika"], (80, 140), "Greek urban folk music with bouzouki, expressing working-class life and melancholy",
         (0.4, 0.7), (0.5, 0.7), ["Markos Vamvakaris", "Vassilis Tsitsanis"], "1920s",
         "a rebetiko track with a bouzouki melody, a minor-key modal scale, and a melancholic Greek groove"),
        ("Slavic Folk", ["Slavic traditional"], (80, 150), "Traditional music from Slavic countries with choral harmonies and accordion",
         (0.4, 0.8), (0.5, 0.7), ["DakhaBrakha", "Loreena McKennitt"], "ancient",
         "a Slavic folk track with multi-part vocal harmonies, accordion, and a traditional Eastern European rhythm"),
        ("Aboriginal Music", ["Australian indigenous", "didgeridoo music"], (60, 120), "Australian Aboriginal music with didgeridoo, clapsticks, and ancient traditions",
         (0.3, 0.6), (0.3, 0.5), ["Yothu Yindi", "Geoffrey Gurrumul Yunupingu"], "ancient",
         "an Aboriginal music piece with didgeridoo drone, clapstick rhythm, and a deep spiritual atmosphere"),
        ("Throat Singing", ["Tuvan throat singing", "overtone singing"], (40, 100), "Central Asian vocal technique producing multiple pitches simultaneously",
         (0.2, 0.5), (0.2, 0.4), ["Huun-Huur-Tu", "Sainkho Namtchylak"], "ancient",
         "a throat singing performance with multiple overtone harmonics, a deep fundamental drone, and a meditative feel"),
        ("Balkan", ["Balkan music", "Balkan brass"], (100, 180), "Southeast European music with brass bands, irregular meters, and Romani influence",
         (0.5, 0.9), (0.6, 0.85), ["Goran Bregovic", "Boban Markovic", "Taraf de Haidouks"], "ancient",
         "a Balkan brass band track with a lively irregular meter, brass melodies, and high-energy Romani-influenced groove"),
        ("Bhangra", ["bhangra music"], (100, 140), "Punjabi folk dance music with dhol drum, tumbi, and energetic rhythms",
         (0.7, 0.9), (0.7, 0.9), ["Malkit Singh", "Jazzy B", "Diljit Dosanjh"], "1960s",
         "a bhangra track with a driving dhol drum, tumbi riffs, and an infectious Punjabi dance rhythm"),
        ("Reggae Fusion", ["reggae-fusion"], (80, 110), "Modern reggae blending with hip-hop, R&B, and electronic elements",
         (0.4, 0.7), (0.6, 0.8), ["Shaggy", "Damian Marley", "Protoje", "Chronixx"], "1990s",
         "a reggae fusion track with a modern riddim, hip-hop-influenced vocal flow, and electronic production"),
    ]

    for sub in _extra_folk:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Folk/World", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["modal", "pentatonic"],
            defining_characteristics=[], typical_instruments=["acoustic instruments", "percussion"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Folk/World"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.4, 0.95),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Classical subgenres ---
    _extra_classical = [
        ("Serialist", ["serialism", "twelve-tone"], (40, 160), "Composition using systematic ordering of pitches, rhythms, and dynamics",
         (0.3, 0.7), (0.1, 0.3), ["Schoenberg", "Webern", "Berg", "Boulez"], "1920s",
         "a serialist composition with a twelve-tone row, atonal harmony, and precisely structured rhythmic patterns"),
        ("Aleatoric", ["chance music", "indeterminate"], (0, 200), "Composition incorporating elements of chance and performer choice",
         (0.2, 0.6), (0.1, 0.2), ["John Cage", "Witold Lutoslawski", "Earle Brown"], "1950s",
         "an aleatoric composition with performer-chosen elements, chance-derived textures, and unpredictable structure"),
        ("Sonata", ["sonata form"], (50, 180), "Multi-movement composition for solo instrument or small ensemble in sonata form",
         (0.2, 0.7), (0.1, 0.3), ["Beethoven", "Mozart", "Schubert", "Chopin"], "1700s",
         "a sonata movement with thematic exposition, development, and recapitulation for solo piano"),
        ("Solo Strings", ["string solo"], (40, 180), "Music for solo string instruments (violin, cello, viola) showcasing virtuosity",
         (0.2, 0.7), (0.1, 0.3), ["Paganini", "Yo-Yo Ma", "Hilary Hahn"], "1600s",
         "a solo violin piece with virtuosic double-stops, expressive vibrato, and dynamic bowing techniques"),
        ("Spectral Music", ["spectralism"], (40, 120), "Composition based on analysis of acoustic sound spectra and overtone structures",
         (0.2, 0.5), (0.1, 0.2), ["Gerard Grisey", "Tristan Murail", "Kaija Saariaho"], "1970s",
         "a spectral music piece with slowly evolving harmonics derived from acoustic overtone analysis"),
    ]

    for sub in _extra_classical:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Classical", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["chromatic", "atonal"],
            defining_characteristics=[], typical_instruments=["orchestra", "piano", "strings"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Classical"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.7, 1.0),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Other subgenres ---
    _extra_other = [
        ("Chiptune", ["8-bit music", "chip music"], (100, 180), "Music created using sound chips from vintage video game consoles and computers",
         (0.5, 0.85), (0.5, 0.75), ["Anamanaguchi", "Bit Shifter", "Chipzel"], "1980s",
         "a chiptune track with 8-bit square wave melodies, pulse-width arpeggios, and a retro game console aesthetic"),
        ("Lo-fi Indie", ["lo-fi indie rock"], (80, 130), "Deliberately rough, home-recorded indie music with tape hiss and imperfections",
         (0.2, 0.5), (0.3, 0.55), ["Guided by Voices", "Pavement", "Elliott Smith"], "1980s",
         "a lo-fi indie track with a rough 4-track recording aesthetic, buried vocals, and a tape-saturated guitar"),
        ("Witch House", ["drag extra", "haunted house"], (70, 120), "Dark, occult-influenced electronic with pitch-shifted vocals and lo-fi aesthetics",
         (0.3, 0.6), (0.3, 0.5), ["Salem", "Pictureplane"], "2000s",
         "a witch house track with slowed-down vocals, eerie synth drones, and an occult-tinged atmosphere"),
        ("Darkwave", ["dark wave", "cold wave"], (100, 140), "Dark, synth-driven post-punk/new wave with gothic aesthetics and minor keys",
         (0.4, 0.7), (0.5, 0.7), ["Clan of Xymox", "Lebanon Hanover", "She Past Away", "Boy Harsher"], "1980s",
         "a darkwave track with cold analog synths, a driving drum machine, and a dark, brooding vocal"),
        ("Trap Metal", ["trap-metal", "nu-metal revival"], (130, 160), "Aggressive hybrid blending trap beats with metal screaming and distorted guitars",
         (0.8, 1.0), (0.4, 0.6), ["Scarlxrd", "ZillaKami", "City Morgue"], "2010s",
         "a trap metal track at 145 BPM with distorted 808 bass, screamed vocals, and metal guitar riffs"),
        ("Afrofuturism", ["afro-futurism", "Afrofuturist"], (80, 140), "Music blending African diaspora traditions with sci-fi and futuristic concepts",
         (0.4, 0.7), (0.4, 0.7), ["Sun Ra", "Janelle Monae", "Flying Lotus", "Shabazz Palaces"], "1960s",
         "an Afrofuturist track with cosmic synths, African percussion, and a forward-looking sci-fi aesthetic"),
        ("Intelligent Techno", ["smart techno"], (120, 135), "Cerebral, detail-oriented techno blending ambient and IDM influences",
         (0.3, 0.6), (0.4, 0.65), ["Plastikman", "Robert Hood", "Jeff Mills"], "1990s",
         "an intelligent techno track at 128 BPM with cerebral sound design, subtle rhythmic detail, and ambient textures"),
        ("Future Garage", ["future-garage"], (130, 140), "Atmospheric, nostalgic evolution of UK garage with reverb and vocal manipulation",
         (0.3, 0.6), (0.5, 0.7), ["Burial", "Jamie xx", "Mount Kimbie"], "2000s",
         "a future garage track at 135 BPM with crackly vinyl textures, time-stretched vocals, and deep sub-bass"),
        ("Post-Dubstep", ["post dubstep"], (130, 140), "Evolution of dubstep toward more atmospheric, experimental, and R&B-influenced territories",
         (0.3, 0.6), (0.4, 0.65), ["James Blake", "Mount Kimbie", "SBTRKT"], "2010s",
         "a post-dubstep track at 135 BPM with atmospheric vocals, sparse beats, and experimental bass textures"),
        ("Deconstructed Club", ["deconstructed", "experimental club"], (80, 160), "Avant-garde club music deconstructing dance tropes with industrial and noise elements",
         (0.5, 0.85), (0.3, 0.6), ["Arca", "SOPHIE", "Lotic", "Rabit"], "2010s",
         "a deconstructed club track with shattered beats, metallic textures, and an avant-garde club energy"),
        ("Kawaii Future Bass", ["kawaii bass", "kawaii"], (140, 170), "Cute, bright future bass with J-pop influence, chiptune elements, and sugary melodies",
         (0.5, 0.8), (0.5, 0.75), ["Snail's House", "YUC'e", "Moe Shop"], "2010s",
         "a kawaii future bass track at 155 BPM with bright chiptune melodies, sugary vocal chops, and bouncy rhythms"),
        ("Tropical Bass", ["tropical-bass", "global bass"], (90, 130), "Bass-heavy electronic music incorporating tropical rhythms from the global south",
         (0.5, 0.8), (0.6, 0.85), ["Major Lazer", "Diplo", "Buraka Som Sistema"], "2000s",
         "a tropical bass track at 110 BPM with Caribbean rhythms, heavy bass drops, and global dance influences"),
        ("Amapiano", ["amapiano music"], (110, 120), "South African house genre with log drum melodies, shakers, and jazzy piano",
         (0.5, 0.75), (0.7, 0.85), ["Kabza De Small", "DJ Maphorisa", "Focalistic"], "2010s",
         "an amapiano track at 115 BPM with log drum melodies, shaker percussion, and a jazzy bass groove"),
        ("UK Drill Remix", ["drill remix"], (138, 142), "UK drill instrumentals remixed or influenced by other genres like Afroswing or pop",
         (0.5, 0.8), (0.5, 0.75), ["Tion Wayne", "Russ Millions"], "2020s",
         "a UK drill remix track at 140 BPM blending drill patterns with Afroswing melodies and pop hooks"),
        ("Phonk House", ["house phonk", "drift phonk house"], (140, 155), "Aggressive house-tempo phonk with cowbell loops, distorted bass, and drift aesthetics",
         (0.7, 0.9), (0.6, 0.8), ["Kordhell", "DVRST", "Sxmpra"], "2020s",
         "a phonk house track at 150 BPM with distorted cowbell loops, heavy bass, and aggressive drift aesthetics"),
    ]

    for sub in _extra_other:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Other", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["varied"],
            defining_characteristics=[], typical_instruments=["varied"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Other"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.5),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More House subgenres ---
    _extra_house = [
        ("Soulful House", ["soulful"], (120, 128), "House centered on gospel and soul vocal performances with lush arrangements",
         (0.5, 0.75), (0.7, 0.9), ["Louie Vega", "Kenny Dope", "Dennis Ferrer"], "1990s",
         "a soulful house track at 124 BPM with a gospel-influenced vocal, warm pads, and a deep groove"),
        ("Minimal House", ["minimal-house"], (118, 128), "Stripped-down house focusing on repetition, subtlety, and micro-textural changes",
         (0.3, 0.55), (0.6, 0.8), ["Ricardo Villalobos", "Zip", "Luciano"], "2000s",
         "a minimal house track at 122 BPM with a sparse kick pattern, subtle clicks, and hypnotic textural evolution"),
        ("Jackin House", ["jackin", "jacking house"], (122, 130), "Raw, Chicago-influenced house with a jacking rhythm and funky samples",
         (0.6, 0.85), (0.75, 0.9), ["DJ Sneak", "Cajmere", "Gene Farris"], "1990s",
         "a jackin house track at 126 BPM with a raw jacking groove, funky vocal samples, and a stripped-down beat"),
        ("Piano House", ["piano-house"], (120, 130), "House music featuring prominent piano riffs and uplifting chord progressions",
         (0.6, 0.8), (0.7, 0.9), ["Robin S", "Robert Owens", "Black Box"], "1990s",
         "a piano house track at 125 BPM with uplifting piano chords, a four-on-the-floor kick, and a soulful vocal"),
        ("Hard House", ["hard-house"], (140, 155), "High-energy house with harder kicks, faster tempos, and rave influence",
         (0.8, 0.95), (0.6, 0.85), ["Lisa Lashes", "Tony De Vit", "BK"], "1990s",
         "a hard house track at 148 BPM with a pounding kick, rave stabs, and high-energy synth riffs"),
        ("Vocal House", ["vocal-house"], (120, 128), "House music centered on prominent, often diva-style vocal performances",
         (0.5, 0.8), (0.7, 0.9), ["Barbara Tucker", "Crystal Waters", "Ultra Nate"], "1990s",
         "a vocal house track at 124 BPM with a powerful diva vocal, lush pads, and a classic house groove"),
    ]

    for sub in _extra_house:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="House", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "Dorian mode"],
            defining_characteristics=[], typical_instruments=["TR-909", "synths", "Rhodes"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["House"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.15),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Techno subgenres ---
    _extra_techno = [
        ("Hypnotic Techno", ["hypnotic"], (125, 135), "Trance-inducing techno with repetitive loops and gradual textural shifts",
         (0.4, 0.7), (0.5, 0.75), ["Boris Brejcha", "Stephan Bodzin", "Mind Against"], "2010s",
         "a hypnotic techno track at 128 BPM with repetitive looping patterns, gradual textural shifts, and a trance-inducing groove"),
        ("Peak-Time Techno", ["peak time", "driving techno"], (130, 140), "High-energy techno designed for peak dancefloor moments",
         (0.7, 0.95), (0.6, 0.85), ["Amelie Lens", "Charlotte de Witte", "Enrico Sangiuliano"], "2010s",
         "a peak-time techno track at 135 BPM with a driving kick, energetic percussion, and an intense build"),
        ("Afro Techno", ["afro-techno"], (120, 135), "Techno incorporating African rhythmic patterns and organic percussion",
         (0.5, 0.8), (0.6, 0.85), ["Dibango", "Hyenah", "Aero Manyelo"], "2010s",
         "an Afro techno track at 126 BPM with African percussion, organic textures, and a deep techno groove"),
        ("Breakbeat Techno", ["broken techno", "break techno"], (128, 140), "Techno replacing the four-on-the-floor kick with breakbeat patterns",
         (0.6, 0.85), (0.5, 0.75), ["Blawan", "Karenn", "Objekt"], "2010s",
         "a breakbeat techno track at 133 BPM with broken drum patterns, industrial textures, and a dark groove"),
    ]

    for sub in _extra_techno:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Techno", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "atonal"],
            defining_characteristics=[], typical_instruments=["modular synth", "TR-909"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Techno"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.1),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More DnB subgenres ---
    _extra_dnb = [
        ("Halftime DnB", ["halftime", "half-time"], (160, 175), "DnB-tempo production with half-time feel, creating a hip-hop-influenced groove",
         (0.4, 0.7), (0.4, 0.65), ["Ivy Lab", "Alix Perez", "Halogenix"], "2010s",
         "a halftime DnB track at 170 BPM with a half-time feel, deep bass, and a hip-hop-influenced groove"),
        ("Rollers", ["roller DnB"], (172, 178), "Smooth, rolling DnB with continuous flowing breaks and deep bass",
         (0.5, 0.75), (0.5, 0.75), ["Calibre", "dBridge", "Marcus Intalex"], "2000s",
         "a roller DnB track at 174 BPM with continuous rolling breaks, a smooth bassline, and deep atmosphere"),
        ("Intelligent DnB", ["intelligent drum and bass", "atmospheric DnB"], (170, 178), "Cerebral, melodic DnB with lush pads and sophisticated arrangements",
         (0.4, 0.7), (0.4, 0.7), ["LTJ Bukem", "Makoto", "Peshay"], "1990s",
         "an intelligent DnB track at 174 BPM with lush atmospheric pads, sophisticated melodies, and rolling breaks"),
        ("Dancefloor DnB", ["dance DnB", "mainstream DnB"], (172, 178), "Accessible, high-energy DnB designed for broad dancefloor appeal",
         (0.7, 0.9), (0.6, 0.8), ["Sub Focus", "Dimension", "Chase & Status", "Sigma"], "2000s",
         "a dancefloor DnB track at 175 BPM with a big vocal hook, punchy drums, and a festival-ready drop"),
        ("Ragga Jungle", ["ragga DnB", "ragga jungle"], (160, 175), "Jungle/DnB with ragga and dancehall vocal samples and Caribbean influence",
         (0.6, 0.85), (0.6, 0.8), ["Congo Natty", "General Levy", "Shy FX"], "1990s",
         "a ragga jungle track at 170 BPM with chopped ragga vocals, Amen breaks, and a Caribbean-influenced bassline"),
    ]

    for sub in _extra_dnb:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Drum and Bass", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"],
            defining_characteristics=[], typical_instruments=["breakbeats", "sub-bass", "sampler"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Drum and Bass"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.1),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Trance subgenres ---
    _extra_trance = [
        ("Tech Trance", ["tech-trance"], (136, 145), "Trance blended with techno's darker energy and harder grooves",
         (0.7, 0.9), (0.6, 0.8), ["Simon Patterson", "Bryan Kearney", "John Askew"], "1990s",
         "a tech trance track at 140 BPM with a driving techno groove, acid bassline, and trance-style builds"),
        ("Balearic Trance", ["Balearic", "Ibiza trance"], (125, 138), "Sun-kissed, melodic trance influenced by Balearic/Ibiza club culture",
         (0.5, 0.75), (0.6, 0.8), ["Roger Shah", "Chicane", "Above & Beyond"], "1990s",
         "a Balearic trance track at 132 BPM with sun-kissed pads, a gentle build, and an Ibiza sunset atmosphere"),
        ("Nitzhonot", ["nitzho", "Israeli trance"], (140, 148), "Israeli psytrance variant with catchy melodies and Middle Eastern influence",
         (0.7, 0.9), (0.6, 0.8), ["Astrix", "Skazi", "Infected Mushroom"], "2000s",
         "a Nitzhonot track at 145 BPM with Middle Eastern melodies, a driving bassline, and psytrance energy"),
        ("Dream Trance", ["dream-trance"], (130, 140), "Melodic trance with dreamy pads, ethereal vocals, and a softer, romantic feel",
         (0.4, 0.7), (0.6, 0.8), ["Robert Miles", "Chicane", "BT"], "1990s",
         "a dream trance track at 135 BPM with dreamy piano, ethereal pads, and a romantic, uplifting melody"),
    ]

    for sub in _extra_trance:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Trance", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"],
            defining_characteristics=[], typical_instruments=["synths", "pads"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Trance"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.1),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Dubstep subgenres ---
    _extra_dubstep = [
        ("Melodic Dubstep", ["melodic-dubstep"], (140, 150), "Emotional dubstep with lush melodies, supersaw chords, and cinematic drops",
         (0.5, 0.85), (0.4, 0.65), ["Seven Lions", "Au5", "Dabin", "Said the Sky"], "2010s",
         "a melodic dubstep track at 150 BPM with emotional supersaw chords, a cinematic build, and a powerful bass drop"),
        ("Tearout", ["tearout dubstep"], (140, 155), "Extremely aggressive dubstep with complex, chaotic bass design",
         (0.85, 1.0), (0.3, 0.55), ["Svdden Death", "Marauda", "PhaseOne"], "2010s",
         "a tearout dubstep track at 150 BPM with chaotic, aggressive bass design and skull-crushing drops"),
        ("Colour Bass", ["color bass", "colour-bass"], (140, 170), "Melodic, sound-design-heavy bass music blending dubstep with future bass aesthetics",
         (0.5, 0.85), (0.4, 0.65), ["Chime", "Au5", "Ace Aura"], "2010s",
         "a colour bass track at 150 BPM with melodic sound design, lush chords, and intricate bass drops"),
    ]

    for sub in _extra_dubstep:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Dubstep", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"],
            defining_characteristics=[], typical_instruments=["Serum", "Massive", "sub-bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Dubstep"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.05),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Ambient subgenres ---
    _extra_ambient = [
        ("Ambient Dub", ["ambient-dub"], (70, 110), "Ambient music using dub production techniques — heavy delay, reverb, and echo",
         (0.1, 0.35), (0.2, 0.45), ["The Orb", "Higher Intelligence Agency", "Bluetech"], "1990s",
         "an ambient dub track with echo-drenched pads, deep reverb bass, and floating dub delay textures"),
        ("Psybient", ["psychedelic ambient", "psychill"], (80, 110), "Psychedelic ambient with organic textures, world music influence, and cosmic vibes",
         (0.1, 0.4), (0.3, 0.5), ["Shpongle", "Ott", "Carbon Based Lifeforms"], "1990s",
         "a psybient track at 95 BPM with psychedelic textures, world music samples, and a cosmic atmosphere"),
    ]

    for sub in _extra_ambient:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Ambient", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["suspended chords", "modal"],
            defining_characteristics=[], typical_instruments=["synthesizer", "delay", "reverb"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Ambient"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.1, 0.5),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Breakbeat subgenres ---
    _extra_breakbeat = [
        ("Broken Beat", ["bruk beat", "brokenbeat"], (110, 130), "Syncopated, jazz-influenced electronic music from West London",
         (0.4, 0.7), (0.6, 0.8), ["IG Culture", "Bugz in the Attic", "4hero", "Dego"], "2000s",
         "a broken beat track at 120 BPM with syncopated rhythms, jazz chords, and a West London electronic groove"),
        ("Acid Breaks", ["acid-breaks"], (120, 140), "Breakbeat music with 303 acid lines, combining breaks with acid house aesthetics",
         (0.6, 0.85), (0.6, 0.8), ["Ceephax Acid Crew", "Reso"], "1990s",
         "an acid breaks track at 130 BPM with chopped breakbeats, squelching 303 acid lines, and rave energy"),
    ]

    for sub in _extra_breakbeat:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Breakbeat", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys"],
            defining_characteristics=[], typical_instruments=["breakbeats", "sampler", "synths"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Breakbeat"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.15),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- More Hardcore Electronic subgenres ---
    _extra_hardcore = [
        ("UK Hardcore", ["UK happy hardcore"], (160, 180), "UK-flavored hardcore with bright synths, euphoric pianos, and breakbeats",
         (0.8, 0.95), (0.6, 0.85), ["Gammer", "Darren Styles", "Dougal & Gammer"], "1990s",
         "a UK hardcore track at 172 BPM with euphoric piano riffs, bright synths, and high-energy breakbeats"),
        ("Terrorcore", ["terrorcore music"], (200, 300), "Extreme, dark hardcore with horror themes and brutal tempos",
         (0.9, 1.0), (0.3, 0.5), ["Hellfish", "DJ Skinhead", "Leathernecks"], "1990s",
         "a terrorcore track at 250 BPM with horror samples, extreme distortion, and brutally fast kicks"),
        ("Breakcore", ["breakcore music"], (160, 300), "Chaotic, sample-heavy electronic music with shredded breakbeats and genre-blending",
         (0.8, 1.0), (0.3, 0.55), ["Venetian Snares", "Igorrr", "Bong-Ra", "Shitmat"], "1990s",
         "a breakcore track at 200 BPM with shredded Amen breaks, chaotic sample collage, and extreme tempo shifts"),
    ]

    for sub in _extra_hardcore:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Hardcore Electronic", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["minor keys", "atonal"],
            defining_characteristics=[], typical_instruments=["distorted synths", "sampler"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Hardcore Electronic"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.0, 0.05),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    # --- Final expansion to hit 500+ ---
    _final_expansion = [
        # More Folk/World
        ("Cumbia Digital", ["digital cumbia", "nu-cumbia"], (90, 115), "Electronic reinterpretation of cumbia with synths and digital production",
         (0.5, 0.75), (0.7, 0.85), ["Chancha Via Circuito", "El Remolón", "Nicola Cruz"], "2000s",
         "Folk/World", "a digital cumbia track at 100 BPM with electronic textures over a traditional cumbia rhythm"),
        ("Tuareg Blues", ["desert blues", "Saharan blues"], (80, 120), "Blues-influenced guitar music from the Sahara with hypnotic repetition",
         (0.4, 0.7), (0.4, 0.65), ["Tinariwen", "Mdou Moctar", "Bombino"], "1970s",
         "Folk/World", "a Tuareg blues track with hypnotic electric guitar riffs, desert atmosphere, and a driving groove"),
        ("Ethio-Groove", ["Ethiopian groove"], (90, 125), "Modern Ethiopian popular music blending funk, soul, and local traditions",
         (0.5, 0.75), (0.5, 0.75), ["Hailu Mergia", "Alemayehu Eshete"], "1960s",
         "Folk/World", "an Ethio-groove track with Ethiopian pentatonic melodies, funky organ, and a soulful groove"),
        # More Pop
        ("Soft Pop", ["soft-pop"], (80, 110), "Gentle, melodic pop with understated arrangements and warm production",
         (0.2, 0.45), (0.4, 0.6), ["Norah Jones", "Jack Johnson", "Colbie Caillat"], "2000s",
         "Pop", "a soft pop track with a gentle acoustic guitar, warm vocals, and a laid-back, breezy arrangement"),
        ("Alt-Pop", ["alternative pop", "alt pop"], (90, 130), "Pop blending mainstream accessibility with alternative/indie sensibilities",
         (0.4, 0.7), (0.5, 0.75), ["Lorde", "Billie Eilish", "Tame Impala", "Glass Animals"], "2010s",
         "Pop", "an alt-pop track with atmospheric production, indie sensibility, and a catchy alternative vocal hook"),
        # More Electronic
        ("Downtempo Electronica", ["chill electronica"], (80, 110), "Atmospheric, home-listening electronic with organic textures",
         (0.2, 0.45), (0.4, 0.6), ["Tycho", "Bonobo", "Emancipator", "Washed Out"], "2000s",
         "Electronic", "a downtempo electronica track at 95 BPM with warm analog textures, gentle beats, and atmospheric pads"),
        ("Trap EDM", ["festival trap", "EDM trap"], (130, 155), "Trap-influenced festival electronic music with heavy drops and 808 bass",
         (0.7, 0.95), (0.6, 0.8), ["RL Grime", "Baauer", "Flosstradamus", "Bro Safari"], "2010s",
         "Electronic", "a trap EDM track at 140 BPM with massive 808 bass drops, brass stabs, and festival-ready builds"),
        ("Midtempo Bass", ["midtempo"], (90, 115), "Dark, cinematic bass music at moderate tempos with heavy sound design",
         (0.5, 0.8), (0.4, 0.65), ["Rezz", "1788-L", "Apashe", "Gramatik"], "2010s",
         "Electronic", "a midtempo bass track at 100 BPM with dark cinematic textures, heavy bass design, and a hypnotic groove"),
        # More Jazz
        ("Afrobeat Jazz", ["Afrobeat-jazz fusion"], (100, 140), "Jazz incorporating West African Afrobeat rhythms and horn arrangements",
         (0.5, 0.8), (0.6, 0.85), ["Kokoroko", "Ezra Collective", "Sons of Kemet", "Nubya Garcia"], "2010s",
         "Jazz", "an Afrobeat jazz track at 120 BPM with West African rhythms, jazz improvisation, and a full horn section"),
        # More R&B
        ("Lo-fi R&B", ["lo-fi rnb"], (60, 90), "Warm, tape-saturated R&B with intimate production and bedroom aesthetics",
         (0.2, 0.45), (0.4, 0.6), ["Daniel Caesar", "Brent Faiyaz", "Snoh Aalegra"], "2010s",
         "R&B/Soul", "a lo-fi R&B track at 75 BPM with tape-saturated warmth, intimate vocals, and a mellow groove"),
        # More Rock
        ("Noise Pop Revival", ["noise-pop revival"], (110, 140), "Modern revival of noise pop with updated production and shoegaze influence",
         (0.5, 0.75), (0.4, 0.6), ["No Age", "Wavves", "Jay Som", "Snail Mail"], "2010s",
         "Rock", "a noise pop revival track with layers of fuzzy guitar, a catchy buried melody, and lo-fi production"),
        ("Midwest Emo Revival", ["emo revival"], (100, 150), "2010s revival of 90s midwest emo with twinkling guitars and emotional vocals",
         (0.4, 0.7), (0.3, 0.55), ["Modern Baseball", "The Hotelier", "Tiny Moving Parts", "Mom Jeans"], "2010s",
         "Rock", "an emo revival track with twinkling guitar arpeggios, earnest vocals, and a dynamic punk-influenced rhythm"),
        # More Hip-Hop
        ("Jersey Club Rap", ["Jersey club hip-hop"], (130, 145), "Jersey club beats with rap vocals, rapid samples, and dance energy",
         (0.7, 0.9), (0.7, 0.85), ["DJ Sliink", "Cookiee Kawaii", "Club Dangerous"], "2010s",
         "Hip-Hop/Rap", "a Jersey club rap track at 140 BPM with rapid-fire vocal chops, rap verses, and a frenetic club groove"),
        # More Latin
        ("Reggaeton Viejo", ["old school reggaeton", "underground reggaeton"], (85, 95), "Original underground reggaeton with raw production and Dem Bow riddim",
         (0.6, 0.8), (0.7, 0.85), ["Tego Calderon", "Hector El Father", "Ivy Queen"], "1990s",
         "Latin", "a reggaeton viejo track at 90 BPM with a raw Dem Bow riddim, aggressive vocal delivery, and underground energy"),
        # More Country
        ("Cosmic Country", ["cosmic american", "cosmic-country"], (80, 120), "Psychedelic-influenced country with spacey production and cosmic themes",
         (0.3, 0.6), (0.3, 0.55), ["Gram Parsons", "Flying Burrito Brothers", "Sturgill Simpson"], "1970s",
         "Country", "a cosmic country track with spacey pedal steel, psychedelic effects, and a laid-back country groove"),
        # More Classical
        ("Postminimalism", ["post-minimalism", "totalism"], (50, 140), "Evolution beyond minimalism with greater harmonic complexity and emotional range",
         (0.2, 0.6), (0.2, 0.4), ["John Adams", "Steve Reich", "David Lang", "Julia Wolfe"], "1980s",
         "Classical", "a postminimalist composition with repeating patterns, evolving harmonic complexity, and emotional depth"),
        # More Other
        ("Synthwave Cyberpunk", ["cyberpunk music"], (90, 130), "Dark, dystopian synthwave with cyberpunk aesthetics and heavy bass",
         (0.5, 0.8), (0.4, 0.65), ["Daniel Deluxe", "Mega Drive", "Power Glove"], "2010s",
         "Other", "a cyberpunk synthwave track with dark dystopian synths, heavy bass, and a futuristic noir atmosphere"),
        ("Nerdcore", ["nerd rap", "nerdcore hip-hop"], (80, 120), "Hip-hop with geek culture themes — video games, science, technology",
         (0.4, 0.7), (0.5, 0.7), ["MC Frontalot", "MC Chris", "Mega Ran", "Schaffer the Darklord"], "2000s",
         "Other", "a nerdcore hip-hop track with geek culture references, a boom-bap beat, and witty lyrical delivery"),
    ]

    for sub in _final_expansion:
        # Tuple: (name, aliases, bpm, prod_style, energy, dance, artists, era, parent, clap)
        parent_name = sub[8]
        genres.append(Genre(
            name=sub[0], id=_id(), parent=parent_name, aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["varied"],
            defining_characteristics=[], typical_instruments=["varied"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=[parent_name], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.05, 0.5),
            famous_artists=sub[6],
            clap_descriptions=[sub[9]],
        ))

    # --- Blues (new top-level) ---
    genres.append(Genre(
        name="Blues", id=_id(), parent=None,
        aliases=["blues music"], bpm_range=(60, 140),
        key_tendencies=["blues scale", "minor pentatonic", "dominant 7th chords"],
        defining_characteristics=["12-bar form", "blue notes", "call-and-response",
                                  "emotional vocal delivery", "guitar bends"],
        typical_instruments=["electric guitar", "harmonica", "bass", "drums", "piano"],
        production_style="Raw, emotion-driven production centered on guitar tone, vocal expression, and the 12-bar blues form",
        era_of_origin="1890s", parent_genres=["African American spirituals", "Work songs"],
        sibling_genres=["Jazz", "R&B/Soul", "Rock"],
        energy_range=(0.3, 0.75), danceability_range=(0.3, 0.65), acousticness_range=(0.15, 0.7),
        famous_artists=["B.B. King", "Muddy Waters", "Robert Johnson", "Howlin' Wolf", "John Lee Hooker"],
        clap_descriptions=[
            "a blues track with an expressive electric guitar, a 12-bar form, and a raw, emotional vocal",
            "a slow blues shuffle with stinging guitar bends, Hammond organ, and a deep groove",
        ],
    ))

    _blues_subs = [
        ("Delta Blues", ["Mississippi blues"], (60, 100), "Raw, acoustic blues from the Mississippi Delta with slide guitar and field-holler vocals",
         (0.3, 0.6), (0.3, 0.5), ["Robert Johnson", "Son House", "Charley Patton"], "1920s",
         "a Delta blues track with raw acoustic slide guitar, a gravelly vocal, and a hypnotic one-chord drone"),
        ("Chicago Blues", ["electric blues"], (80, 130), "Amplified, band-oriented blues from Chicago with electric guitar and harmonica",
         (0.5, 0.75), (0.4, 0.65), ["Muddy Waters", "Howlin' Wolf", "Buddy Guy", "Little Walter"], "1940s",
         "a Chicago blues track with an amplified guitar, harmonica, and a driving rhythm section"),
        ("Texas Blues", ["Texas-style blues"], (80, 130), "Energetic, guitar-driven blues from Texas with a horn-influenced big-band feel",
         (0.5, 0.8), (0.4, 0.65), ["Stevie Ray Vaughan", "T-Bone Walker", "Albert Collins"], "1920s",
         "a Texas blues track with fiery guitar licks, a swinging rhythm, and a horn section"),
        ("British Blues", ["UK blues"], (80, 130), "Blues revival led by British musicians, bridging American blues and rock",
         (0.5, 0.8), (0.4, 0.6), ["John Mayall", "Eric Clapton", "Fleetwood Mac"], "1960s",
         "a British blues track with a classic Les Paul guitar tone, a steady shuffle beat, and emotional vocal"),
        ("Jump Blues", ["jump"], (130, 180), "Uptempo, danceable blues with a big-band swing feel and energetic performance",
         (0.6, 0.85), (0.7, 0.85), ["Louis Jordan", "Big Joe Turner", "Wynonie Harris"], "1940s",
         "a jump blues track with a swinging horn section, an uptempo shuffle, and an energetic vocal"),
        ("Piedmont Blues", ["East Coast blues"], (80, 120), "Fingerpicking-oriented acoustic blues from the Southeastern US",
         (0.3, 0.55), (0.4, 0.6), ["Blind Blake", "Reverend Gary Davis", "Elizabeth Cotten"], "1920s",
         "a Piedmont blues track with intricate fingerpicking, a ragtime influence, and a gentle acoustic groove"),
        ("Swamp Blues", ["Louisiana blues"], (70, 110), "Heavy, murky blues from Louisiana with a slow, swampy feel and echo effects",
         (0.3, 0.6), (0.3, 0.55), ["Slim Harpo", "Lightnin' Slim", "Lazy Lester"], "1950s",
         "a swamp blues track with a murky guitar tone, heavy echo, and a slow, humid Louisiana groove"),
        ("Hill Country Blues", ["North Mississippi blues"], (70, 110), "Hypnotic, drone-based blues from North Mississippi with trance-like repetition",
         (0.3, 0.6), (0.4, 0.6), ["R.L. Burnside", "Junior Kimbrough", "Mississippi Fred McDowell"], "1920s",
         "a hill country blues track with a hypnotic one-chord drone, trance-like repetition, and raw electric guitar"),
        ("Soul Blues", ["soul-blues"], (70, 110), "Blues blending with soul music for a smoother, more vocally polished sound",
         (0.4, 0.7), (0.5, 0.7), ["Bobby Bland", "Z.Z. Hill", "Johnnie Taylor"], "1960s",
         "a soul blues track with a smooth vocal, horn section, and a groove blending blues and soul"),
        ("Modern Blues", ["contemporary blues"], (70, 130), "Current blues drawing on traditional forms with modern production and diverse influences",
         (0.4, 0.75), (0.4, 0.65), ["Gary Clark Jr.", "Joe Bonamassa", "Christone Kingfish Ingram"], "2000s",
         "a modern blues track with a fiery guitar solo, contemporary production, and a classic blues form"),
    ]

    for sub in _blues_subs:
        genres.append(Genre(
            name=sub[0], id=_id(), parent="Blues", aliases=sub[1], bpm_range=sub[2],
            key_tendencies=["blues scale", "minor pentatonic"],
            defining_characteristics=[], typical_instruments=["electric guitar", "harmonica", "bass"],
            production_style=sub[3], era_of_origin=sub[7],
            parent_genres=["Blues"], sibling_genres=[],
            energy_range=sub[4], danceability_range=sub[5], acousticness_range=(0.15, 0.7),
            famous_artists=sub[6], clap_descriptions=[sub[8]],
        ))

    return genres


# ---------------------------------------------------------------------------
# Module-level registry (built once on import)
# ---------------------------------------------------------------------------

_ALL_GENRES: List[Genre] = _build_genres()

# Index by lowercase name and aliases for fast lookup
_NAME_INDEX: Dict[str, Genre] = {}
for _g in _ALL_GENRES:
    _NAME_INDEX[_g.name.lower()] = _g
    for _a in _g.aliases:
        _NAME_INDEX[_a.lower()] = _g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_genres() -> List[Genre]:
    """Return the full list of Genre instances."""
    return list(_ALL_GENRES)


def get_genre_hierarchy() -> Dict[str, List[str]]:
    """Return a dict mapping each parent genre name to a list of child genre names."""
    hierarchy: Dict[str, List[str]] = {}
    for g in _ALL_GENRES:
        if g.parent is not None:
            hierarchy.setdefault(g.parent, []).append(g.name)
    return hierarchy


def get_genre_by_name(name: str) -> Optional[Genre]:
    """Look up a genre by exact name or alias (case-insensitive).

    Falls back to fuzzy matching (ratio >= 0.8) if no exact match is found.
    """
    key = name.strip().lower()
    exact = _NAME_INDEX.get(key)
    if exact is not None:
        return exact

    # Fuzzy fallback
    best_match: Optional[Genre] = None
    best_score = 0.0
    for stored_name, genre in _NAME_INDEX.items():
        score = SequenceMatcher(None, key, stored_name).ratio()
        if score > best_score:
            best_score = score
            best_match = genre
    if best_score >= 0.8:
        return best_match
    return None


def get_genre_labels() -> List[Tuple[int, str]]:
    """Return (id, name) tuples for every genre — suitable for a classification head."""
    return [(g.id, g.name) for g in _ALL_GENRES]


def get_top_level_genres() -> List[str]:
    """Return the names of all top-level (parent=None) genres."""
    return [g.name for g in _ALL_GENRES if g.parent is None]


def get_subgenres(parent: str) -> List[Genre]:
    """Return all genres whose parent matches *parent* (case-insensitive)."""
    key = parent.strip().lower()
    return [g for g in _ALL_GENRES if g.parent is not None and g.parent.lower() == key]


def generate_clap_descriptions() -> List[Tuple[str, str]]:
    """Return (text_description, genre_label) tuples for CLAP text-audio alignment.

    Each genre contributes 2-3 descriptions.  If the genre has fewer
    pre-written descriptions we auto-generate extras from its metadata.
    """
    results: List[Tuple[str, str]] = []
    for g in _ALL_GENRES:
        descs = list(g.clap_descriptions)  # copy

        # Auto-generate additional descriptions to reach at least 2
        while len(descs) < 2:
            bpm_mid = (g.bpm_range[0] + g.bpm_range[1]) // 2
            instruments = ", ".join(g.typical_instruments[:3]) if g.typical_instruments else "synthesizers"
            energy_word = "high-energy" if g.energy_range[1] > 0.7 else "mellow"
            desc = (
                f"a {energy_word} {g.name.lower()} track at {bpm_mid} BPM "
                f"with {instruments} and {g.production_style.split(',')[0].lower().strip()}"
            )
            descs.append(desc)

        for d in descs:
            results.append((d, g.name))

    return results


# ---------------------------------------------------------------------------
# Convenience: genre count sanity check
# ---------------------------------------------------------------------------

def genre_count() -> int:
    """Return the total number of genres in the taxonomy."""
    return len(_ALL_GENRES)


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Total genres: {genre_count()}")
    print(f"Top-level genres: {get_top_level_genres()}")
    hierarchy = get_genre_hierarchy()
    for parent, children in sorted(hierarchy.items()):
        print(f"  {parent}: {len(children)} subgenres")
    print(f"CLAP descriptions: {len(generate_clap_descriptions())} total")

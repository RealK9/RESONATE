"""
RESONATE Production Model (RPM) — Production Techniques Knowledge Base

Comprehensive corpus of audio production techniques used to:
1. Generate CLAP text descriptions for text-audio alignment during training
2. Help the model understand production quality and techniques
3. Inform the perceptual quality assessment head

Each technique, mix position, and quality marker is structured as a dataclass
so downstream code can iterate, filter, and compose training descriptions
programmatically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProductionTechnique:
    name: str
    category: str  # dynamics | eq | spatial | time_based | distortion | modulation | stereo | creative | mastering
    description: str
    parameters: list[str]
    sonic_effect: str
    when_to_use: list[str]
    genre_associations: list[str]
    quality_markers: dict[str, str]
    famous_examples: list[str]


@dataclass
class MixPosition:
    instrument: str
    frequency_home: str
    typical_panning: str
    depth_position: str
    common_processing: list[str]
    genre_variations: dict[str, str]


@dataclass
class QualityMarker:
    marker: str
    good_indicator: str
    bad_indicator: str
    frequency_relevance: str
    importance: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Dynamics Processing (~15 techniques)
# ---------------------------------------------------------------------------

_DYNAMICS: list[ProductionTechnique] = [
    ProductionTechnique(
        name="VCA Compression",
        category="dynamics",
        description="Voltage-controlled amplifier compression offering fast, precise gain reduction with a clean, transparent character.",
        parameters=["threshold", "ratio", "attack", "release", "knee", "makeup gain"],
        sonic_effect="Tight, punchy dynamics control that preserves transient detail while reducing dynamic range.",
        when_to_use=["Drum bus taming", "Transparent vocal leveling", "Mix bus glue with moderate settings"],
        genre_associations=["pop", "rock", "electronic", "hip-hop"],
        quality_markers={
            "good": "Transparent gain reduction that tightens performance without audible pumping",
            "bad": "Over-compressed, lifeless signal with obvious gain-riding artifacts",
        },
        famous_examples=["SSL G-Bus compressor on mix bus", "dbx 160 on snare drum"],
    ),
    ProductionTechnique(
        name="FET Compression",
        category="dynamics",
        description="Field-effect transistor compression known for aggressive, colorful character with very fast attack times.",
        parameters=["input gain", "output gain", "attack", "release", "ratio"],
        sonic_effect="Aggressive, exciting compression that adds harmonic bite and energy to the signal.",
        when_to_use=["Vocal attitude and presence", "Drum room mic crushing", "Parallel aggression on guitars"],
        genre_associations=["rock", "punk", "metal", "indie", "hip-hop"],
        quality_markers={
            "good": "Energetic, forward sound with musical distortion character adding excitement",
            "bad": "Harsh, gritty distortion from overdriven input with uncontrolled transients",
        },
        famous_examples=["1176 on lead vocals", "1176 all-buttons-in on drum room"],
    ),
    ProductionTechnique(
        name="Optical Compression",
        category="dynamics",
        description="Light-dependent resistor compression with naturally smooth, program-dependent attack and release curves.",
        parameters=["peak reduction", "gain", "attack (fixed/program)", "release (fixed/program)"],
        sonic_effect="Smooth, gentle compression that feels natural and musical with a warm, rounded character.",
        when_to_use=["Vocal smoothing", "Bass guitar leveling", "Acoustic instrument dynamics"],
        genre_associations=["jazz", "soul", "R&B", "folk", "singer-songwriter"],
        quality_markers={
            "good": "Invisible gain riding that makes performances feel polished without sounding processed",
            "bad": "Sluggish response missing transients or failing to control peaks adequately",
        },
        famous_examples=["LA-2A on vocals", "LA-3A on bass guitar"],
    ),
    ProductionTechnique(
        name="Tube Compression",
        category="dynamics",
        description="Vacuum tube variable-mu compression providing warm, harmonically rich gain reduction.",
        parameters=["threshold", "ratio (variable-mu)", "attack", "release", "makeup gain"],
        sonic_effect="Warm, thick compression with added even-order harmonics that glues program material together.",
        when_to_use=["Mix bus warmth and cohesion", "Mastering chain smoothing", "Vocal richness"],
        genre_associations=["jazz", "soul", "classic rock", "R&B", "Motown"],
        quality_markers={
            "good": "Rich, cohesive sound with subtle harmonic warmth binding the mix",
            "bad": "Muddy, indistinct sound from excessive harmonic buildup and slow transient response",
        },
        famous_examples=["Fairchild 670 on Beatles mixes", "Manley Variable Mu on mastering bus"],
    ),
    ProductionTechnique(
        name="Multiband Compression",
        category="dynamics",
        description="Independent compression across multiple frequency bands allowing targeted dynamic control.",
        parameters=["crossover frequencies", "threshold per band", "ratio per band", "attack per band", "release per band"],
        sonic_effect="Frequency-selective dynamics control that can tame problem areas without affecting the full spectrum.",
        when_to_use=["Taming boomy low-end without dulling highs", "De-essing with precision", "Mastering frequency balance"],
        genre_associations=["electronic", "pop", "hip-hop", "mastering"],
        quality_markers={
            "good": "Transparent frequency-dependent control that balances the spectrum naturally",
            "bad": "Phasey, unnatural crossover artifacts with pumping in individual bands",
        },
        famous_examples=["Waves C4 on vocals", "FabFilter Pro-MB on mix bus"],
    ),
    ProductionTechnique(
        name="Parallel Compression",
        category="dynamics",
        description="Blending a heavily compressed signal with the dry original to preserve transients while adding body and sustain.",
        parameters=["blend/mix", "threshold (wet path)", "ratio (wet path)", "attack", "release"],
        sonic_effect="Best of both worlds: natural transients and dynamics on top, with dense body and sustain underneath.",
        when_to_use=["Drum bus power and punch", "Vocal presence without squashing", "Full-mix density in mastering"],
        genre_associations=["rock", "pop", "metal", "hip-hop", "electronic"],
        quality_markers={
            "good": "Punchy, full sound retaining natural dynamics while adding weight and density",
            "bad": "Phase issues between wet and dry, or overly crushed blend losing clarity",
        },
        famous_examples=["New York compression on drums", "Parallel 1176 on vocals"],
    ),
    ProductionTechnique(
        name="Sidechain Compression",
        category="dynamics",
        description="Compression triggered by an external signal, allowing one element to duck in volume when another plays.",
        parameters=["sidechain source", "threshold", "ratio", "attack", "release", "sidechain filter"],
        sonic_effect="Rhythmic pumping effect or transparent frequency carving to create space between competing elements.",
        when_to_use=["Kick-bass relationship clarity", "EDM pumping effect", "Vocal ducking of instruments"],
        genre_associations=["EDM", "house", "techno", "pop", "hip-hop"],
        quality_markers={
            "good": "Musical pumping that enhances groove, or transparent ducking creating clean separation",
            "bad": "Unmusical, jerky pumping with audible breathing artifacts",
        },
        famous_examples=["Daft Punk sidechain pumping", "Eric Prydz bass ducking"],
    ),
    ProductionTechnique(
        name="Bus/Glue Compression",
        category="dynamics",
        description="Gentle mix-bus compression that binds individual tracks into a cohesive whole.",
        parameters=["threshold", "ratio (typically 2:1-4:1)", "attack (slow)", "release (auto/program)", "makeup gain"],
        sonic_effect="Subtle cohesion that makes separate tracks feel like they belong together as a single performance.",
        when_to_use=["Mix bus final polish", "Drum bus cohesion", "Submix glue"],
        genre_associations=["pop", "rock", "R&B", "any genre during mixing"],
        quality_markers={
            "good": "Mix elements feel unified with subtle, musical gain reduction of 1-3 dB",
            "bad": "Squashed, flat mix with lost dynamics and obvious pumping on transients",
        },
        famous_examples=["SSL G-Bus on mix bus", "API 2500 on drum bus"],
    ),
    ProductionTechnique(
        name="Brickwall Limiting",
        category="dynamics",
        description="Hard ceiling limiting that prevents signal from exceeding a set output level, using infinite ratio.",
        parameters=["ceiling", "threshold/input gain", "release", "lookahead"],
        sonic_effect="Absolute peak control allowing maximum loudness while preventing digital clipping.",
        when_to_use=["Final mastering stage", "Loudness maximizing", "Broadcast compliance"],
        genre_associations=["all genres (mastering)", "EDM", "pop", "hip-hop"],
        quality_markers={
            "good": "Transparent loudness increase with minimal audible distortion or pumping",
            "bad": "Crushed, distorted sound with inter-sample peaks, loss of punch and clarity",
        },
        famous_examples=["Waves L2 in the loudness wars", "FabFilter Pro-L 2 on modern masters"],
    ),
    ProductionTechnique(
        name="Soft Clipping",
        category="dynamics",
        description="Gentle waveform clipping that rounds off peaks with saturation rather than hard digital clipping.",
        parameters=["ceiling", "drive", "knee", "saturation type"],
        sonic_effect="Warm peak reduction that adds subtle harmonic richness while controlling transients.",
        when_to_use=["Taming drum transients before a limiter", "Adding analog warmth", "Pre-limiter peak control"],
        genre_associations=["hip-hop", "electronic", "pop", "lo-fi"],
        quality_markers={
            "good": "Natural-sounding peak control with pleasant harmonic coloration",
            "bad": "Audible, harsh clipping artifacts that sound digital and unpleasant",
        },
        famous_examples=["Kazrog KClip on drums", "StandardCLIP on mix bus"],
    ),
    ProductionTechnique(
        name="Noise Gate",
        category="dynamics",
        description="Dynamics processor that silences signal below a threshold, removing unwanted noise between wanted audio.",
        parameters=["threshold", "attack", "hold", "release", "range", "sidechain filter"],
        sonic_effect="Clean silence between notes or hits, removing bleed, hum, or ambient noise.",
        when_to_use=["Drum mic bleed reduction", "Removing amp hum between guitar phrases", "Cleaning up vocal tracks"],
        genre_associations=["rock", "metal", "pop", "live recording"],
        quality_markers={
            "good": "Natural-sounding silence between passages with no audible gating chatter",
            "bad": "Choppy, unnatural cutoffs with audible gating on sustained notes or breaths",
        },
        famous_examples=["Gated drums in 1980s pop", "Tight metal rhythm guitar gating"],
    ),
    ProductionTechnique(
        name="Expander",
        category="dynamics",
        description="Gentler alternative to gating that reduces (rather than silences) signal below threshold.",
        parameters=["threshold", "ratio", "attack", "release", "range"],
        sonic_effect="Reduced background noise and bleed while maintaining a more natural decay than hard gating.",
        when_to_use=["Subtle bleed reduction on drums", "Reducing room noise on vocals", "Gentle noise reduction"],
        genre_associations=["jazz", "classical", "acoustic", "live recording"],
        quality_markers={
            "good": "Natural-sounding noise reduction that preserves room ambience and tail of notes",
            "bad": "Audible breathing or pumping as the expander opens and closes",
        },
        famous_examples=["Downward expansion on orchestral recordings", "Gentle vocal cleanup"],
    ),
    ProductionTechnique(
        name="Transient Shaping",
        category="dynamics",
        description="Threshold-independent processor that enhances or reduces the attack and sustain portions of a sound.",
        parameters=["attack", "sustain", "output gain"],
        sonic_effect="Direct control over punchiness (attack) and body/ring (sustain) independent of overall level.",
        when_to_use=["Adding snap to drums", "Reducing room sound on overheads", "Making synths punchier"],
        genre_associations=["electronic", "pop", "hip-hop", "rock"],
        quality_markers={
            "good": "Natural-sounding transient enhancement that adds punch without artifacts",
            "bad": "Clicky, unnatural attack emphasis or hollow, lifeless sustain reduction",
        },
        famous_examples=["SPL Transient Designer on snare", "Transient shaping on electronic kicks"],
    ),
    ProductionTechnique(
        name="De-essing",
        category="dynamics",
        description="Frequency-selective compression targeting sibilant frequencies (typically 4-10 kHz) in vocal recordings.",
        parameters=["frequency", "threshold", "range", "mode (split-band/wideband)", "listen/monitor"],
        sonic_effect="Reduced harshness on 's', 'sh', and 't' sounds while preserving vocal brightness and presence.",
        when_to_use=["Vocal sibilance control", "Cymbal harshness reduction", "Acoustic guitar string squeak taming"],
        genre_associations=["pop", "R&B", "any vocal-centric genre"],
        quality_markers={
            "good": "Smooth, natural sibilance control that keeps vocals bright but not harsh",
            "bad": "Lisping, dull vocals from over-processing or incorrect frequency targeting",
        },
        famous_examples=["Waves DeEsser on pop vocals", "FabFilter Pro-DS on broadcast"],
    ),
]


# ---------------------------------------------------------------------------
# EQ & Filtering (~15 techniques)
# ---------------------------------------------------------------------------

_EQ: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Parametric EQ",
        category="eq",
        description="Fully adjustable equalizer with control over frequency, gain, and bandwidth (Q) for each band.",
        parameters=["frequency", "gain", "Q/bandwidth", "band type", "number of bands"],
        sonic_effect="Precise tonal sculpting allowing surgical cuts or broad musical boosts at any frequency.",
        when_to_use=["Surgical problem frequency removal", "Tonal shaping of any source", "Mix balancing"],
        genre_associations=["all genres"],
        quality_markers={
            "good": "Natural, musical tonal shaping that enhances the source without ringing or phase issues",
            "bad": "Narrow, resonant boosts creating ringing artifacts or excessive phase shift",
        },
        famous_examples=["Pultec EQP-1A broad boosts", "FabFilter Pro-Q 3 surgical cuts"],
    ),
    ProductionTechnique(
        name="Graphic EQ",
        category="eq",
        description="Fixed-frequency band equalizer with slider controls at predetermined frequency centers.",
        parameters=["band gains (fixed frequencies)", "number of bands (typically 10 or 31)"],
        sonic_effect="Quick, visual tonal adjustment useful for broad shaping and live sound correction.",
        when_to_use=["Live sound room correction", "Quick tonal shaping", "Monitor tuning"],
        genre_associations=["live sound", "broadcast"],
        quality_markers={
            "good": "Smooth, broad tonal correction that addresses room problems naturally",
            "bad": "Comb-filter-like artifacts from excessive adjacent band adjustments",
        },
        famous_examples=["API 560 on guitar amps", "31-band graphic on PA systems"],
    ),
    ProductionTechnique(
        name="Dynamic EQ",
        category="eq",
        description="EQ bands that activate only when signal exceeds a threshold at the target frequency.",
        parameters=["frequency", "gain", "Q", "threshold", "attack", "release", "ratio"],
        sonic_effect="Frequency-selective compression that only acts when problem frequencies become excessive.",
        when_to_use=["Taming resonant notes on bass", "Controlling vocal harshness on loud passages", "Frequency-dependent dynamics"],
        genre_associations=["pop", "mastering", "broadcast"],
        quality_markers={
            "good": "Transparent, adaptive frequency control that only acts when needed",
            "bad": "Audible pumping or unnatural tonal shifts as bands engage and release",
        },
        famous_examples=["TDR Nova on vocal resonances", "FabFilter Pro-Q 3 dynamic bands"],
    ),
    ProductionTechnique(
        name="High Shelf EQ",
        category="eq",
        description="Boosts or cuts all frequencies above a chosen corner frequency with a smooth shelf shape.",
        parameters=["frequency", "gain", "slope/Q"],
        sonic_effect="Broad brightening or darkening of the top end, affecting air and presence.",
        when_to_use=["Adding air and sparkle to a mix", "Darkening overly bright sources", "Vocal presence enhancement"],
        genre_associations=["all genres"],
        quality_markers={
            "good": "Smooth, musical high-frequency lift that adds openness without harshness",
            "bad": "Brittle, harsh top end from excessive boosting or too steep a slope",
        },
        famous_examples=["Pultec 10 kHz air shelf", "Neve 1073 high shelf on overheads"],
    ),
    ProductionTechnique(
        name="Low Shelf EQ",
        category="eq",
        description="Boosts or cuts all frequencies below a chosen corner frequency with a smooth shelf shape.",
        parameters=["frequency", "gain", "slope/Q"],
        sonic_effect="Broad thickening or thinning of the low end, affecting warmth and weight.",
        when_to_use=["Adding warmth and body", "Reducing boominess", "Bass enhancement"],
        genre_associations=["all genres"],
        quality_markers={
            "good": "Warm, controlled low-end enhancement that adds weight without muddiness",
            "bad": "Boomy, undefined low end that masks midrange detail",
        },
        famous_examples=["Pultec low shelf boost-and-cut trick", "API 550A low shelf on kick"],
    ),
    ProductionTechnique(
        name="High-Pass Filter",
        category="eq",
        description="Removes all frequencies below a cutoff point, cleaning up unwanted low-end rumble and mud.",
        parameters=["cutoff frequency", "slope (dB/octave)", "resonance"],
        sonic_effect="Cleaner low end with reduced rumble, proximity effect, and sub-bass buildup.",
        when_to_use=["Removing mic rumble", "Cleaning non-bass instruments", "Reducing mud in dense mixes"],
        genre_associations=["all genres"],
        quality_markers={
            "good": "Clean, open mix with each element occupying its proper frequency range",
            "bad": "Thin, gutted sound from over-filtering or audible resonant peak at cutoff",
        },
        famous_examples=["HPF on every non-bass channel", "Steep HPF on overheads for tight low end"],
    ),
    ProductionTechnique(
        name="Low-Pass Filter",
        category="eq",
        description="Removes all frequencies above a cutoff point, darkening the sound and removing harsh high frequencies.",
        parameters=["cutoff frequency", "slope (dB/octave)", "resonance"],
        sonic_effect="Darker, warmer sound with reduced brightness, harshness, and high-frequency noise.",
        when_to_use=["Creating distance and depth", "Taming harsh digital sources", "Lo-fi and vintage effects"],
        genre_associations=["lo-fi", "ambient", "electronic", "hip-hop"],
        quality_markers={
            "good": "Warm, distant character that serves the mix context without sounding muffled",
            "bad": "Dull, lifeless sound that buries the element and removes essential presence",
        },
        famous_examples=["LPF sweeps in EDM builds", "Lo-fi hip-hop filtered samples"],
    ),
    ProductionTechnique(
        name="Band-Pass Filter",
        category="eq",
        description="Passes only a band of frequencies, cutting both lows and highs outside the band.",
        parameters=["center frequency", "bandwidth/Q"],
        sonic_effect="Focused, telephone-like or radio-like effect isolating a narrow frequency range.",
        when_to_use=["Telephone/radio vocal effect", "Isolating midrange character", "Creative sound design"],
        genre_associations=["hip-hop", "electronic", "pop", "indie"],
        quality_markers={
            "good": "Characterful frequency focus that serves the arrangement and feels intentional",
            "bad": "Thin, boxy sound that loses too much musical information",
        },
        famous_examples=["Filtered vocal intros in pop", "Lo-fi sample processing in hip-hop"],
    ),
    ProductionTechnique(
        name="Notch Filter",
        category="eq",
        description="Very narrow cut at a specific frequency, used to remove resonances or feedback without affecting surrounding frequencies.",
        parameters=["frequency", "depth", "Q (very high)"],
        sonic_effect="Surgical removal of a single problematic frequency with minimal impact on surrounding content.",
        when_to_use=["Feedback elimination in live sound", "Removing room resonances", "Eliminating 50/60 Hz hum"],
        genre_associations=["live sound", "broadcast", "film"],
        quality_markers={
            "good": "Inaudible correction that removes the problem without tonal side effects",
            "bad": "Audible hole in the frequency spectrum from too wide or too deep a notch",
        },
        famous_examples=["Live feedback suppression", "Hum removal on guitar amps"],
    ),
    ProductionTechnique(
        name="Tilt EQ",
        category="eq",
        description="Single-knob EQ that pivots the entire frequency spectrum around a center point, brightening or darkening.",
        parameters=["tilt amount", "center frequency"],
        sonic_effect="Broad, gentle tonal shift that brightens or warms the overall character in one move.",
        when_to_use=["Quick overall tonal adjustment", "Subtle mastering correction", "Fast mix rebalancing"],
        genre_associations=["all genres", "mastering"],
        quality_markers={
            "good": "Subtle, natural tonal rebalancing that shifts the character without artifacts",
            "bad": "Excessive tilt creating unbalanced, lopsided frequency response",
        },
        famous_examples=["Tonelux Tilt on mix bus", "Brainworx bx_digital tilt in mastering"],
    ),
    ProductionTechnique(
        name="Linear Phase EQ",
        category="eq",
        description="EQ that introduces no phase shift, preserving transient shape at the cost of latency and pre-ringing.",
        parameters=["frequency", "gain", "Q", "phase mode"],
        sonic_effect="Phase-coherent frequency shaping ideal for mastering and parallel processing.",
        when_to_use=["Mastering EQ", "Parallel EQ processing", "Mid/side work where phase matters"],
        genre_associations=["mastering", "classical", "acoustic"],
        quality_markers={
            "good": "Clean, phase-accurate frequency adjustment with no transient smearing",
            "bad": "Audible pre-ringing artifacts on transient-heavy material",
        },
        famous_examples=["FabFilter Pro-Q 3 in linear phase mode", "Waves Linear Phase EQ"],
    ),
    ProductionTechnique(
        name="Mid/Side EQ",
        category="eq",
        description="EQ processing applied independently to the center (mid) and sides of a stereo signal.",
        parameters=["frequency", "gain", "Q", "mid/side mode"],
        sonic_effect="Independent tonal control of center vs. side content for width and focus manipulation.",
        when_to_use=["Widening stereo image", "Tightening bass in the center", "Adding air to sides only"],
        genre_associations=["mastering", "electronic", "pop"],
        quality_markers={
            "good": "Enhanced stereo width and focus with natural-sounding separation",
            "bad": "Mono compatibility issues or unnatural stereo image from excessive M/S processing",
        },
        famous_examples=["Brainworx bx_digital in mastering", "Mid/side bass tightening"],
    ),
    ProductionTechnique(
        name="Resonant Filter",
        category="eq",
        description="Filter with adjustable resonance peak at the cutoff frequency, fundamental to subtractive synthesis.",
        parameters=["cutoff frequency", "resonance/Q", "filter type (LP/HP/BP)", "drive"],
        sonic_effect="Characterful filtering with an emphasized peak that can self-oscillate at high resonance.",
        when_to_use=["Synth sound design", "Filter sweeps in electronic music", "Creative effect processing"],
        genre_associations=["electronic", "techno", "house", "synth-pop", "ambient"],
        quality_markers={
            "good": "Musical, characterful filtering that enhances the synth sound and creates movement",
            "bad": "Piercing, painful resonance peaks or unstable self-oscillation",
        },
        famous_examples=["Moog ladder filter sweeps", "Roland TB-303 acid bassline filter"],
    ),
    ProductionTechnique(
        name="Formant Filter",
        category="eq",
        description="Filter shaped to emulate vowel sounds of the human vocal tract, creating speech-like resonances.",
        parameters=["formant shape (vowel)", "frequency shift", "resonance", "morph speed"],
        sonic_effect="Vowel-like tonal character imposed on any source, creating talking or singing textures.",
        when_to_use=["Vocal-like synth sounds", "Formant shifting on vocals", "Creative sound design"],
        genre_associations=["electronic", "experimental", "pop", "hip-hop"],
        quality_markers={
            "good": "Natural-sounding vowel character that adds personality and movement",
            "bad": "Artificial, robotic formant artifacts that sound unmusical",
        },
        famous_examples=["Talk box guitar sounds", "Formant-shifted vocal effects in electronic music"],
    ),
]


# ---------------------------------------------------------------------------
# Spatial / Time-Based (~15 techniques)
# ---------------------------------------------------------------------------

_SPATIAL: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Plate Reverb",
        category="spatial",
        description="Reverb generated by a vibrating metal plate, producing a dense, smooth decay with bright character.",
        parameters=["decay time", "pre-delay", "damping", "mix/send level"],
        sonic_effect="Dense, bright, shimmering reverb tail with fast buildup and smooth decay.",
        when_to_use=["Vocal ambience", "Snare depth", "Adding sheen to instruments"],
        genre_associations=["pop", "rock", "R&B", "soul"],
        quality_markers={
            "good": "Smooth, dense reverb that adds dimension without muddying the source",
            "bad": "Metallic, ringy artifacts or excessive brightness creating harshness",
        },
        famous_examples=["EMT 140 on vocals", "Lexicon plate on snare drum"],
    ),
    ProductionTechnique(
        name="Hall Reverb",
        category="spatial",
        description="Emulation of a large concert hall with long, complex decay patterns and natural diffusion.",
        parameters=["decay time", "pre-delay", "early reflections", "diffusion", "damping", "size"],
        sonic_effect="Grand, spacious ambience with natural-sounding early reflections and long, evolving tail.",
        when_to_use=["Orchestral depth", "Ballad vocals", "Creating epic, large spaces"],
        genre_associations=["classical", "orchestral", "cinematic", "ambient", "ballad pop"],
        quality_markers={
            "good": "Natural, immersive space that transports the listener into a believable acoustic environment",
            "bad": "Washy, indistinct reverb that drowns the source and creates muddy low-end buildup",
        },
        famous_examples=["Lexicon 480L hall on film scores", "Bricasti M7 hall on vocals"],
    ),
    ProductionTechnique(
        name="Room Reverb",
        category="spatial",
        description="Emulation of small to medium rooms with short decay times and prominent early reflections.",
        parameters=["room size", "decay time", "early reflections level", "damping", "diffusion"],
        sonic_effect="Intimate, natural ambience that places the source in a realistic small space.",
        when_to_use=["Natural drum ambience", "Intimate vocal recordings", "Acoustic instrument realism"],
        genre_associations=["rock", "jazz", "folk", "acoustic", "indie"],
        quality_markers={
            "good": "Believable room character that adds life without sounding overly processed",
            "bad": "Boxy, colored sound with flutter echoes or metallic early reflections",
        },
        famous_examples=["Live room at Abbey Road", "Drum room mics blended into the mix"],
    ),
    ProductionTechnique(
        name="Spring Reverb",
        category="spatial",
        description="Reverb produced by metal springs in a tank, with characteristic boing and splash.",
        parameters=["decay time", "mix", "tone", "spring count"],
        sonic_effect="Characteristic twangy, splashy reverb with a distinctive metallic crash on transients.",
        when_to_use=["Guitar amp ambience", "Surf rock character", "Vintage vocal effect", "Dub reggae"],
        genre_associations=["surf rock", "rockabilly", "dub", "psych rock", "lo-fi"],
        quality_markers={
            "good": "Characterful, vintage ambience with musical spring crash adding personality",
            "bad": "Excessive boing and metallic rattle dominating the source sound",
        },
        famous_examples=["Fender spring reverb in surf guitar", "Dub reggae spring effects"],
    ),
    ProductionTechnique(
        name="Convolution Reverb",
        category="spatial",
        description="Reverb that uses impulse responses captured from real spaces to recreate their exact acoustic properties.",
        parameters=["impulse response", "decay time", "pre-delay", "wet/dry mix", "EQ"],
        sonic_effect="Photorealistic reproduction of actual acoustic spaces, from cathedrals to closets.",
        when_to_use=["Placing instruments in real spaces", "Film/game audio realism", "Matching existing room sound"],
        genre_associations=["film scoring", "classical", "ambient", "sound design"],
        quality_markers={
            "good": "Convincingly real acoustic environment that integrates naturally with the source",
            "bad": "Static, lifeless sound lacking the organic movement of a real space",
        },
        famous_examples=["Altiverb cathedral IRs", "Audio Ease Speakerphone for creative convolution"],
    ),
    ProductionTechnique(
        name="Shimmer Reverb",
        category="spatial",
        description="Reverb with pitch-shifted feedback creating ethereal, octave-shifted trails that build in brightness.",
        parameters=["decay time", "pitch shift amount", "shimmer level", "mix", "damping"],
        sonic_effect="Ethereal, angelic reverb tail with ascending pitch content creating a crystalline, otherworldly space.",
        when_to_use=["Ambient soundscapes", "Ethereal vocal effects", "Post-rock guitar textures"],
        genre_associations=["ambient", "post-rock", "shoegaze", "cinematic", "new age"],
        quality_markers={
            "good": "Beautiful, evolving harmonic content that enhances atmosphere without becoming shrill",
            "bad": "Harsh, dissonant pitch artifacts or overwhelming shimmer drowning the source",
        },
        famous_examples=["Strymon BigSky shimmer", "Brian Eno ambient guitar textures"],
    ),
    ProductionTechnique(
        name="Gated Reverb",
        category="spatial",
        description="Reverb abruptly cut off by a noise gate, creating a big initial burst followed by sudden silence.",
        parameters=["reverb time (pre-gate)", "gate threshold", "gate hold", "gate release"],
        sonic_effect="Explosive, powerful initial reverb burst that stops dead, creating massive perceived size with clarity.",
        when_to_use=["Big drum sounds", "Power ballad snare", "Dramatic vocal effects"],
        genre_associations=["1980s pop", "rock", "power ballad", "synthwave"],
        quality_markers={
            "good": "Massive, powerful burst that enhances impact without cluttering the mix",
            "bad": "Unnatural, chopped-off decay with obvious gate chatter",
        },
        famous_examples=["Phil Collins 'In the Air Tonight' drum fill", "1980s power snare sound"],
    ),
    ProductionTechnique(
        name="Nonlinear Reverb",
        category="spatial",
        description="Reverb with a reverse or swelling envelope, growing louder before cutting off abruptly.",
        parameters=["reverb time", "envelope shape", "gate time", "mix"],
        sonic_effect="Unnatural reverb that builds in intensity before stopping, creating tension and drama.",
        when_to_use=["Dramatic drum effects", "Aggressive snare sound", "Sound design tension"],
        genre_associations=["rock", "metal", "industrial", "cinematic"],
        quality_markers={
            "good": "Dramatic, impactful envelope that adds excitement and energy",
            "bad": "Distracting, gimmicky effect that sounds disconnected from the source",
        },
        famous_examples=["AMS RMX16 nonlinear program", "Reverse-envelope snare in metal"],
    ),
    ProductionTechnique(
        name="Stereo Delay",
        category="spatial",
        description="Delay with independent left and right channels, creating width and rhythmic interest.",
        parameters=["left time", "right time", "feedback", "mix", "filter", "sync to tempo"],
        sonic_effect="Wide stereo image with rhythmic echoes bouncing between channels.",
        when_to_use=["Vocal width and depth", "Guitar spatial effects", "Rhythmic interest"],
        genre_associations=["pop", "rock", "ambient", "electronic"],
        quality_markers={
            "good": "Musical, rhythmic echoes that enhance width without cluttering the center",
            "bad": "Confusing, disorienting echoes that fight the rhythm or cause phase issues in mono",
        },
        famous_examples=["U2 Edge guitar delays", "Stereo vocal throws in pop"],
    ),
    ProductionTechnique(
        name="Ping-Pong Delay",
        category="spatial",
        description="Delay that alternates echoes between left and right channels in a bouncing pattern.",
        parameters=["delay time", "feedback", "width", "mix", "tempo sync"],
        sonic_effect="Echoes bouncing between speakers creating a wide, playful spatial effect.",
        when_to_use=["Creating width from mono sources", "Playful rhythmic effects", "Ambient textures"],
        genre_associations=["pop", "electronic", "ambient", "psychedelic"],
        quality_markers={
            "good": "Fun, musical bouncing echoes that fill the stereo field with rhythmic interest",
            "bad": "Distracting, excessive bouncing that undermines the groove or sounds gimmicky",
        },
        famous_examples=["Beatles stereo delay effects", "Pink Floyd spatial delays"],
    ),
    ProductionTechnique(
        name="Tape Delay",
        category="spatial",
        description="Delay emulating magnetic tape echo machines with characteristic wow, flutter, and degradation.",
        parameters=["delay time", "feedback", "wow", "flutter", "saturation", "filter", "mix"],
        sonic_effect="Warm, organic echoes that degrade naturally over repeats with tape saturation and pitch drift.",
        when_to_use=["Vintage vocal echo", "Guitar slapback", "Dub reggae effects", "Warm ambient textures"],
        genre_associations=["dub", "rockabilly", "vintage rock", "ambient", "country"],
        quality_markers={
            "good": "Warm, musical echoes with natural degradation that sit behind the source organically",
            "bad": "Excessive warble or pitch instability that sounds broken rather than vintage",
        },
        famous_examples=["Roland RE-201 Space Echo in dub", "Echoplex slapback on vocals"],
    ),
    ProductionTechnique(
        name="Slapback Delay",
        category="spatial",
        description="Very short single delay (50-120 ms) with no feedback, creating a doubling effect with space.",
        parameters=["delay time (50-120 ms)", "mix", "filter"],
        sonic_effect="Quick, single echo that thickens the source and adds perceived depth without obvious repetition.",
        when_to_use=["Vocal doubling effect", "Rockabilly guitar", "Thickening snare drum"],
        genre_associations=["rockabilly", "country", "early rock and roll", "pop"],
        quality_markers={
            "good": "Subtle thickening and depth enhancement that feels natural and cohesive",
            "bad": "Obvious, distracting echo that sounds like a timing error",
        },
        famous_examples=["Elvis Presley vocal slapback", "Sun Records rockabilly guitar"],
    ),
    ProductionTechnique(
        name="Dub Delay",
        category="spatial",
        description="Heavily effected delay with high feedback, filtering, and modulation characteristic of dub reggae.",
        parameters=["delay time", "feedback (high)", "filter cutoff", "modulation", "saturation", "spring reverb send"],
        sonic_effect="Evolving, psychedelic echo trails that build, modulate, and degrade into ambient washes.",
        when_to_use=["Dub reggae production", "Psychedelic effects", "Ambient sound design"],
        genre_associations=["dub", "reggae", "trip-hop", "ambient", "electronic"],
        quality_markers={
            "good": "Evolving, musical echo trails that create hypnotic, immersive textures",
            "bad": "Uncontrolled feedback buildup leading to harsh, runaway oscillation",
        },
        famous_examples=["King Tubby dub mixes", "Lee Scratch Perry echo experiments"],
    ),
    ProductionTechnique(
        name="Mono Delay",
        category="spatial",
        description="Single-channel delay that adds depth and rhythmic interest without widening the stereo image.",
        parameters=["delay time", "feedback", "mix", "filter", "tempo sync"],
        sonic_effect="Rhythmic echoes in a single position that add depth and groove without affecting stereo width.",
        when_to_use=["Rhythmic vocal echo", "Centered delay throws", "Creating depth without width"],
        genre_associations=["all genres"],
        quality_markers={
            "good": "Musical, tempo-locked echoes that enhance the groove and add depth",
            "bad": "Cluttered, off-time echoes that muddy the arrangement",
        },
        famous_examples=["Quarter-note vocal delays", "Dotted-eighth guitar delays"],
    ),
]


# ---------------------------------------------------------------------------
# Distortion / Saturation (~10 techniques)
# ---------------------------------------------------------------------------

_DISTORTION: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Tape Saturation",
        category="distortion",
        description="Harmonic enrichment and gentle compression from driving analog tape, adding warmth and cohesion.",
        parameters=["input level/drive", "tape speed", "bias", "saturation type", "wow/flutter"],
        sonic_effect="Warm, rounded tone with gentle compression, enhanced harmonics, and subtle high-frequency softening.",
        when_to_use=["Adding analog warmth to digital mixes", "Gentle drum glue", "Vocal richness", "Mix bus cohesion"],
        genre_associations=["all genres", "lo-fi", "vintage", "soul", "rock"],
        quality_markers={
            "good": "Subtle warmth and cohesion that makes digital recordings feel alive and musical",
            "bad": "Muddy, dull sound from excessive saturation or distracting wow/flutter",
        },
        famous_examples=["Studer A800 on drum bus", "Ampex ATR-102 on mix bus"],
    ),
    ProductionTechnique(
        name="Tube Saturation",
        category="distortion",
        description="Even-order harmonic distortion from vacuum tubes, adding richness and musical warmth.",
        parameters=["drive", "bias", "tube type", "output level"],
        sonic_effect="Rich, warm coloration with primarily even-order harmonics that thicken the sound musically.",
        when_to_use=["Vocal warmth and presence", "Bass guitar fatness", "Drum coloration", "Mastering warmth"],
        genre_associations=["blues", "rock", "jazz", "soul", "vintage"],
        quality_markers={
            "good": "Musical harmonic enrichment that adds depth and presence without harshness",
            "bad": "Fizzy, harsh distortion from overdriven tubes or excessive even-harmonic buildup",
        },
        famous_examples=["Neve 1073 preamp coloration", "Thermionic Culture Vulture on drums"],
    ),
    ProductionTechnique(
        name="Transformer Saturation",
        category="distortion",
        description="Harmonic coloration from audio transformers, adding iron-core warmth and gentle limiting.",
        parameters=["drive", "impedance", "transformer type"],
        sonic_effect="Thick, weighty coloration with enhanced low-mid presence and gentle transient rounding.",
        when_to_use=["Adding weight to thin sources", "Low-end thickening", "Vintage console coloration"],
        genre_associations=["rock", "blues", "vintage", "R&B"],
        quality_markers={
            "good": "Subtle weight and density that makes thin sources sound fuller and more analog",
            "bad": "Boomy, muddy low-mids from excessive transformer loading",
        },
        famous_examples=["Neve console transformer coloration", "API transformer sound"],
    ),
    ProductionTechnique(
        name="Overdrive",
        category="distortion",
        description="Moderate, dynamic distortion that responds to playing intensity, adding grit while retaining note definition.",
        parameters=["drive", "tone", "level", "clipping type"],
        sonic_effect="Warm, dynamic grit that adds edge and sustain while preserving the character of the source.",
        when_to_use=["Electric guitar tone", "Bass grit", "Adding edge to synths", "Vocal aggression"],
        genre_associations=["rock", "blues", "indie", "alternative"],
        quality_markers={
            "good": "Responsive, dynamic distortion that cleans up with volume and adds musical grit",
            "bad": "Fizzy, harsh upper harmonics or loss of dynamics and note definition",
        },
        famous_examples=["Ibanez Tube Screamer on guitar", "Slight amp overdrive in blues"],
    ),
    ProductionTechnique(
        name="Distortion",
        category="distortion",
        description="Heavy harmonic clipping that fundamentally transforms the waveform, adding aggressive harmonic content.",
        parameters=["gain/drive", "tone", "level", "distortion type"],
        sonic_effect="Aggressive, sustained tone with rich harmonic overtones and compressed dynamics.",
        when_to_use=["Heavy guitar tones", "Aggressive synth sounds", "Intentional lo-fi destruction"],
        genre_associations=["rock", "metal", "punk", "industrial", "noise"],
        quality_markers={
            "good": "Powerful, defined distortion with clear note articulation despite heavy saturation",
            "bad": "Indistinct, fizzy mess where notes and chords become unintelligible mush",
        },
        famous_examples=["Marshall amp stack distortion", "Boss DS-1 on guitar"],
    ),
    ProductionTechnique(
        name="Fuzz",
        category="distortion",
        description="Extreme clipping creating a thick, buzzy, almost square-wave distortion with unique harmonic character.",
        parameters=["fuzz level", "tone", "volume", "bias/starve"],
        sonic_effect="Thick, woolly, buzzing distortion with massive sustain and a raw, vintage character.",
        when_to_use=["Psychedelic guitar", "Bass fuzz", "Synth destruction", "Vintage rock tones"],
        genre_associations=["psychedelic rock", "stoner rock", "garage rock", "blues rock"],
        quality_markers={
            "good": "Thick, characterful buzz with musical harmonic content and massive sustain",
            "bad": "Farty, gated sputtering from dying batteries or excessive fuzz without musicality",
        },
        famous_examples=["Jimi Hendrix Fuzz Face", "Big Muff Pi on bass"],
    ),
    ProductionTechnique(
        name="Bitcrushing",
        category="distortion",
        description="Digital distortion that reduces bit depth, creating quantization noise and lo-fi digital character.",
        parameters=["bit depth", "mix", "tone"],
        sonic_effect="Crunchy, digital distortion with stepped quantization artifacts and retro digital character.",
        when_to_use=["Lo-fi aesthetics", "Video game sounds", "Drum destruction", "Creative sound design"],
        genre_associations=["electronic", "lo-fi", "chiptune", "industrial", "glitch"],
        quality_markers={
            "good": "Characterful digital crunch that adds retro texture and grit purposefully",
            "bad": "Harsh, unpleasant quantization noise that sounds like a broken system",
        },
        famous_examples=["Bitcrushed drums in electronic music", "Chiptune bass sounds"],
    ),
    ProductionTechnique(
        name="Sample Rate Reduction",
        category="distortion",
        description="Reducing the sample rate to create aliasing artifacts and a lo-fi digital sound.",
        parameters=["sample rate", "mix", "filter"],
        sonic_effect="Aliased, metallic digital distortion with reduced high-frequency content and digital artifacts.",
        when_to_use=["Lo-fi sound design", "Vintage sampler emulation", "Creative destruction"],
        genre_associations=["electronic", "lo-fi", "hip-hop", "experimental"],
        quality_markers={
            "good": "Musical aliasing that adds vintage digital character reminiscent of early samplers",
            "bad": "Harsh, inharmonic aliasing that sounds broken and unpleasant",
        },
        famous_examples=["SP-1200 crunch on hip-hop drums", "Early Akai sampler sound"],
    ),
    ProductionTechnique(
        name="Waveshaping",
        category="distortion",
        description="Distortion using a mathematical transfer function to reshape the waveform in controlled ways.",
        parameters=["drive", "transfer curve", "mix", "oversampling"],
        sonic_effect="Precise, controllable distortion from subtle warmth to extreme waveform mangling.",
        when_to_use=["Synth sound design", "Controlled harmonic enrichment", "Complex distortion textures"],
        genre_associations=["electronic", "sound design", "experimental"],
        quality_markers={
            "good": "Precisely controlled harmonic content that serves the sound design intention",
            "bad": "Harsh, aliased artifacts from insufficient oversampling or inappropriate transfer curves",
        },
        famous_examples=["Ableton Saturator waveshaping", "Native Instruments Driver"],
    ),
    ProductionTechnique(
        name="Amp Simulation",
        category="distortion",
        description="Digital modeling of guitar/bass amplifier circuits including preamp, power amp, and tone stack.",
        parameters=["amp model", "gain", "EQ (bass/mid/treble)", "presence", "master volume"],
        sonic_effect="Realistic amplifier tones from clean to heavily distorted, with characteristic EQ curves.",
        when_to_use=["Recording guitar without an amp", "Re-amping DI tracks", "Bass amp emulation"],
        genre_associations=["rock", "metal", "blues", "country", "pop"],
        quality_markers={
            "good": "Convincing, responsive amp feel with natural dynamics and musical distortion character",
            "bad": "Digital, fizzy, static-feeling tone that lacks the organic response of a real amp",
        },
        famous_examples=["Neural DSP Quad Cortex models", "Kemper profiling amp"],
    ),
    ProductionTechnique(
        name="Cabinet Simulation",
        category="distortion",
        description="Emulation of speaker cabinet frequency response and microphone placement using IRs or modeling.",
        parameters=["cabinet type", "speaker size", "mic type", "mic position", "room"],
        sonic_effect="Essential frequency shaping that transforms raw amp sound into a finished, natural guitar tone.",
        when_to_use=["Completing amp sim signal chain", "Re-amping through different cabinets", "Matching cabinet tones"],
        genre_associations=["rock", "metal", "blues", "any genre with electric guitar/bass"],
        quality_markers={
            "good": "Natural, mic'd-cabinet sound with appropriate frequency roll-off and resonance",
            "bad": "Boxy, artificial speaker sound lacking the natural air and depth of a real cabinet",
        },
        famous_examples=["Celestion Vintage 30 IRs", "OwnHammer cabinet impulse responses"],
    ),
]


# ---------------------------------------------------------------------------
# Modulation (~10 techniques)
# ---------------------------------------------------------------------------

_MODULATION: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Chorus",
        category="modulation",
        description="Copies the signal with slight pitch and time modulation to create a thicker, wider sound.",
        parameters=["rate", "depth", "mix", "voices", "delay", "feedback"],
        sonic_effect="Lush, shimmering doubling effect that thickens and widens the sound with gentle motion.",
        when_to_use=["Thickening clean guitars", "Adding width to synths", "Lush vocal doubling"],
        genre_associations=["80s pop", "shoegaze", "new wave", "dream pop", "jangle pop"],
        quality_markers={
            "good": "Lush, musical thickening that adds dimension without obvious pitch wobble",
            "bad": "Seasick, excessive pitch warble or metallic, phasey artifacts",
        },
        famous_examples=["Roland Juno-60 built-in chorus", "Boss CE-1 on clean guitar"],
    ),
    ProductionTechnique(
        name="Flanger",
        category="modulation",
        description="Short modulated delay mixed with the original creating a sweeping comb-filter effect.",
        parameters=["rate", "depth", "feedback", "mix", "manual/center"],
        sonic_effect="Dramatic, jet-like sweeping with metallic, resonant character from comb filtering.",
        when_to_use=["Dramatic sweeps on guitars", "Psychedelic effects", "Drum kit coloration"],
        genre_associations=["psychedelic rock", "metal", "electronic", "80s rock"],
        quality_markers={
            "good": "Dramatic, musical sweeping that adds movement and excitement",
            "bad": "Harsh, metallic resonance or excessive jet-engine effect that overwhelms the source",
        },
        famous_examples=["Van Halen flanged guitars", "Electric Mistress on bass"],
    ),
    ProductionTechnique(
        name="Phaser",
        category="modulation",
        description="All-pass filter sweeps creating moving notches and peaks in the frequency spectrum.",
        parameters=["rate", "depth", "feedback", "stages", "mix"],
        sonic_effect="Smooth, swirling tonal movement with a more subtle, liquid character than flanging.",
        when_to_use=["Funky rhythm guitar", "Synth pad movement", "Subtle vocal coloring", "Electric piano shimmer"],
        genre_associations=["funk", "disco", "progressive rock", "psychedelic", "electronic"],
        quality_markers={
            "good": "Smooth, musical phase shifting that adds hypnotic movement and depth",
            "bad": "Unnatural, robotic sweeping that sounds mechanical rather than musical",
        },
        famous_examples=["Small Stone on funk guitar", "MXR Phase 90 on Van Halen leads"],
    ),
    ProductionTechnique(
        name="Tremolo",
        category="modulation",
        description="Periodic modulation of amplitude (volume) creating a pulsing, wavering effect.",
        parameters=["rate", "depth", "waveform (sine/square/triangle)", "stereo spread"],
        sonic_effect="Rhythmic pulsing of volume that adds movement and vintage character.",
        when_to_use=["Vintage guitar ambience", "Rhythmic pulse effects", "Organ/keys character"],
        genre_associations=["surf rock", "country", "indie", "ambient", "cinematic"],
        quality_markers={
            "good": "Musical, rhythmic pulsing that enhances the groove and adds vintage vibe",
            "bad": "Choppy, distracting volume changes that fight the musical rhythm",
        },
        famous_examples=["Fender amp tremolo in surf rock", "Tremolo guitars in Nancy Sinatra recordings"],
    ),
    ProductionTechnique(
        name="Vibrato",
        category="modulation",
        description="Periodic modulation of pitch creating a wavering, vocal-like expressiveness.",
        parameters=["rate", "depth"],
        sonic_effect="Gentle pitch wavering that adds expressiveness and warmth, mimicking natural vocal vibrato.",
        when_to_use=["Adding expressiveness to synths", "Vintage organ emulation", "Guitar character"],
        genre_associations=["blues", "jazz", "classical", "soul"],
        quality_markers={
            "good": "Subtle, natural pitch variation that adds life and expressiveness",
            "bad": "Excessive, seasick pitch wobble that sounds out of tune",
        },
        famous_examples=["Hammond organ vibrato/chorus", "Vocal vibrato in opera and jazz"],
    ),
    ProductionTechnique(
        name="Ring Modulation",
        category="modulation",
        description="Multiplies two signals together producing sum and difference frequencies, creating inharmonic content.",
        parameters=["carrier frequency", "mix", "LFO rate", "LFO depth"],
        sonic_effect="Metallic, bell-like, or robotic timbres with inharmonic frequency content.",
        when_to_use=["Sci-fi sound effects", "Metallic percussion", "Experimental sound design"],
        genre_associations=["experimental", "electronic", "industrial", "sci-fi soundtracks"],
        quality_markers={
            "good": "Controlled, characterful metallic tones that serve the artistic vision",
            "bad": "Dissonant, unpleasant clashing frequencies without musical purpose",
        },
        famous_examples=["Dalek voices in Doctor Who", "Black Sabbath guitar effects"],
    ),
    ProductionTechnique(
        name="Frequency Shifting",
        category="modulation",
        description="Adds a fixed frequency offset to all harmonics, creating non-harmonic, detuned textures.",
        parameters=["shift amount (Hz)", "mix", "feedback"],
        sonic_effect="Increasingly dissonant, detuned sound as harmonics lose their natural integer relationships.",
        when_to_use=["Experimental textures", "Thickening through subtle detuning", "Barber-pole phasing"],
        genre_associations=["experimental", "ambient", "electronic", "noise"],
        quality_markers={
            "good": "Subtle, interesting harmonic shifts that create unique textures and movement",
            "bad": "Dissonant, unpleasant detuning that sounds like a broken pitch shifter",
        },
        famous_examples=["Bode frequency shifter in electronic music", "Barber-pole phasing effects"],
    ),
    ProductionTechnique(
        name="Rotary Speaker (Leslie)",
        category="modulation",
        description="Emulation of a rotating speaker cabinet creating complex Doppler-based pitch and amplitude modulation.",
        parameters=["speed (slow/fast)", "acceleration", "horn level", "drum level", "distance"],
        sonic_effect="Rich, swirling 3D modulation with complex interactions between pitch, amplitude, and frequency.",
        when_to_use=["Hammond organ sound", "Guitar psychedelic effect", "Vocal warping"],
        genre_associations=["blues", "jazz", "rock", "gospel", "progressive rock"],
        quality_markers={
            "good": "Organic, three-dimensional swirl with natural Doppler character and depth",
            "bad": "Static, fake-sounding rotation lacking the complex physics of a real Leslie",
        },
        famous_examples=["Hammond B3 through Leslie 122", "Beatles psychedelic vocal effects"],
    ),
    ProductionTechnique(
        name="Auto-Pan",
        category="modulation",
        description="Automated stereo panning that moves the signal between left and right at a set rate.",
        parameters=["rate", "depth", "waveform", "phase", "tempo sync"],
        sonic_effect="Rhythmic stereo movement that creates a sense of motion across the stereo field.",
        when_to_use=["Creating stereo motion from mono sources", "Rhythmic spatial effects", "Psychedelic textures"],
        genre_associations=["psychedelic", "electronic", "ambient", "experimental"],
        quality_markers={
            "good": "Musical, rhythmic panning that enhances the arrangement without being distracting",
            "bad": "Excessive, disorienting panning that causes listener fatigue or mono compatibility issues",
        },
        famous_examples=["Beatles hard panning effects", "Jimi Hendrix stereo experiments"],
    ),
    ProductionTechnique(
        name="Wow and Flutter",
        category="modulation",
        description="Slow (wow) and fast (flutter) pitch variations emulating tape or vinyl playback imperfections.",
        parameters=["wow rate", "wow depth", "flutter rate", "flutter depth"],
        sonic_effect="Gentle pitch instability that adds warmth, nostalgia, and an analog-imperfect character.",
        when_to_use=["Lo-fi aesthetics", "Vintage tape emulation", "Nostalgic sound design"],
        genre_associations=["lo-fi", "vaporwave", "chillwave", "vintage", "indie"],
        quality_markers={
            "good": "Subtle, nostalgic pitch drift that adds character without sounding broken",
            "bad": "Excessive warble that makes the pitch sound unstable and unpleasant",
        },
        famous_examples=["Lo-fi hip-hop tape warble", "Vaporwave pitch drift aesthetics"],
    ),
]


# ---------------------------------------------------------------------------
# Stereo / Width (~8 techniques)
# ---------------------------------------------------------------------------

_STEREO: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Stereo Widening (Haas Effect)",
        category="stereo",
        description="Creating perceived width by delaying one channel by 1-30 ms, exploiting the precedence effect.",
        parameters=["delay time (1-30 ms)", "channel (L/R)", "EQ difference"],
        sonic_effect="Dramatically wider perceived stereo image while maintaining a centered phantom image.",
        when_to_use=["Widening mono sources", "Creating space for vocals in center", "Doubling effects"],
        genre_associations=["pop", "rock", "electronic"],
        quality_markers={
            "good": "Natural-sounding width enhancement with solid center image and mono compatibility",
            "bad": "Phasey, comb-filtered sound when collapsed to mono or unnatural localization",
        },
        famous_examples=["Haas-widened backing vocals", "Wide guitar doubles"],
    ),
    ProductionTechnique(
        name="Mid/Side Processing",
        category="stereo",
        description="Encoding stereo as mid (center) and side (difference) for independent processing of each.",
        parameters=["mid level", "side level", "mid processing", "side processing"],
        sonic_effect="Independent control over center and side content for precise stereo field manipulation.",
        when_to_use=["Mastering stereo enhancement", "Tightening bass in center", "Widening ambient elements"],
        genre_associations=["mastering", "pop", "electronic"],
        quality_markers={
            "good": "Precise, controlled stereo manipulation with maintained mono compatibility",
            "bad": "Excessive side boost causing hollow center or mono collapse",
        },
        famous_examples=["M/S mastering EQ for width", "M/S compression for stereo balance"],
    ),
    ProductionTechnique(
        name="Static Panning",
        category="stereo",
        description="Fixed placement of elements in the stereo field to create a balanced, spacious mix.",
        parameters=["pan position (-100 to +100)", "pan law (dB compensation)"],
        sonic_effect="Clear spatial positioning of mix elements creating width, separation, and a defined soundstage.",
        when_to_use=["Basic mix panning", "Creating stereo balance", "Separating competing elements"],
        genre_associations=["all genres"],
        quality_markers={
            "good": "Balanced, spacious stereo image with clear element placement and mono compatibility",
            "bad": "Lopsided, unbalanced image or everything piled in the center",
        },
        famous_examples=["LCR panning technique", "Drum overhead panning (audience vs. drummer)"],
    ),
    ProductionTechnique(
        name="Stereo Imaging",
        category="stereo",
        description="Processing that adjusts the perceived width, focus, and shape of the stereo image.",
        parameters=["width", "center focus", "frequency-dependent width", "correlation"],
        sonic_effect="Adjusted stereo field width from mono-narrowed to hyper-wide, with control over focus.",
        when_to_use=["Mastering width adjustment", "Fixing stereo problems", "Enhancing spatial perception"],
        genre_associations=["mastering", "all genres"],
        quality_markers={
            "good": "Enhanced stereo perception that sounds natural and translates well across playback systems",
            "bad": "Phasey, unfocused image with poor mono compatibility or exaggerated wideness",
        },
        famous_examples=["iZotope Ozone Imager", "Waves S1 stereo imager"],
    ),
    ProductionTechnique(
        name="Mono Compatibility Processing",
        category="stereo",
        description="Ensuring stereo mixes translate well when summed to mono for phone speakers and PA systems.",
        parameters=["correlation meter", "bass mono cutoff", "phase correction"],
        sonic_effect="A mix that maintains its essential character and balance when played in mono.",
        when_to_use=["Final mix checking", "Mastering QC", "Content for mobile playback"],
        genre_associations=["all genres", "broadcast", "streaming"],
        quality_markers={
            "good": "Mix sounds full and balanced in both stereo and mono with minimal level changes",
            "bad": "Significant frequency cancellations or level drops when summed to mono",
        },
        famous_examples=["Bass mono-summing below 120 Hz", "Phase-checking stereo effects"],
    ),
    ProductionTechnique(
        name="Binaural Processing",
        category="stereo",
        description="HRTF-based processing that creates 3D spatial audio for headphone listening.",
        parameters=["HRTF profile", "elevation", "azimuth", "distance", "room model"],
        sonic_effect="Three-dimensional sound placement including height and depth perceived through headphones.",
        when_to_use=["Immersive audio for headphones", "VR/game audio", "ASMR production"],
        genre_associations=["ambient", "experimental", "ASMR", "VR audio", "film"],
        quality_markers={
            "good": "Convincing 3D placement with natural head-tracking-like spatial cues",
            "bad": "Inside-the-head sound with unconvincing externalization or tonal coloration",
        },
        famous_examples=["Binaural recordings for ASMR", "Sony 360 Reality Audio"],
    ),
    ProductionTechnique(
        name="Spatial Audio (Atmos/Immersive)",
        category="stereo",
        description="Object-based or channel-based 3D audio for immersive speaker arrays and headphone rendering.",
        parameters=["object position (x/y/z)", "bed channels", "binaural render", "height channels"],
        sonic_effect="Fully immersive 3D sound field with height, depth, and enveloping spatial perception.",
        when_to_use=["Dolby Atmos music mixing", "Immersive experiences", "Spatial audio content"],
        genre_associations=["all genres (Atmos mixes)", "cinematic", "ambient", "electronic"],
        quality_markers={
            "good": "Immersive, enveloping sound field that enhances the musical experience with natural movement",
            "bad": "Gimmicky object placement that distracts from the music or poor fold-down to stereo",
        },
        famous_examples=["Apple Music Spatial Audio releases", "Dolby Atmos film scores"],
    ),
    ProductionTechnique(
        name="Stereo Microphone Techniques",
        category="stereo",
        description="Capturing natural stereo width at recording stage using paired microphones in specific configurations.",
        parameters=["technique (XY/ORTF/AB/Blumlein/MS)", "mic spacing", "angle", "mic type"],
        sonic_effect="Natural stereo image captured at the source, preserving the acoustic reality of the performance.",
        when_to_use=["Recording ensembles", "Drum overheads", "Orchestral capture", "Ambient recording"],
        genre_associations=["classical", "jazz", "acoustic", "live recording"],
        quality_markers={
            "good": "Natural, realistic stereo image with excellent mono compatibility and accurate localization",
            "bad": "Phasey, unfocused image from improper spacing or excessive width",
        },
        famous_examples=["Decca Tree for orchestral recording", "ORTF overheads on drum kit"],
    ),
]


# ---------------------------------------------------------------------------
# Creative / Modern (~15 techniques)
# ---------------------------------------------------------------------------

_CREATIVE: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Sidechain Pumping",
        category="creative",
        description="Exaggerated sidechain compression creating an intentional rhythmic volume pumping effect.",
        parameters=["sidechain source (kick)", "threshold", "ratio", "attack (fast)", "release (medium)"],
        sonic_effect="Dramatic rhythmic breathing where pads and basses duck hard on each kick, creating groove.",
        when_to_use=["EDM energy and groove", "Dance music pumping", "Creating rhythmic movement"],
        genre_associations=["EDM", "house", "trance", "future bass", "electro pop"],
        quality_markers={
            "good": "Musical, groove-enhancing pumping that drives the track forward with energy",
            "bad": "Excessive, gasping pumping that sounds suffocating and fatiguing",
        },
        famous_examples=["Daft Punk sidechain pumping", "Deadmau5 bass pumping"],
    ),
    ProductionTechnique(
        name="Vocal Tuning (Auto-Tune/Melodyne)",
        category="creative",
        description="Pitch correction of vocal performances ranging from transparent to heavily stylized.",
        parameters=["retune speed", "key/scale", "correction amount", "humanize", "formant correction"],
        sonic_effect="Pitch-corrected vocals from imperceptibly natural to the iconic hard-tuned robotic effect.",
        when_to_use=["Correcting pitch inaccuracies", "Hard-tuned vocal effect", "Harmonizing"],
        genre_associations=["pop", "hip-hop", "R&B", "trap", "country (transparent)"],
        quality_markers={
            "good": "Transparent correction preserving natural expression, or intentional stylistic effect",
            "bad": "Warbling artifacts, unnatural formant shifts, or correction fighting vibrato",
        },
        famous_examples=["Cher 'Believe' hard-tuned effect", "T-Pain vocal style"],
    ),
    ProductionTechnique(
        name="Time Stretching",
        category="creative",
        description="Changing the duration of audio without affecting pitch, using granular or phase-vocoder algorithms.",
        parameters=["stretch factor", "algorithm", "grain size", "formant preservation"],
        sonic_effect="Tempo-adjusted audio that can range from transparent to characterfully artifacted.",
        when_to_use=["Matching sample tempos", "Creative slow-motion effects", "Sound design"],
        genre_associations=["electronic", "hip-hop", "ambient", "experimental"],
        quality_markers={
            "good": "Transparent tempo change or intentionally characterful stretching artifacts",
            "bad": "Phasey, metallic artifacts or choppy, glitchy audio from poor algorithm choice",
        },
        famous_examples=["Ableton warp modes", "Paul Stretch extreme time stretching"],
    ),
    ProductionTechnique(
        name="Pitch Shifting",
        category="creative",
        description="Changing the pitch of audio without altering its duration.",
        parameters=["semitones", "cents", "formant shift", "algorithm", "mix"],
        sonic_effect="Transposed audio from subtle thickening detuning to dramatic octave shifts.",
        when_to_use=["Creating harmonies", "Thickening through micro-detuning", "Sound design"],
        genre_associations=["pop", "electronic", "hip-hop", "experimental"],
        quality_markers={
            "good": "Clean, natural-sounding transposition or characterful shifting with musical artifacts",
            "bad": "Chipmunk-like formant distortion or phasey, metallic artifacts",
        },
        famous_examples=["Eventide H3000 harmonizer", "Micro-pitch detuning for width"],
    ),
    ProductionTechnique(
        name="Granular Processing",
        category="creative",
        description="Breaking audio into tiny grains (1-100 ms) and reorganizing them for texture and sound design.",
        parameters=["grain size", "grain density", "pitch randomization", "position randomization", "spray", "envelope"],
        sonic_effect="Ethereal, textural clouds of sound from smooth pads to glitchy, stuttered textures.",
        when_to_use=["Ambient texture creation", "Sound design", "Transforming recordings into pads"],
        genre_associations=["ambient", "experimental", "electronic", "sound design"],
        quality_markers={
            "good": "Rich, evolving textures that transform the source into something new and compelling",
            "bad": "Harsh, repetitive clicking from poor grain windowing or excessive randomization",
        },
        famous_examples=["Granulator II in Ableton", "Robert Henke granular soundscapes"],
    ),
    ProductionTechnique(
        name="Spectral Processing",
        category="creative",
        description="FFT-based processing that manipulates individual frequency bins for surgical editing or sound design.",
        parameters=["FFT size", "overlap", "spectral operation", "frequency range"],
        sonic_effect="Precise frequency-domain manipulation from noise removal to otherworldly spectral transformations.",
        when_to_use=["Advanced noise removal", "Spectral editing", "Creative frequency manipulation"],
        genre_associations=["sound design", "experimental", "post-production"],
        quality_markers={
            "good": "Clean, precise frequency manipulation with minimal artifacts",
            "bad": "Musical noise, phasing, or tinkly artifacts from aggressive spectral processing",
        },
        famous_examples=["iZotope RX spectral repair", "Spectral freeze in sound design"],
    ),
    ProductionTechnique(
        name="Vocoder",
        category="creative",
        description="Imposes the spectral envelope of one signal (voice) onto another (synth), creating talking instruments.",
        parameters=["carrier signal", "modulator signal", "bands", "attack", "release", "formant shift"],
        sonic_effect="Robotic, singing synthesizer effect where instruments appear to speak or sing.",
        when_to_use=["Robot voice effects", "Talking synth pads", "Retro electronic vocals"],
        genre_associations=["electronic", "funk", "electro", "synthwave", "80s pop"],
        quality_markers={
            "good": "Clear, intelligible vocoded speech with rich harmonic carrier content",
            "bad": "Unintelligible mush from too few bands or poor carrier/modulator matching",
        },
        famous_examples=["Kraftwerk vocoder vocals", "Daft Punk 'Around the World'"],
    ),
    ProductionTechnique(
        name="Talkbox Effect",
        category="creative",
        description="Routing a sound through a tube into the performer's mouth, using mouth shape to filter the sound.",
        parameters=["source signal", "tube type", "mic placement"],
        sonic_effect="Organic, human-like vowel filtering creating a singing instrument effect more natural than a vocoder.",
        when_to_use=["Funk guitar talk effects", "Singing synth leads", "Retro R&B/funk"],
        genre_associations=["funk", "R&B", "hip-hop", "80s rock"],
        quality_markers={
            "good": "Clear, expressive vowel articulation with natural-sounding musical phrasing",
            "bad": "Muffled, unintelligible sound with excessive tube coloration",
        },
        famous_examples=["Peter Frampton 'Show Me the Way'", "Roger Troutman/Zapp talkbox"],
    ),
    ProductionTechnique(
        name="Glitch Effects",
        category="creative",
        description="Intentional digital artifacts including buffer repeats, stutters, bit errors, and audio glitches.",
        parameters=["glitch type", "probability", "buffer size", "repeat count", "mix"],
        sonic_effect="Digital chaos ranging from subtle hiccups to full audio destruction and reorganization.",
        when_to_use=["IDM and glitch music", "Transition effects", "Adding digital chaos"],
        genre_associations=["glitch", "IDM", "electronic", "experimental"],
        quality_markers={
            "good": "Rhythmically musical glitches that enhance groove and add unpredictable excitement",
            "bad": "Random, unmusical chaos that sounds like a broken audio file",
        },
        famous_examples=["Autechre glitch rhythms", "dBlue Glitch plugin effects"],
    ),
    ProductionTechnique(
        name="Stutter Edit",
        category="creative",
        description="Rapid-fire buffer repeat effects creating machine-gun-like rhythmic stutters.",
        parameters=["stutter rate", "gate pattern", "pitch", "filter", "pan"],
        sonic_effect="Rapid-fire rhythmic repetitions from subtle stutters to aggressive machine-gun effects.",
        when_to_use=["Build-ups and drops", "Transition effects", "Rhythmic fills"],
        genre_associations=["EDM", "dubstep", "trap", "electronic"],
        quality_markers={
            "good": "Exciting, energetic stutters that build tension and release effectively",
            "bad": "Monotonous, repetitive stuttering that sounds mechanical and unmusical",
        },
        famous_examples=["iZotope Stutter Edit 2", "BT's signature stutter technique"],
    ),
    ProductionTechnique(
        name="Reverse Reverb",
        category="creative",
        description="Reverb applied to a reversed signal, then the result is reversed back, creating a swelling pre-echo.",
        parameters=["reverb type", "decay time", "reverse amount", "mix"],
        sonic_effect="Ethereal swell leading into a note or word, creating anticipation and dramatic tension.",
        when_to_use=["Vocal entrances", "Dramatic build-ups", "Ethereal transitions"],
        genre_associations=["pop", "ambient", "shoegaze", "cinematic", "electronic"],
        quality_markers={
            "good": "Smooth, dramatic swell that perfectly leads into the source sound",
            "bad": "Awkward timing or abrupt cutoff that sounds disconnected from the source",
        },
        famous_examples=["Reverse reverb on vocal entrances", "Shoegaze guitar swells"],
    ),
    ProductionTechnique(
        name="Spectral Freeze",
        category="creative",
        description="Capturing and sustaining a single moment of a sound's frequency content indefinitely.",
        parameters=["freeze point", "FFT size", "jitter", "pitch", "mix"],
        sonic_effect="A single spectral snapshot sustained as an evolving, shimmering drone or pad texture.",
        when_to_use=["Creating drones from any source", "Ambient pad generation", "Sound design"],
        genre_associations=["ambient", "drone", "experimental", "sound design"],
        quality_markers={
            "good": "Rich, evolving frozen texture that creates an immersive, sustained atmosphere",
            "bad": "Static, buzzy, or harsh frozen spectrum that sounds digital and lifeless",
        },
        famous_examples=["Spectral freeze pads in ambient music", "Michael Norris spectral tools"],
    ),
    ProductionTechnique(
        name="Creative Convolution",
        category="creative",
        description="Using impulse responses of non-reverb sources to impose their character onto audio.",
        parameters=["impulse response", "mix", "pre-delay", "decay", "EQ"],
        sonic_effect="Source audio takes on the timbral and resonant qualities of the convolution source.",
        when_to_use=["Imposing textures from objects", "Creative sound morphing", "Unique reverb characters"],
        genre_associations=["sound design", "experimental", "electronic", "ambient"],
        quality_markers={
            "good": "Fascinating hybrid sounds that combine source and impulse characteristics musically",
            "bad": "Muddy, indistinct convolution that loses both source and impulse character",
        },
        famous_examples=["Convolving guitar with a metal pipe IR", "Using book impacts as reverb"],
    ),
    ProductionTechnique(
        name="Re-amping",
        category="creative",
        description="Playing back recorded DI signals through physical amplifiers and re-recording the result.",
        parameters=["amp selection", "mic selection", "mic position", "room", "gain staging"],
        sonic_effect="Real amplifier tone and interaction applied to a clean recording after the performance.",
        when_to_use=["Perfecting guitar tone after recording", "Trying multiple amp sounds", "Adding real amp character"],
        genre_associations=["rock", "metal", "blues", "indie"],
        quality_markers={
            "good": "Authentic, lively amp tone with natural dynamics and room interaction",
            "bad": "Thin, lifeless re-amp from poor gain staging or impedance mismatch",
        },
        famous_examples=["Re-amping DI bass through an SVT", "Trying different amp heads on DI guitar"],
    ),
]


# ---------------------------------------------------------------------------
# Mastering (~10 techniques)
# ---------------------------------------------------------------------------

_MASTERING: list[ProductionTechnique] = [
    ProductionTechnique(
        name="Mastering EQ",
        category="mastering",
        description="Broad, gentle EQ moves to balance the overall frequency spectrum of a finished mix.",
        parameters=["frequency", "gain (subtle, 0.5-2 dB)", "Q (broad)", "type (shelf/bell)"],
        sonic_effect="Subtle tonal rebalancing that perfects the frequency spectrum without changing the mix character.",
        when_to_use=["Correcting overall tonal balance", "Adding final polish", "Matching reference tracks"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Subtle corrections of 1-2 dB that perfect balance without altering the mix identity",
            "bad": "Drastic EQ moves that fundamentally change the mixer's tonal intent",
        },
        famous_examples=["Manley Massive Passive in mastering", "Dangerous BAX EQ"],
    ),
    ProductionTechnique(
        name="Mastering Compression",
        category="mastering",
        description="Gentle, transparent compression to add cohesion and control overall dynamics.",
        parameters=["threshold", "ratio (1.5:1-3:1)", "attack (slow)", "release (auto)", "makeup gain"],
        sonic_effect="Subtle cohesion and dynamic control that glues the mix while preserving musicality.",
        when_to_use=["Adding mix cohesion", "Controlling macro dynamics", "Gluing the final mix"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "1-3 dB of gain reduction adding subtle cohesion without audible compression",
            "bad": "Squashed, lifeless master with lost transients and obvious pumping",
        },
        famous_examples=["Shadow Hills Mastering Compressor", "Manley Variable Mu in mastering"],
    ),
    ProductionTechnique(
        name="Mastering Limiting",
        category="mastering",
        description="Final-stage brickwall limiting to maximize loudness while preventing digital clipping.",
        parameters=["ceiling (-0.3 to -1.0 dBTP)", "threshold/input gain", "release", "algorithm/mode"],
        sonic_effect="Maximized loudness at the final stage with true peak prevention for distribution compliance.",
        when_to_use=["Final loudness stage", "Meeting platform targets", "Distribution preparation"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Transparent loudness increase that meets targets without audible distortion or pumping",
            "bad": "Crushed, distorted master with inter-sample peaks and destroyed transients",
        },
        famous_examples=["FabFilter Pro-L 2 on masters", "Waves L2 in commercial mastering"],
    ),
    ProductionTechnique(
        name="Stereo Enhancement (Mastering)",
        category="mastering",
        description="Subtle stereo field adjustment at the mastering stage to optimize width and focus.",
        parameters=["width", "mid/side balance", "frequency-dependent width", "mono bass cutoff"],
        sonic_effect="Optimized stereo image with wide sides, focused center, and tight mono bass.",
        when_to_use=["Perfecting stereo image", "Ensuring mono compatibility", "Width optimization"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Enhanced sense of space and width that translates across all playback systems",
            "bad": "Exaggerated width causing mono collapse or hollow center image",
        },
        famous_examples=["iZotope Ozone Imager in mastering", "Brainworx M/S mastering"],
    ),
    ProductionTechnique(
        name="LUFS Targeting",
        category="mastering",
        description="Adjusting the integrated loudness to meet specific platform or broadcast loudness standards.",
        parameters=["target LUFS", "platform (Spotify/-14, Apple/-16, YouTube/-14)", "true peak ceiling"],
        sonic_effect="Consistent playback loudness across streaming platforms without normalization penalties.",
        when_to_use=["Preparing for streaming distribution", "Broadcast compliance", "Platform-specific masters"],
        genre_associations=["all genres (distribution)"],
        quality_markers={
            "good": "Hitting target loudness with maintained dynamics and no normalization penalty",
            "bad": "Overloud master that gets turned down by platform normalization, losing impact",
        },
        famous_examples=["Spotify -14 LUFS masters", "Apple Music -16 LUFS compliance"],
    ),
    ProductionTechnique(
        name="Dithering",
        category="mastering",
        description="Adding shaped noise when reducing bit depth to preserve low-level detail and avoid quantization distortion.",
        parameters=["bit depth target", "dither type (TPDF/noise-shaped)", "noise shaping curve"],
        sonic_effect="Preserved low-level detail and smooth fade-outs when converting from 24/32-bit to 16-bit.",
        when_to_use=["Final bit-depth reduction for CD", "Converting 24-bit to 16-bit", "Last step in mastering chain"],
        genre_associations=["all genres (final export)"],
        quality_markers={
            "good": "Clean, artifact-free bit-depth reduction with smooth fade-outs and low-level detail",
            "bad": "Audible quantization distortion on quiet passages or truncation artifacts on fades",
        },
        famous_examples=["POW-r dithering in mastering", "Apogee UV22HR"],
    ),
    ProductionTechnique(
        name="Mid/Side Mastering",
        category="mastering",
        description="Independent EQ, compression, and processing of mid and side channels at the mastering stage.",
        parameters=["mid EQ", "side EQ", "mid compression", "side compression", "balance"],
        sonic_effect="Precise control over center content and stereo spread independently for optimal spatial balance.",
        when_to_use=["Tightening bass in center", "Widening reverb tails on sides", "Balancing stereo field"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Focused center with wide, detailed sides maintaining natural stereo balance",
            "bad": "Disconnected mid and side causing phasey, unnatural stereo image",
        },
        famous_examples=["Brainworx bx_digital M/S mastering", "Mid/side bass tightening in mastering"],
    ),
    ProductionTechnique(
        name="Multiband Mastering Compression",
        category="mastering",
        description="Frequency-selective compression at the mastering stage for precise dynamic control per band.",
        parameters=["crossover frequencies", "threshold per band", "ratio per band", "attack per band", "release per band"],
        sonic_effect="Frequency-dependent dynamic control that balances the spectrum without global compression artifacts.",
        when_to_use=["Taming inconsistent low end", "Controlling sibilant buildup", "Frequency-specific dynamics"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Transparent, frequency-specific control that evens the spectrum naturally",
            "bad": "Obvious crossover artifacts, phasiness, or individual band pumping",
        },
        famous_examples=["Waves LinMB in mastering", "FabFilter Pro-MB on master bus"],
    ),
    ProductionTechnique(
        name="Harmonic Excitement",
        category="mastering",
        description="Adding subtle harmonic overtones to enhance perceived brightness, warmth, or presence at mastering.",
        parameters=["drive", "frequency range", "harmonics type (even/odd)", "mix"],
        sonic_effect="Enhanced perceived loudness and presence through added harmonics without EQ boosts.",
        when_to_use=["Adding sparkle without EQ", "Enhancing perceived loudness", "Warming thin masters"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Subtle enhancement that makes the master feel more alive and present without being harsh",
            "bad": "Harsh, grainy overtones that add fatiguing distortion",
        },
        famous_examples=["SPL Vitalizer in mastering", "Maag EQ Air Band"],
    ),
    ProductionTechnique(
        name="Reference Matching",
        category="mastering",
        description="Analyzing and matching the tonal balance, loudness, and dynamics of a reference track.",
        parameters=["reference track", "match amount", "frequency matching", "loudness matching"],
        sonic_effect="Mastered track that shares the spectral balance and loudness characteristics of a proven reference.",
        when_to_use=["Ensuring competitive loudness and tone", "Matching genre standards", "Client reference matching"],
        genre_associations=["all genres (mastering stage)"],
        quality_markers={
            "good": "Master that sits confidently alongside references while maintaining its own identity",
            "bad": "Slavish imitation that forces inappropriate tonal balance onto the source material",
        },
        famous_examples=["iZotope Ozone reference matching", "REFERENCE plugin by Mastering The Mix"],
    ),
]


# ---------------------------------------------------------------------------
# Mix Positions
# ---------------------------------------------------------------------------

_MIX_POSITIONS: list[MixPosition] = [
    MixPosition(
        instrument="Kick Drum",
        frequency_home="Sub (30-60 Hz) for weight, attack click at 2-5 kHz, body at 80-150 Hz",
        typical_panning="Dead center",
        depth_position="Front of the mix, very present",
        common_processing=["EQ (HPF ~30 Hz, boost click 3-5 kHz)", "compression (fast attack for control or slow attack for punch)", "saturation for harmonic weight", "sidechain trigger for bass"],
        genre_variations={
            "EDM": "Heavily processed, layered with samples, sub-heavy, tight and punchy",
            "jazz": "Natural, minimal processing, room mic blended for realism",
            "hip-hop": "808 sub-bass dominant, long sustain, distorted or clean depending on style",
            "rock": "Punchy, mid-forward attack, less sub emphasis than electronic genres",
            "metal": "Tight, clicky attack with fast gate, triggered or replaced for consistency",
        },
    ),
    MixPosition(
        instrument="Snare Drum",
        frequency_home="Body at 150-300 Hz, crack at 1-4 kHz, snap/air at 5-10 kHz",
        typical_panning="Center or slightly off-center",
        depth_position="Front of the mix, prominent",
        common_processing=["EQ (body 200 Hz, crack 2-4 kHz)", "compression (medium attack to let transient through)", "parallel compression for body", "reverb for depth"],
        genre_variations={
            "pop": "Bright, snappy, often layered with claps and samples",
            "rock": "Full, fat tone with ring, less processed, room reverb",
            "hip-hop": "Tight, dry, sometimes heavily layered or pitched",
            "jazz": "Natural, dynamic, minimal compression, brush or stick texture preserved",
            "metal": "Tight, aggressive crack, gated reverb or completely dry",
        },
    ),
    MixPosition(
        instrument="Hi-Hat",
        frequency_home="Shimmer at 6-12 kHz, body at 200-500 Hz (to be cut), presence at 3-6 kHz",
        typical_panning="Slightly off-center (15-40% left or right depending on perspective)",
        depth_position="Mid-depth, slightly behind kick and snare",
        common_processing=["HPF at 200-400 Hz", "gentle compression", "subtle EQ for brightness", "low-level sends to reverb"],
        genre_variations={
            "hip-hop": "Often programmed, tight closed hat patterns, crisp and forward",
            "jazz": "Natural, dynamic, ride cymbal often more prominent than hi-hat",
            "electronic": "Heavily processed, filtered, layered, rhythmically complex",
            "rock": "Natural, moderate processing, part of overhead mic capture",
        },
    ),
    MixPosition(
        instrument="Bass (Electric/Synth)",
        frequency_home="Sub at 30-80 Hz, fundamental at 60-200 Hz, presence/growl at 500 Hz-2 kHz",
        typical_panning="Dead center",
        depth_position="Front to mid, foundational element",
        common_processing=["HPF at 25-30 Hz", "compression for consistency", "EQ for clarity (cut mud 200-300 Hz)", "saturation for harmonic presence on small speakers"],
        genre_variations={
            "funk": "Bright, snappy, forward, prominent slap and pop frequencies",
            "rock": "Mid-heavy, driven, picks up grit from amp overdrive",
            "electronic": "Sub-dominant, clean or heavily processed, sidechain ducked",
            "jazz": "Natural upright sound, warm, less compressed, more dynamic range",
            "hip-hop": "808 sub-bass or deep synth bass, dominant in the low end",
        },
    ),
    MixPosition(
        instrument="Electric Guitar",
        frequency_home="Body at 200-500 Hz, bite at 1-4 kHz, presence at 4-8 kHz",
        typical_panning="Hard left and right for doubles, slightly off-center for single, center for leads",
        depth_position="Varies: front for leads, mid-depth for rhythm",
        common_processing=["HPF at 80-120 Hz", "amp/cab simulation or real mic'd amp", "compression for leads", "delay and reverb for space", "EQ cuts in vocal range"],
        genre_variations={
            "rock": "Crunchy, mid-heavy, double-tracked rhythm hard-panned, prominent",
            "jazz": "Clean, warm, single coil or hollow-body tone, minimal effects",
            "metal": "High-gain, tight low end, heavily EQ'd for definition in dense mixes",
            "country": "Clean with compression, telecaster twang, moderate reverb",
            "indie": "Varied textures, chorus/delay heavy, jangly or atmospheric",
        },
    ),
    MixPosition(
        instrument="Acoustic Guitar",
        frequency_home="Body resonance at 100-300 Hz, string detail at 2-5 kHz, air at 8-12 kHz",
        typical_panning="Slightly off-center or hard-panned if doubled, center for solo performance",
        depth_position="Mid-depth, supportive role in full mixes, front in acoustic settings",
        common_processing=["HPF at 80-100 Hz", "compression for even strumming", "EQ (cut body mud, boost string shimmer)", "light reverb for space"],
        genre_variations={
            "folk": "Natural, minimal processing, room ambience, front and center",
            "pop": "Bright, compressed, sits as a texture element behind vocals",
            "country": "Crisp, present, moderate compression, Nashville-style bright",
            "singer-songwriter": "Intimate, close-mic'd, natural dynamics, centered",
        },
    ),
    MixPosition(
        instrument="Piano/Keys",
        frequency_home="Low register 80-300 Hz, body 300 Hz-1 kHz, brilliance 2-8 kHz, air above 8 kHz",
        typical_panning="Centered or spread stereo across the field for grand piano",
        depth_position="Mid-depth in full arrangements, front in piano-led pieces",
        common_processing=["HPF at 60-80 Hz in full mixes", "gentle compression", "EQ for clarity in context", "reverb for space and depth"],
        genre_variations={
            "jazz": "Wide, natural, dynamic, minimal compression, real acoustic piano",
            "pop": "Bright, compressed, often sits behind vocals as harmonic support",
            "classical": "Full dynamic range, natural room, minimal processing",
            "hip-hop": "Sampled, often filtered or lo-fi processed, rhythmic chops",
            "R&B": "Rhodes or Wurlitzer, warm, slight chorus, mid-depth in the mix",
        },
    ),
    MixPosition(
        instrument="Synth Pad",
        frequency_home="Warm body at 200-800 Hz, presence at 1-4 kHz, air/shimmer at 5-12 kHz",
        typical_panning="Wide stereo spread, filling the sides",
        depth_position="Back of the mix, atmospheric bed",
        common_processing=["HPF at 100-200 Hz to keep out of bass range", "LPF for darkness if needed", "reverb and delay for depth", "sidechain for movement", "subtle chorus or ensemble"],
        genre_variations={
            "ambient": "Lush, reverb-heavy, evolving filters, very wide and deep",
            "pop": "Subtle, supportive, filtered to not compete with vocals",
            "synthwave": "Prominent, analog character, moderate reverb, warm and wide",
            "electronic": "Sidechained, rhythmic, heavily modulated and evolving",
        },
    ),
    MixPosition(
        instrument="Synth Lead",
        frequency_home="Fundamental at 200 Hz-2 kHz depending on pitch, harmonics up to 8 kHz",
        typical_panning="Center or slightly off-center",
        depth_position="Front of the mix during lead sections, competing with vocals",
        common_processing=["EQ for presence and clarity", "compression for consistency", "delay for width and depth", "distortion/saturation for character"],
        genre_variations={
            "EDM": "Bright, aggressive, heavily processed, super-saw or FM tones",
            "synthwave": "Analog-style, warm, moderate effects, retro character",
            "pop": "Clean, bright, sits alongside vocals, automated for impact",
            "trance": "Soaring, heavily reverbed, delayed, emotional and prominent",
        },
    ),
    MixPosition(
        instrument="Lead Vocals",
        frequency_home="Fundamental at 100-400 Hz, presence at 2-5 kHz, air at 8-12 kHz, sibilance at 4-9 kHz",
        typical_panning="Dead center",
        depth_position="Front and center, the most prominent element",
        common_processing=["HPF at 80-120 Hz", "de-essing", "compression (often serial)", "EQ for presence and clarity", "reverb and delay for depth", "automation for consistency"],
        genre_variations={
            "pop": "Heavily processed, compressed, tuned, bright and present, effects automated",
            "rock": "Dynamic, moderate compression, distortion for energy, room reverb",
            "jazz": "Natural, minimal compression, real room ambience, dynamic range preserved",
            "hip-hop": "Dry, present, tuned, often doubled or ad-libbed, close-mic'd",
            "R&B": "Warm, smooth, layered harmonies, plate reverb, tasteful delay",
        },
    ),
    MixPosition(
        instrument="Backing Vocals",
        frequency_home="Similar to lead but often HPF higher (150-200 Hz), presence at 2-4 kHz",
        typical_panning="Spread wide, doubled and hard-panned, or arranged in stereo clusters",
        depth_position="Behind lead vocals, mid-depth, supportive",
        common_processing=["HPF higher than lead", "compression for blend", "EQ to avoid masking lead", "more reverb than lead", "chorus for thickening"],
        genre_variations={
            "pop": "Tightly tuned, layered, wide stereo spread, blended into a pad",
            "gospel": "Dynamic, natural, wide harmonies, powerful and present",
            "rock": "Raw, energetic, gang-vocal style for choruses, moderate processing",
            "R&B": "Lush stacks, tightly arranged, smooth and blended with lead",
        },
    ),
    MixPosition(
        instrument="Strings (Orchestral)",
        frequency_home="Low strings 80-500 Hz, mid strings 200 Hz-2 kHz, high strings 500 Hz-8 kHz, bow noise 2-6 kHz",
        typical_panning="Orchestral seating: violins left, violas center-left, cellos center-right, basses right (or reversed)",
        depth_position="Mid to back of mix, atmospheric and supportive in pop; front in classical",
        common_processing=["Gentle EQ for room correction", "minimal compression in classical", "reverb for hall space", "HPF in pop/rock contexts"],
        genre_variations={
            "classical": "Natural, wide, full dynamic range, real hall acoustics, minimal processing",
            "pop": "Compressed, bright, supporting harmonic role, reverb-blended",
            "film": "Wide, dynamic, large hall reverb, emotionally expressive",
            "hip-hop": "Sampled, filtered, often pitched or time-stretched, textural role",
        },
    ),
    MixPosition(
        instrument="Brass",
        frequency_home="Body at 200-800 Hz, presence/bite at 1-4 kHz, air at 5-8 kHz",
        typical_panning="Orchestral seating or arranged in stereo pairs, center for solo",
        depth_position="Mid-depth in arrangements, front for solos or stabs",
        common_processing=["EQ for brightness or warmth", "compression for consistency in pop", "reverb for blend", "HPF at 80-100 Hz"],
        genre_variations={
            "jazz": "Natural, dynamic, minimal processing, real room, front of mix for solos",
            "pop/funk": "Bright, punchy, compressed, tight arrangements, moderate reverb",
            "hip-hop": "Sampled, filtered, often distorted or pitched, stab-heavy",
            "orchestral": "Natural dynamics, large hall reverb, sectional seating panned",
        },
    ),
    MixPosition(
        instrument="Percussion (Auxiliary)",
        frequency_home="Varies widely: congas 100-800 Hz, shakers 4-12 kHz, tambourine 2-10 kHz",
        typical_panning="Spread across the stereo field for width and interest",
        depth_position="Mid to back of mix, rhythmic texture and decoration",
        common_processing=["HPF appropriate to instrument", "light compression", "EQ for clarity", "reverb matched to main drums", "transient shaping"],
        genre_variations={
            "Latin": "Prominent, natural, dynamic, wide stereo placement, minimal processing",
            "pop": "Supporting role, processed for clarity, filtered for space",
            "electronic": "Heavily processed, layered, filtered, rhythmically programmed",
            "world": "Natural, authentic, room ambience, dynamic, prominent",
        },
    ),
]


# ---------------------------------------------------------------------------
# Quality Markers
# ---------------------------------------------------------------------------

_QUALITY_MARKERS: list[QualityMarker] = [
    QualityMarker(
        marker="Low-End Clarity",
        good_indicator="Tight, defined bass with clear separation between sub, bass, and low-mids; kick and bass occupy distinct frequency ranges",
        bad_indicator="Muddy, boomy low end where bass instruments mask each other; undefined sub with excessive rumble",
        frequency_relevance="20-300 Hz",
        importance=0.95,
    ),
    QualityMarker(
        marker="Stereo Balance",
        good_indicator="Balanced, symmetrical stereo image with energy distributed evenly; strong mono compatibility",
        bad_indicator="Lopsided image with more energy on one side; excessive width causing mono collapse",
        frequency_relevance="Full spectrum, especially 200 Hz-8 kHz",
        importance=0.85,
    ),
    QualityMarker(
        marker="Dynamic Range",
        good_indicator="Appropriate dynamics for the genre with preserved transients and musical breathing; peak-to-average ratio suits the style",
        bad_indicator="Over-compressed, fatiguing master with no dynamic variation; or uncontrolled peaks with inconsistent levels",
        frequency_relevance="Full spectrum (measured as crest factor and LUFS range)",
        importance=0.90,
    ),
    QualityMarker(
        marker="Frequency Balance",
        good_indicator="Smooth, even frequency response with no harsh peaks or dull valleys; translates well across playback systems",
        bad_indicator="Unbalanced spectrum with harsh 2-5 kHz buildup, missing air, or boomy 200-400 Hz mud",
        frequency_relevance="Full spectrum (20 Hz-20 kHz)",
        importance=0.95,
    ),
    QualityMarker(
        marker="Transient Definition",
        good_indicator="Clear, defined transients that give drums punch and instruments articulation without being harsh",
        bad_indicator="Smeared, soft transients from over-compression, or harsh, clicky transients from over-processing",
        frequency_relevance="1-10 kHz (transient content), full spectrum for envelope",
        importance=0.80,
    ),
    QualityMarker(
        marker="Noise Floor",
        good_indicator="Clean silence between passages; no audible hiss, hum, or digital artifacts; appropriate room tone if present",
        bad_indicator="Audible hiss, 50/60 Hz hum, digital clicks, or excessive room noise interfering with the music",
        frequency_relevance="Full spectrum, especially 50/60 Hz (hum) and above 4 kHz (hiss)",
        importance=0.75,
    ),
    QualityMarker(
        marker="Phase Coherence",
        good_indicator="Tight, focused stereo image with no comb filtering; all elements maintain integrity in mono",
        bad_indicator="Hollow, phasey sound from poor mic placement or excessive stereo processing; mono cancellation",
        frequency_relevance="Full spectrum, especially critical in bass (below 300 Hz)",
        importance=0.80,
    ),
    QualityMarker(
        marker="Vocal Clarity",
        good_indicator="Vocals sit clearly above the mix with intelligible lyrics; present without harshness; well-controlled dynamics",
        bad_indicator="Buried, masked vocals fighting instruments for space; or excessively harsh, sibilant vocals that fatigue the ear",
        frequency_relevance="200 Hz-8 kHz (vocal fundamental and harmonics)",
        importance=0.95,
    ),
    QualityMarker(
        marker="Separation/Definition",
        good_indicator="Each instrument occupies its own space in frequency, stereo field, and depth; elements are individually distinguishable",
        bad_indicator="Congested, cluttered mix where instruments mask each other; inability to distinguish individual elements",
        frequency_relevance="Full spectrum (frequency separation), stereo field, and depth dimension",
        importance=0.90,
    ),
    QualityMarker(
        marker="Loudness Consistency",
        good_indicator="Even loudness across sections; verse-to-chorus transitions feel natural; no sudden jumps or drops",
        bad_indicator="Jarring level changes between sections; choruses too loud or verses too quiet; uneven automation",
        frequency_relevance="Full spectrum (measured as short-term LUFS variation)",
        importance=0.75,
    ),
]


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

_ALL_TECHNIQUES: list[ProductionTechnique] | None = None


def _build_all() -> list[ProductionTechnique]:
    global _ALL_TECHNIQUES
    if _ALL_TECHNIQUES is None:
        _ALL_TECHNIQUES = (
            _DYNAMICS + _EQ + _SPATIAL + _DISTORTION
            + _MODULATION + _STEREO + _CREATIVE + _MASTERING
        )
    return _ALL_TECHNIQUES


def get_all_techniques() -> list[ProductionTechnique]:
    """Return every registered ProductionTechnique."""
    return list(_build_all())


def get_techniques_by_category(category: str) -> list[ProductionTechnique]:
    """Return techniques filtered by category name.

    Valid categories: dynamics, eq, spatial, time_based, distortion,
    modulation, stereo, creative, mastering.
    """
    return [t for t in _build_all() if t.category == category]


def get_mix_positions() -> list[MixPosition]:
    """Return all MixPosition entries."""
    return list(_MIX_POSITIONS)


def get_quality_markers() -> list[QualityMarker]:
    """Return all QualityMarker entries."""
    return list(_QUALITY_MARKERS)


# ---------------------------------------------------------------------------
# CLAP description generation
# ---------------------------------------------------------------------------

def _technique_descriptions(t: ProductionTechnique) -> list[tuple[str, dict]]:
    """Generate 2-3 CLAP text descriptions for a single technique."""
    descs: list[tuple[str, dict]] = []
    labels = {
        "technique": t.name,
        "category": t.category,
    }

    # Description 1: core sonic effect
    descs.append((
        f"{t.sonic_effect.rstrip('.')} — characteristic of {t.name.lower()} processing.",
        {**labels, "type": "sonic_effect"},
    ))

    # Description 2: good-quality application
    good = t.quality_markers.get("good", "")
    if good:
        genre_str = t.genre_associations[0] if t.genre_associations else "modern"
        descs.append((
            f"A {genre_str} track with expertly applied {t.name.lower()}: {good.lower().rstrip('.')}.",
            {**labels, "type": "good_quality"},
        ))

    # Description 3: bad-quality application
    bad = t.quality_markers.get("bad", "")
    if bad:
        descs.append((
            f"A poorly mixed track suffering from bad {t.name.lower()} usage: {bad.lower().rstrip('.')}.",
            {**labels, "type": "bad_quality"},
        ))

    return descs


def _mix_position_descriptions(mp: MixPosition) -> list[tuple[str, dict]]:
    """Generate CLAP descriptions from mix position knowledge."""
    descs: list[tuple[str, dict]] = []
    labels = {"instrument": mp.instrument, "category": "mix_position"}

    descs.append((
        f"A {mp.instrument.lower()} sitting {mp.depth_position.lower()} panned {mp.typical_panning.lower()}, "
        f"occupying {mp.frequency_home.split(',')[0].lower().strip()}.",
        {**labels, "type": "position"},
    ))

    for genre, desc in mp.genre_variations.items():
        descs.append((
            f"A {genre.lower()} {mp.instrument.lower()}: {desc.lower().rstrip('.')}.",
            {**labels, "type": "genre_variation", "genre": genre},
        ))

    return descs


def _quality_marker_descriptions(qm: QualityMarker) -> list[tuple[str, dict]]:
    """Generate CLAP descriptions from quality markers."""
    labels = {"marker": qm.marker, "category": "quality"}
    return [
        (
            f"A professionally produced track demonstrating excellent {qm.marker.lower()}: "
            f"{qm.good_indicator.lower().rstrip('.')}.",
            {**labels, "type": "good"},
        ),
        (
            f"A poorly produced track with problematic {qm.marker.lower()}: "
            f"{qm.bad_indicator.lower().rstrip('.')}.",
            {**labels, "type": "bad"},
        ),
    ]


# Additional hand-crafted CLAP descriptions that combine multiple concepts
_COMPOSITE_DESCRIPTIONS: list[tuple[str, dict]] = [
    (
        "A heavily compressed vocal with fast attack and slow release creating a thick, present sound.",
        {"technique": "FET Compression", "instrument": "vocals", "category": "composite"},
    ),
    (
        "A kick drum with sidechain compression pumping the synth pad, typical of EDM production.",
        {"technique": "Sidechain Compression", "genre": "EDM", "category": "composite"},
    ),
    (
        "A warm analog tape saturation on a bass guitar adding harmonic richness.",
        {"technique": "Tape Saturation", "instrument": "bass", "category": "composite"},
    ),
    (
        "A professionally mastered track with balanced frequency spectrum and -14 LUFS loudness.",
        {"technique": "LUFS Targeting", "category": "composite"},
    ),
    (
        "A poorly mixed track with muddy low-mids and harsh high frequencies.",
        {"marker": "Frequency Balance", "category": "composite", "type": "bad"},
    ),
    (
        "Bright, shimmering plate reverb on a snare drum adding depth and presence to a pop mix.",
        {"technique": "Plate Reverb", "instrument": "snare", "genre": "pop", "category": "composite"},
    ),
    (
        "A heavily distorted electric guitar with tight low end and aggressive midrange bite in a metal mix.",
        {"technique": "Distortion", "instrument": "electric guitar", "genre": "metal", "category": "composite"},
    ),
    (
        "Lush stereo chorus on a clean electric guitar creating a wide, shimmering 80s pop texture.",
        {"technique": "Chorus", "instrument": "electric guitar", "genre": "80s pop", "category": "composite"},
    ),
    (
        "Dub-style delay with high feedback and filtering on a vocal, creating hypnotic echo trails.",
        {"technique": "Dub Delay", "instrument": "vocals", "genre": "dub", "category": "composite"},
    ),
    (
        "A tight, punchy hip-hop mix with sidechained 808 bass, crisp hi-hats, and a present, dry vocal.",
        {"genre": "hip-hop", "category": "composite"},
    ),
    (
        "An ambient soundscape with granular-processed piano, shimmer reverb, and wide stereo imaging.",
        {"genre": "ambient", "category": "composite"},
    ),
    (
        "A lo-fi beat with bitcrushed drums, tape wow and flutter, and a low-pass filtered sample.",
        {"genre": "lo-fi", "category": "composite"},
    ),
    (
        "Hard-tuned vocals with robotic pitch correction, typical of modern trap and pop production.",
        {"technique": "Vocal Tuning", "genre": "trap", "category": "composite"},
    ),
    (
        "A jazz trio recording with natural room reverb, wide stereo piano, and dynamic, uncompressed performance.",
        {"genre": "jazz", "category": "composite"},
    ),
    (
        "Gated reverb on a snare drum creating the iconic 1980s power-ballad drum sound.",
        {"technique": "Gated Reverb", "instrument": "snare", "genre": "80s", "category": "composite"},
    ),
    (
        "A vocal processed through a vocoder with a rich synthesizer carrier, creating a robotic singing texture.",
        {"technique": "Vocoder", "instrument": "vocals", "category": "composite"},
    ),
    (
        "Parallel compression on a drum bus adding massive body and sustain while preserving natural transients.",
        {"technique": "Parallel Compression", "instrument": "drums", "category": "composite"},
    ),
    (
        "A mastered track with subtle multiband compression, transparent limiting, and -14 LUFS integrated loudness.",
        {"category": "composite", "stage": "mastering"},
    ),
    (
        "Mid-side mastering EQ with tightened center bass and enhanced side brightness for stereo width.",
        {"technique": "Mid/Side Mastering", "category": "composite"},
    ),
    (
        "A congested mix where guitars, synths, and vocals compete in the 1-4 kHz range with poor separation.",
        {"marker": "Separation/Definition", "category": "composite", "type": "bad"},
    ),
]


def generate_clap_descriptions() -> list[tuple[str, dict]]:
    """Generate all CLAP text descriptions for training.

    Returns a list of (text_description, labels_dict) tuples combining:
    - Per-technique descriptions (2-3 each)
    - Per-mix-position descriptions
    - Per-quality-marker descriptions (good and bad)
    - Hand-crafted composite descriptions
    """
    descriptions: list[tuple[str, dict]] = []

    for t in _build_all():
        descriptions.extend(_technique_descriptions(t))

    for mp in _MIX_POSITIONS:
        descriptions.extend(_mix_position_descriptions(mp))

    for qm in _QUALITY_MARKERS:
        descriptions.extend(_quality_marker_descriptions(qm))

    descriptions.extend(_COMPOSITE_DESCRIPTIONS)

    return descriptions


# ---------------------------------------------------------------------------
# Quick stats (useful for debugging / logging)
# ---------------------------------------------------------------------------

def summary() -> dict:
    """Return a summary of the knowledge base contents."""
    techniques = _build_all()
    categories = {}
    for t in techniques:
        categories[t.category] = categories.get(t.category, 0) + 1

    clap = generate_clap_descriptions()
    return {
        "total_techniques": len(techniques),
        "techniques_by_category": categories,
        "mix_positions": len(_MIX_POSITIONS),
        "quality_markers": len(_QUALITY_MARKERS),
        "total_clap_descriptions": len(clap),
    }


if __name__ == "__main__":
    s = summary()
    print("RESONATE Production Techniques Knowledge Base")
    print("=" * 50)
    print(f"Total techniques:        {s['total_techniques']}")
    print(f"Mix positions:           {s['mix_positions']}")
    print(f"Quality markers:         {s['quality_markers']}")
    print(f"Total CLAP descriptions: {s['total_clap_descriptions']}")
    print()
    print("Techniques by category:")
    for cat, count in sorted(s["techniques_by_category"].items()):
        print(f"  {cat:20s} {count}")
    print()
    print("Sample CLAP descriptions:")
    for desc, labels in generate_clap_descriptions()[:5]:
        print(f"  - {desc}")
        print(f"    labels: {labels}")

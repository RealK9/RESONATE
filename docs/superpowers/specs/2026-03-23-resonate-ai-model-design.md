# RESONATE AI Model System — Full Design Specification

## Overview

A complete AI-powered audio analysis and recommendation system for RESONATE. The system analyzes every sample in the library with rich feature extraction, understands uploaded mixes without requiring stems, compares against commercial reference priors, recommends samples that *improve* a mix (not just match it), and learns from user behavior to become uniquely valuable over time.

**Architecture**: Hybrid — local DSP feature extraction on the desktop app, heavy ML inference (embeddings, classification, perceptual models) on a self-hosted FastAPI + GPU server.

**Database**: PostgreSQL + pgvector — replaces SQLite entirely. Stores structured features, embedding vectors, sample metadata, reference profiles, user preferences, and feedback data.

**Pretrained model bundle**: CLAP (general audio-text similarity), PANNs (audio tagging/embedding), AST (spectrogram transformer representations), plus custom classifiers trained on role labels.

---

## Directory Structure

Two codebases — the existing desktop app backend and a new ML inference service:

```
backend/                          # Desktop app backend (existing, modified)
├── analysis/
│   ├── dsp_features.py           # Core + spectral descriptors (local DSP)
│   ├── pitch_harmony.py          # Harmonic/pitch descriptors (local DSP)
│   ├── transients.py             # Transient detection, attack/decay (local DSP)
│   ├── mix_profile.py            # Mix-level analysis (local DSP)
│   ├── needs_engine.py           # Deficiency/opportunity diagnosis
│   └── reference_profiles.py     # Reference corpus analysis and comparison
├── db/
│   ├── models.py                 # SQLAlchemy + pgvector ORM models
│   ├── migrations/               # Alembic migration scripts
│   └── database.py               # PostgreSQL connection, session management
├── retrieval/
│   └── index_library.py          # Ingestion pipeline, DB indexing, vector search
├── recommendation/
│   ├── candidate_generator.py    # Candidate pool narrowing
│   ├── reranker.py               # Ranking model (XGBoost → neural)
│   └── explanations.py           # Human-readable recommendation reasoning
├── training/
│   ├── preference_dataset.py     # Feedback collection and dataset construction
│   └── train_ranker.py           # Model training pipeline
├── ml_client.py                  # Client SDK to call ML service API
└── ...                           # Existing app.py, routes/, etc.

ml_service/                       # New: self-hosted GPU inference service
├── src/
│   ├── api/
│   │   ├── app.py                # FastAPI app factory
│   │   ├── routes.py             # Inference endpoints
│   │   └── schemas.py            # Request/response Pydantic models
│   ├── embeddings/
│   │   ├── extract_embeddings.py # CLAP, PANNs, AST, timbre extraction
│   │   └── model_registry.py     # Model loading, versioning, GPU management
│   ├── classifiers/
│   │   ├── role_classifier.py    # Sound role prediction
│   │   ├── genre_classifier.py   # Genre/era affinity
│   │   ├── style_classifier.py   # Style cluster classification
│   │   ├── quality_scorer.py     # Commercial readiness scoring
│   │   └── tag_predictor.py      # Style tag prediction
│   ├── perceptual/
│   │   └── perceptual_model.py   # Learned perceptual descriptors
│   └── training/
│       ├── style_priors.py       # Learning priors from style clusters
│       └── preference_serving.py # Deploying learned preference models
├── Dockerfile                    # GPU container specification
├── requirements.txt              # ML dependencies (torch, transformers, etc.)
└── config.py                     # Model paths, GPU config, service settings
```

---

## Unified Role Taxonomy

A single canonical role set used across all modules. Sample roles map to mix roles for cross-module compatibility.

### Canonical Roles (23 roles)
| Role | Description | Category |
|------|-------------|----------|
| `kick` | Kick drums | Percussion |
| `snare` | Snare drums | Percussion |
| `clap` | Claps, snaps | Percussion |
| `hat` | Hi-hats, cymbals | Percussion |
| `perc` | Other percussion, shakers, tambourines | Percussion |
| `bass` | Bass sounds, sub-bass | Tonal |
| `lead` | Lead melodies, toplines | Tonal |
| `chord` | Chord stabs, chord progressions, harmonic support | Tonal |
| `pad` | Pads, sustained textures | Tonal |
| `vocal` | Vocals, vocal chops, vocal textures | Tonal |
| `fx` | Risers, downlifters, impacts, transitions | FX |
| `texture` | Noise, foley, ambient layers, glue textures | FX |
| `ambience` | Room tones, reverb tails, spatial layers | FX |

### Mix-Role Mapping
When analyzing a mix, source-role inference maps to the same taxonomy:
- `kick` → kick presence/confidence
- `snare` + `clap` → snare/clap combined presence (often indistinguishable in a mix)
- `hat` + `perc` → tops/hats combined presence
- `bass` → bass presence
- `lead` → lead presence
- `chord` → chord support presence
- `pad` → pad presence
- `vocal` → vocal texture presence
- `fx` → FX/transitions presence
- `texture` + `ambience` → ambience/texture combined presence

This ensures the recommendation engine can directly map "needs chord support" to samples tagged with role `chord`.

---

## Perceptual Descriptors — Canonical List

A single list used across Module 1, Module 2, and the Labels Taxonomy:
- **brightness** — spectral center of gravity, high-frequency presence
- **warmth** — low-mid energy, harmonic richness in 200-800Hz
- **air** — ultra-high frequency presence (10kHz+), openness
- **punch** — transient impact, attack strength relative to body
- **body** — mid-frequency fullness, sustain weight
- **bite** — upper-mid presence (2-5kHz), edge, aggression
- **harshness** — resonant peaks in 2-6kHz, unpleasant upper-mid buildup
- **smoothness** — absence of sharp transients, gentle spectral rolloff
- **width** — stereo image spread, mid/side ratio
- **depth** — front-to-back impression, reverb/delay presence
- **grit** — distortion artifacts, saturation harmonics, lo-fi texture

---

## Module 1: Sample Analyzer

For every sound in the library, compute a comprehensive profile.

### Core Descriptors
- duration
- sample rate
- mono/stereo
- transient positions
- onset strength
- RMS / LUFS / peak
- crest factor
- attack / decay / sustain profile
- loop detection confidence (one-shot vs loop)

### Spectral Descriptors
- spectral centroid
- rolloff
- flatness
- contrast
- bandwidth
- skewness / kurtosis
- high-band harshness zones
- low-end energy distribution
- sub-to-bass ratio
- resonant peak analysis
- MFCC statistics (13 coefficients — mean, std, delta)

### Harmonic / Pitch Descriptors
- F0 estimate
- pitch confidence
- chroma profile
- harmonic-to-noise ratio
- inharmonicity
- overtone slope
- tonalness / noisiness
- dissonance / roughness

### Perceptual Descriptors
Computed directly where possible, inferred via learned models otherwise. Uses the canonical perceptual list: brightness, warmth, air, punch, body, bite, harshness, smoothness, width, depth, grit.

### Learned Embeddings
Multiple embedding spaces, not a single vector:
- **General audio embedding** (CLAP) — ~512d
- **Music-focused embedding** (PANNs) — ~2048d
- **Transient/percussion-focused embedding** (AST fine-tuned or custom) — ~768d
- **Timbre embedding** (custom encoder if possible) — ~256d

### Predicted Labels
- **Sound role**: one of the 13 canonical roles (probability distribution)
- **Tonal vs atonal** (confidence score)
- **One-shot vs loop** (confidence score, cross-referenced with loop detection)
- **Genre affinity**: probability distribution across: EDM, trap, pop, house, cinematic, hip-hop, Afro house, melodic techno, R&B, drum & bass, ambient, experimental
- **Era affinity**: probability distribution across decades (1970s, 1980s, 1990s, 2000s, 2010s, 2020s)
- **Commercial readiness score** (0.0–1.0): measures production quality, clarity, mixability, and professional polish. Trained on curated "production-ready" vs "raw/unfinished" sample pairs. Consumed by Module 4 as the QualityPrior term in the scoring function.
- **Style tags**: bright, dark, wide, punchy, analog, digital, gritty, clean, lo-fi, saturated, crisp, metallic, organic, synthetic

### Files (backend/)
- `analysis/dsp_features.py` — core + spectral descriptors (local DSP)
- `analysis/pitch_harmony.py` — harmonic/pitch descriptors (local DSP)
- `analysis/transients.py` — transient detection, attack/decay profiling (local DSP)

### Files (ml_service/)
- `src/embeddings/extract_embeddings.py` — CLAP, PANNs, AST, timbre embeddings
- `src/classifiers/role_classifier.py` — sound role prediction
- `src/classifiers/genre_classifier.py` — genre/era affinity
- `src/classifiers/tag_predictor.py` — style tag prediction
- `src/classifiers/quality_scorer.py` — commercial readiness score
- `src/perceptual/perceptual_model.py` — learned perceptual descriptors

### Files (shared)
- `backend/retrieval/index_library.py` — ingestion pipeline, database indexing, vector index
- `backend/ml_client.py` — client SDK to call ML service

---

## Module 2: Mix Analyzer

Given an uploaded mix (polyphonic, multi-source), create a mix context profile without requiring stem separation.

### Mix-Level Analysis
- BPM / tempo confidence
- key / tonal center
- harmonic density
- section energy evolution
- overall loudness / dynamics
- stereo width by band
- spectral occupancy by band over time

### Source-Role Inference
Estimate presence and confidence for each canonical role (using the mix-role mapping defined above):
- kick
- snare / clap (combined)
- hats / tops (combined)
- bass
- lead
- chord support
- pad
- vocal texture
- FX / transitions
- ambience / texture (combined)

### Deficiency / Opportunity Analysis
The diagnostic heart of the system. Identifies:
- top-end too sparse
- upper mids overcrowded
- weak attack support
- no glue texture
- low-end too broad
- no rhythmic sparkle
- harmonic layer too thin
- no lift into transitions
- width imbalance
- lack of emotional support layers

This is not just measurement — it is diagnosis. The needs engine translates spectral/temporal measurements into actionable production insights.

### Needs Engine Outputs
Each need maps to one or more canonical roles for recommendation:
- more top sparkle needed → `hat`, `perc`, `fx`
- avoid more sub → penalty on `bass`, `kick`
- needs glue texture → `texture`, `ambience`
- needs transitional FX → `fx`
- needs chord support → `chord`, `pad`
- needs transient reinforcement → `kick`, `snare`, `clap`, `perc`
- needs width layer → `pad`, `texture`, `ambience`
- needs high-mid restraint → penalty on `lead`, `vocal` in 2-5kHz

### Style Classification
Classifies the uploaded mix into one or more style clusters (probability distribution). **This output feeds directly into Module 3** — the style cluster probabilities determine which reference priors to compare against.

### Files (backend/)
- `analysis/mix_profile.py` — BPM, key, spectral occupancy, density, width maps (local DSP)
- `analysis/needs_engine.py` — deficiency/opportunity detection (rule-based + learned)

### Files (ml_service/)
- `src/classifiers/style_classifier.py` — style cluster classification (used by both Module 2 and Module 3)

---

## Module 3: Commercial Reference Engine

Style-aware reference model learned from successful commercial music.

### Inputs
- Curated reference songs
- Metadata: genre, era, mood, section

### Style Clusters
Build from genres that matter to the target audience:
- 2010s commercial EDM drops
- 2020s melodic house
- 2000s pop choruses
- 1990s boom bap drums
- modern trap toplines
- cinematic
- Afro house
- melodic techno
- hip-hop
- pop

A universal "industry standard" is a mirage. A style-conditional standard is real. Start with curated style buckets, not "everything."

### Learned Priors Per Cluster
- typical role co-occurrence
- spectral balance norms
- transient density
- width patterns
- arrangement density
- tonal complexity
- layering depth
- common mix trajectories over a section
- common complementary elements
- section-lift patterns

### Operation
When a user uploads a mix:
1. **Module 2's style_classifier** classifies the mix into style cluster(s) with probabilities
2. Module 3 retrieves the matching reference priors for those clusters
3. Compares the mix profile against the cluster priors
4. Generates style-contextualized diagnostics (feeds into needs_engine)

### Files (backend/)
- `analysis/reference_profiles.py` — reference corpus analysis, storage, and comparison

### Files (ml_service/)
- `src/training/style_priors.py` — learning priors from style clusters
- `src/classifiers/style_classifier.py` — (shared with Module 2)

---

## Module 4: Recommendation Engine

Two-stage system: candidate generation followed by learned reranking.

### Stage 1: Candidate Generation
Generate candidates by:
- sound role needed (from needs engine, mapped via canonical role taxonomy)
- tonal compatibility (key/pitch match)
- groove compatibility (BPM proximity, rhythmic alignment)
- style match (genre/era affinity overlap with mix style clusters)
- commercial quality floor (minimum commercial readiness score threshold)
- avoid already-overoccupied bands (spectral masking check)

**Database query strategy**: Hybrid query combining structured filters (role, key, BPM range, quality threshold) with vector similarity search (embedding cosine distance in the most relevant embedding space for the need type). Candidate pool size: top 200 before reranking.

### Stage 2: Reranking
Rank candidates using a model that predicts **improvement**, not similarity.

Input features:
- uploaded mix profile (spectral, role presence, needs vector)
- candidate sample profile (all descriptors + embeddings)
- pairwise compatibility metrics (spectral complement, tonal distance, rhythmic alignment)
- style priors (from Module 3 reference engine)
- quality priors (commercial readiness score)
- masking penalties (spectral overlap with existing mix content)
- novelty penalties (redundancy with already-recommended samples)
- arrangement-role logic (what roles are already covered)

This is a learning-to-rank problem.

### Reranker Model Progression
1. Start with structured feature model: XGBoost / LightGBM / CatBoost
2. Then add neural ranking model
3. Then add pairwise preference training from Module 5

### Scoring Function
```
Score = α·NeedFit + β·RoleFit + γ·SpectralComplement + δ·TonalCompatibility
      + ε·RhythmicCompatibility + ζ·StylePriorFit + η·QualityPrior
      + θ·UserPreferencePrior − λ·MaskingPenalty − μ·RedundancyPenalty
```

Begins partly rule-based, then learned. The `QualityPrior` term uses the commercial readiness score from Module 1. The `UserPreferencePrior` term uses the learned preference model from Module 5 (zero until Module 5 is active).

### Files (backend/)
- `recommendation/candidate_generator.py` — candidate pool narrowing (DB queries + vector search)
- `recommendation/reranker.py` — ranking model (XGBoost → neural)
- `recommendation/explanations.py` — human-readable recommendation reasoning

---

## Module 5: Feedback Engine

Human preference learning — this is the proprietary moat.

### Collected Signals
- clicked sample
- auditioned sample
- dragged into DAW
- kept vs discarded
- paired with which genre/style
- user rating after insertion
- accepted / rejected / skipped recommendations
- audition counts
- save/ignore actions
- inserted samples by context
- genre tags
- post-insertion approval

### Training Data Format
For training recommendation models:
- "in this context, sample A is better than B" (pairwise preferences)
- accepted / rejected / skipped (pointwise labels)
- improved / worsened / neutral (outcome labels)

### Training Targets
- ranking models (improve reranker)
- per-user taste adaptation
- style-specific recommendation policies

### Inference / Serving
The trained preference model is deployed as a scoring component that feeds into Module 4's reranker via the `UserPreferencePrior` term. The preference serving module loads the latest trained model and provides per-user bias scores for candidate samples.

### Files (backend/)
- `training/preference_dataset.py` — feedback collection and dataset construction

### Files (ml_service/)
- `src/training/preference_serving.py` — deploying learned preference models for inference
- `backend/training/train_ranker.py` — model training pipeline (runs offline, produces model artifacts)

---

## Labels Taxonomy

### Per Sample — Enumerated Values

**Role** (13 values): kick, snare, clap, hat, perc, bass, lead, chord, pad, vocal, fx, texture, ambience

**Genre affinity** (12 values): EDM, trap, pop, house, cinematic, hip-hop, Afro house, melodic techno, R&B, drum & bass, ambient, experimental

**Era affinity** (6 values): 1970s, 1980s, 1990s, 2000s, 2010s, 2020s

**Quality tier** (derived from commercial readiness score):
- platinum: 0.85–1.0 (production-ready, professional)
- gold: 0.65–0.84 (good quality, may need minor processing)
- silver: 0.40–0.64 (usable, needs processing)
- raw: 0.0–0.39 (unprocessed, rough)

**Style tags** (14 values): bright, dark, wide, punchy, analog, digital, gritty, clean, lo-fi, saturated, crisp, metallic, organic, synthetic

**Perceptual traits** (11 values): brightness, warmth, air, punch, body, bite, harshness, smoothness, width, depth, grit (each 0.0–1.0)

**Tonal/atonal**: boolean with confidence (0.0–1.0)

**One-shot/loop**: boolean with confidence (0.0–1.0), cross-referenced with loop detection from core descriptors

### Per Uploaded Mix
- style cluster (probability distribution across defined clusters)
- role presence (confidence 0.0–1.0 for each mix-role group)
- quality diagnostics (per-band spectral health scores)
- missing-role opportunities (list of canonical roles with need-strength 0.0–1.0)

### Per Recommendation (training)
- in this context, sample A is better than B (pairwise)
- accepted / rejected / skipped (pointwise)
- improved / worsened / neutral (outcome)

---

## Pretrained Models

Priority: audio encoders, not generators.

| Model | Purpose | Embedding Dim | GPU Memory |
|-------|---------|---------------|------------|
| **CLAP** (LAION-AI/CLAP) | General audio-text/audio similarity space | ~512d | ~2GB |
| **PANNs** (Cnn14) | Robust audio tagging/embedding baselines | ~2048d | ~1GB |
| **AST** (MIT/ast-finetuned-audioset) | Spectrogram-based transformer representations | ~768d | ~1.5GB |
| **Custom classifiers** | Role, genre, era, style tags, quality, perceptual | varies | ~0.5GB |

**Total GPU requirement**: ~6GB minimum for all models loaded. Recommend 8GB+ GPU (e.g., RTX 3070, A10G, L4).

Future: multimodal recommendation encoder over uploaded mix profile + candidate sample profile + style priors + theory features.

---

## Infrastructure

### Desktop App (existing Electron + Python backend)
- Local DSP feature extraction (fast, offline-capable)
- UI, playback, DAW bridge, user sessions
- Calls ML service via `ml_client.py` for heavy inference
- Writes/reads from PostgreSQL via SQLAlchemy + pgvector
- Runs recommendation engine locally (candidate generation + reranking)

### ML Service (new, self-hosted FastAPI + GPU)
- All embedding extraction (CLAP, PANNs, AST)
- Classification inference (role, genre, era, style tags, quality)
- Perceptual model inference (brightness, warmth, punch, etc.)
- Style cluster classification
- Preference model serving
- Stateless inference API with Pydantic request/response schemas
- Deployed on GPU hardware via Docker container
- Health check + model readiness endpoints

### Database (PostgreSQL + pgvector)
- Structured feature storage (all descriptors as typed columns)
- Embedding vector storage with pgvector indexes (one per embedding space)
- Sample metadata and predicted labels
- Reference profiles and style priors
- User sessions, ratings, feedback data
- Preference training data
- Managed via Alembic migrations

### Offline / Fallback Behavior
When the ML service is unreachable:
- Local DSP features are still extracted and stored (core, spectral, harmonic/pitch descriptors)
- Samples can be browsed and searched by DSP features only
- Embedding-based similarity search is unavailable
- Classification labels show as "pending" — queued for processing when ML service reconnects
- Recommendation engine runs in degraded mode using DSP features + rule-based scoring only
- A queue stores pending ML inference requests, processes them when connection restores

### Data Flow — Detailed
```
SAMPLE INGESTION:
  Audio file discovered
  → backend/analysis/dsp_features.py (local: core + spectral)
  → backend/analysis/pitch_harmony.py (local: harmonic/pitch)
  → backend/analysis/transients.py (local: transients + attack/decay)
  → backend/ml_client.py → ML Service API:
      → embeddings/extract_embeddings.py (CLAP, PANNs, AST, timbre)
      → classifiers/role_classifier.py (sound role)
      → classifiers/genre_classifier.py (genre + era affinity)
      → classifiers/tag_predictor.py (style tags)
      → classifiers/quality_scorer.py (commercial readiness)
      → perceptual/perceptual_model.py (brightness, warmth, etc.)
  → All results stored in PostgreSQL via backend/db/models.py
  → Vector indexes updated for each embedding space

MIX UPLOAD:
  User uploads mix
  → backend/analysis/mix_profile.py (local: BPM, key, spectral occupancy, density, width)
  → backend/ml_client.py → ML Service API:
      → classifiers/style_classifier.py (style cluster probabilities)
  → backend/analysis/reference_profiles.py (compare against style priors from Module 3)
  → backend/analysis/needs_engine.py (deficiency/opportunity diagnosis)
  → backend/recommendation/candidate_generator.py (query DB: structured + vector)
  → backend/recommendation/reranker.py (score top 200 → rank top N)
  → backend/recommendation/explanations.py (generate reasoning)
  → Results returned to frontend UI
  → User interactions collected → backend/training/preference_dataset.py → PostgreSQL

PREFERENCE LEARNING (offline):
  backend/training/train_ranker.py (periodic batch training)
  → Produces model artifacts
  → Deployed to ml_service/src/training/preference_serving.py
  → Reranker incorporates UserPreferencePrior on next inference
```

---

## Database Schema (PostgreSQL + pgvector)

### Core Tables

**samples**
- `id` (UUID, PK)
- `filepath` (text, unique)
- `filename` (text)
- `duration` (float)
- `sample_rate` (int)
- `channels` (int)
- Core descriptors: `rms`, `lufs`, `peak`, `crest_factor` (float)
- `attack_ms`, `decay_ms`, `sustain_level` (float)
- `onset_count` (int), `onset_strength_mean` (float)
- `is_loop` (bool), `loop_confidence` (float)
- Spectral descriptors: `spectral_centroid`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`, `spectral_bandwidth`, `spectral_skewness`, `spectral_kurtosis` (float)
- `harshness_zones` (jsonb), `low_end_distribution` (jsonb), `sub_bass_ratio` (float)
- `resonant_peaks` (jsonb)
- `mfcc_mean` (float[13]), `mfcc_std` (float[13])
- Harmonic: `f0_hz` (float), `pitch_confidence` (float), `chroma_profile` (float[12])
- `hnr` (float), `inharmonicity` (float), `overtone_slope` (float)
- `tonalness` (float), `noisiness` (float), `dissonance` (float)
- Perceptual: `brightness`, `warmth`, `air`, `punch`, `body`, `bite`, `harshness_score`, `smoothness`, `width`, `depth`, `grit` (float, 0.0–1.0)
- Labels: `role` (text), `role_probabilities` (jsonb), `tonal` (bool), `tonal_confidence` (float)
- `genre_affinity` (jsonb), `era_affinity` (jsonb)
- `commercial_readiness` (float), `quality_tier` (text)
- `style_tags` (text[])
- Embeddings: `emb_clap` (vector(512)), `emb_panns` (vector(2048)), `emb_ast` (vector(768)), `emb_timbre` (vector(256))
- `analyzed_at` (timestamp), `ml_analyzed_at` (timestamp, nullable — null if ML pending)

**mix_profiles**
- `id` (UUID, PK)
- `filepath` (text)
- `bpm` (float), `bpm_confidence` (float)
- `key` (text), `tonal_center` (text)
- `harmonic_density` (float)
- `section_energy` (jsonb) — time-series
- `loudness_lufs` (float), `dynamic_range` (float)
- `stereo_width_by_band` (jsonb)
- `spectral_occupancy` (jsonb) — band × time matrix
- `role_presence` (jsonb) — {role: confidence}
- `style_clusters` (jsonb) — {cluster: probability}
- `needs_vector` (jsonb) — {need: strength}
- `created_at` (timestamp)

**reference_profiles**
- `id` (UUID, PK)
- `style_cluster` (text)
- `role_cooccurrence` (jsonb)
- `spectral_norms` (jsonb)
- `transient_density` (float)
- `width_norms` (jsonb)
- `arrangement_density` (float)
- `tonal_complexity` (float)
- `layering_depth` (float)
- `section_trajectories` (jsonb)
- `complementary_elements` (jsonb)
- `sample_count` (int) — how many reference songs built this profile
- `updated_at` (timestamp)

**user_feedback**
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `session_id` (UUID, FK)
- `sample_id` (UUID, FK)
- `mix_id` (UUID, FK)
- `action` (text) — click, audition, drag, keep, discard, rate, skip
- `rating` (int, nullable)
- `context_genre` (text, nullable)
- `context_style_cluster` (text, nullable)
- `recommendation_rank` (int, nullable) — position in the recommendation list
- `created_at` (timestamp)

**preference_pairs** (training data)
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `mix_id` (UUID, FK)
- `preferred_sample_id` (UUID, FK)
- `rejected_sample_id` (UUID, FK)
- `context` (jsonb) — mix profile snapshot at time of preference
- `created_at` (timestamp)

**users**
- `id` (UUID, PK)
- `created_at` (timestamp)
- `preference_model_version` (int, nullable) — tracks which model version applies

**sessions**
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `mix_id` (UUID, FK, nullable)
- `recommendations` (jsonb) — snapshot of recommended samples + scores
- `created_at` (timestamp)

### Vector Indexes
- `idx_emb_clap` — IVFFlat index on `samples.emb_clap`
- `idx_emb_panns` — IVFFlat index on `samples.emb_panns`
- `idx_emb_ast` — IVFFlat index on `samples.emb_ast`
- `idx_emb_timbre` — IVFFlat index on `samples.emb_timbre`

---

## Build Phases

### Phase 1 — Foundation: Analyze Every Sample
**Goal**: Build a rich profile database for all stock sounds.

**Deliverables**:
- File ingestion pipeline
- Sample role classification baseline
- DSP feature extraction pipeline (all core, spectral, harmonic/pitch descriptors)
- Embedding extraction pipeline (CLAP, PANNs, AST)
- ML service with inference endpoints
- PostgreSQL schema with pgvector indexes
- Searchable database and vector index
- ML client SDK for desktop backend

**Minimum features per sample**:
- duration, peak, RMS, LUFS, crest factor
- mono/stereo width
- onset/transient stats, attack/decay length
- spectral centroid, rolloff, contrast, flatness, bandwidth
- MFCC statistics (13 coefficients)
- chroma if tonal, pitch estimate, pitch confidence
- harmonicity / noisiness, inharmonicity
- loop detection confidence
- embedding vectors (CLAP, PANNs, AST)
- sound role classification
- genre/era affinity
- style tags
- commercial readiness score
- perceptual descriptors (brightness, warmth, air, punch, body, bite, smoothness, width, depth, grit)

### Phase 2 — Mix Intelligence
**Goal**: Given a 30-second upload, create a mix context profile.

**Deliverables**:
- Mix-level DSP analysis (BPM, key, spectral occupancy, density, width maps)
- Source-role inference model
- Style cluster classification
- Needs engine with deficiency/opportunity detection
- Integration with frontend upload flow

### Phase 3 — Reference Priors
**Goal**: Create a style-aware reference model from successful commercial music.

**Deliverables**:
- Reference corpus ingestion pipeline
- Style cluster prior learning
- Per-cluster production norms (spectral, density, width, role, trajectory)
- Style-contextualized diagnostic comparison
- Integration with needs engine

### Phase 4 — Complement Recommendation
**Goal**: Recommend samples that improve the uploaded mix.

**Deliverables**:
- Candidate generator with hybrid DB queries (structured + vector)
- Rule-based scoring function implementation
- XGBoost/LightGBM reranker trained on initial data
- Explanation generator
- Integration with frontend results display

### Phase 5 — Human Preference Learning
**Goal**: Turn the system from "smart" into "special."

**Deliverables**:
- Feedback collection across all user interaction signals
- Preference pair dataset construction
- Ranker training pipeline
- Per-user taste model
- Preference model serving integration with reranker
- A/B testing framework to measure recommendation quality

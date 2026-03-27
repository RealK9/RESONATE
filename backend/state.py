"""
RESONATE — Shared application state.
Mutable globals that routes need to read/write across requests.
"""

# Latest track analysis data (set by /analyze, read by /samples)
latest_track_profile = {}
latest_ai_analysis = {}
latest_track_file = None  # path to uploaded track for dual playback
latest_mix_profile = None  # v2 MixProfile dict from ml.analysis.mix_analyzer
latest_gap_result = None  # v2 GapAnalysisResult from ml.analysis.gap_analyzer
latest_recommendations = None  # v2 RecommendationResult from ml.recommendation
latest_gap_result = None  # v2 GapAnalysisResult from ml.analysis.gap_analyzer

# DAW Bridge state (set by bridge.py, read by routes)
daw_bpm = 0.0       # Live DAW tempo (0 = not connected)
daw_playing = False  # DAW transport playing
last_scored_bpm = 0.0  # BPM used in last scoring pass (for change detection)

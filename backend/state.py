"""
RESONATE — Shared application state.
Mutable globals that routes need to read/write across requests.
"""

# Latest track analysis data (set by /analyze, read by /samples)
latest_track_profile = {}
latest_ai_analysis = {}
latest_track_file = None  # path to uploaded track for dual playback

# DAW Bridge state (set by bridge.py, read by routes)
daw_bpm = 0.0       # Live DAW tempo (0 = not connected)
daw_playing = False  # DAW transport playing
last_scored_bpm = 0.0  # BPM used in last scoring pass (for change detection)

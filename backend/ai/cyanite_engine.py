"""
RESONATE — Cyanite.ai Analysis Engine.
Real audio AI analysis via GraphQL API.
"""

import json
import time as _time
from pathlib import Path

from config import CYANITE_TOKEN, HAS_CYANITE


def _cyanite_graphql(query, variables=None):
    """Send a GraphQL request to Cyanite.ai API."""
    import urllib.request
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.cyanite.ai/graphql",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CYANITE_TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.request.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else "no body"
        print(f"  Cyanite API error {e.code}: {body[:500]}")
        raise


def _convert_to_mp3(wav_path):
    """Convert WAV to MP3 for Cyanite (which only accepts MP3)."""
    mp3_path = str(wav_path).rsplit(".", 1)[0] + "_cyanite.mp3"
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(wav_path))
        audio.export(mp3_path, format="mp3", bitrate="320k")
        return mp3_path
    except Exception:
        pass
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "320k", mp3_path],
            capture_output=True, timeout=60,
        )
        if Path(mp3_path).exists():
            return mp3_path
    except Exception:
        pass
    return None


def cyanite_analyze_track(filepath, latest_track_profile=None):
    """Upload track to Cyanite.ai for real AI audio analysis."""
    if not HAS_CYANITE:
        return None

    import urllib.request

    print("  Uploading to Cyanite.ai for real audio analysis...")

    # Clean up old library tracks first (free tier has a limit)
    print("  Checking Cyanite library space...")
    try:
        total_deleted = 0
        for attempt in range(5):
            old_tracks = _cyanite_graphql("""
                query LibraryCleanup {
                    libraryTracks {
                        edges { node { id } }
                        pageInfo { hasNextPage }
                    }
                }
            """)
            data = old_tracks.get("data")
            if not data or not data.get("libraryTracks"):
                print(f"  Library query result: {json.dumps(old_tracks)[:300]}")
                break
            edges = data["libraryTracks"].get("edges", [])
            if not edges:
                break
            old_ids = [e["node"]["id"] for e in edges[:100]]
            print(f"  Deleting {len(old_ids)} old Cyanite tracks (round {attempt + 1})...")
            del_result = _cyanite_graphql("""
                mutation Delete($input: LibraryTracksDeleteInput!) {
                    libraryTracksDelete(input: $input) {
                        __typename
                        ... on LibraryTracksDeleteSuccess { __typename }
                        ... on Error { message }
                    }
                }
            """, {"input": {"libraryTrackIds": old_ids}})
            print(f"  Delete result: {del_result.get('data', {})}")
            total_deleted += len(old_ids)
            _time.sleep(1)
        if total_deleted > 0:
            print(f"  ✓ Cleaned up {total_deleted} old tracks")
            _time.sleep(2)
    except Exception as e:
        print(f"  Cleanup warning: {e}")

    # Step 1: Convert to MP3 if needed
    fpath = str(filepath)
    if not fpath.lower().endswith(".mp3"):
        print("  Converting to MP3 for Cyanite...")
        mp3_path = _convert_to_mp3(fpath)
        if not mp3_path:
            print("  ✗ MP3 conversion failed")
            return None
        fpath = mp3_path
    else:
        mp3_path = None

    track_id = None
    try:
        # Step 2: Request file upload URL
        print("  Requesting upload URL...")
        result = _cyanite_graphql("""
            mutation { fileUploadRequest { id uploadUrl } }
        """)
        upload_info = result["data"]["fileUploadRequest"]
        file_id = upload_info["id"]
        upload_url = upload_info["uploadUrl"]

        # Step 3: Upload MP3 to S3
        print("  Uploading audio to Cyanite...")
        with open(fpath, "rb") as f:
            audio_data = f.read()
        req = urllib.request.Request(
            upload_url,
            data=audio_data,
            headers={"Content-Type": "audio/mpeg"},
            method="PUT",
        )
        urllib.request.urlopen(req, timeout=120)

        # Step 4: Create library track
        print("  Creating library track...")
        result = _cyanite_graphql("""
            mutation CreateTrack($input: LibraryTrackCreateInput!) {
                libraryTrackCreate(input: $input) {
                    __typename
                    ... on LibraryTrackCreateSuccess {
                        createdLibraryTrack { id }
                    }
                    ... on Error { message }
                }
            }
        """, {"input": {"uploadId": file_id, "title": Path(filepath).stem}})

        create_data = result["data"]["libraryTrackCreate"]
        if create_data["__typename"] != "LibraryTrackCreateSuccess":
            print(f"  ✗ Cyanite create error: {create_data.get('message', 'unknown')}")
            return None

        track_id = create_data["createdLibraryTrack"]["id"]
        print(f"  Track created: {track_id}")

        # Step 5: Enqueue analysis
        print("  Enqueuing analysis...")
        try:
            enqueue_result = _cyanite_graphql("""
                mutation Enqueue($input: LibraryTrackEnqueueInput!) {
                    libraryTrackEnqueue(input: $input) {
                        __typename
                        ... on LibraryTrackEnqueueSuccess { success }
                        ... on Error { message }
                    }
                }
            """, {"input": {"libraryTrackId": track_id}})
            enqueue_data = enqueue_result.get("data", {}).get("libraryTrackEnqueue", {})
            print(f"  Enqueue result: {enqueue_data.get('__typename', 'unknown')}")
            if enqueue_data.get("message"):
                print(f"  Enqueue message: {enqueue_data['message']}")
        except Exception as e:
            print(f"  Enqueue error (may already be queued): {e}")

        # Step 6: Poll for results
        print("  Waiting for Cyanite analysis", end="", flush=True)
        analysis_query = """
            query Track($id: ID!) {
                libraryTrack(id: $id) {
                    __typename
                    ... on LibraryTrack {
                        id
                        audioAnalysisV6 {
                            __typename
                            ... on AudioAnalysisV6Finished {
                                result {
                                    bpmPrediction { value }
                                    keyPrediction { value }
                                    energyLevel
                                    moodTags
                                    genreTags
                                    voice { female male instrumental }
                                }
                            }
                            ... on AudioAnalysisV6Enqueued { __typename }
                            ... on AudioAnalysisV6Processing { __typename }
                            ... on AudioAnalysisV6Failed { error { message } }
                        }
                        audioAnalysisV7 {
                            __typename
                            ... on AudioAnalysisV7Finished {
                                result {
                                    advancedGenreTags
                                    advancedSubgenreTags
                                    advancedInstrumentTags
                                    freeGenreTags
                                }
                            }
                            ... on AudioAnalysisV7Enqueued { __typename }
                            ... on AudioAnalysisV7Processing { __typename }
                            ... on AudioAnalysisV7Failed { error { message } }
                        }
                    }
                    ... on Error { message }
                }
            }
        """

        max_polls = 30
        for i in range(max_polls):
            _time.sleep(3)
            print(".", end="", flush=True)
            result = _cyanite_graphql(analysis_query, {"id": track_id})

            track_data = result.get("data", {}).get("libraryTrack", {})
            if track_data.get("__typename") != "LibraryTrack":
                continue

            v6 = track_data.get("audioAnalysisV6", {})
            v7 = track_data.get("audioAnalysisV7", {})

            v6_done = v6.get("__typename") == "AudioAnalysisV6Finished"
            v7_done = v7.get("__typename") == "AudioAnalysisV7Finished"

            if v6_done:
                print(" ✓")
                r6 = v6["result"]
                r7 = v7.get("result", {}) if v7_done else {}

                cyanite_bpm = r6.get("bpmPrediction", {}).get("value")
                cyanite_key = r6.get("keyPrediction", {}).get("value")
                energy_level = r6.get("energyLevel")
                mood_tags = r6.get("moodTags", [])
                v6_genre_tags = r6.get("genreTags", [])
                voice = r6.get("voice", {})

                genre_tags = r7.get("advancedGenreTags", v6_genre_tags)
                subgenre_tags = r7.get("advancedSubgenreTags", [])
                instrument_tags = r7.get("advancedInstrumentTags", [])
                free_genres = r7.get("freeGenreTags", [])

                genre_str = genre_tags[0] if genre_tags else "unknown"
                subgenre_str = subgenre_tags[0] if subgenre_tags else ""
                mood_str = ", ".join(mood_tags[:3]) if mood_tags else "unknown"

                print(f"  Cyanite results:")
                print(f"    Key: {cyanite_key}")
                print(f"    BPM: {cyanite_bpm}")
                print(f"    Genre: {genre_str} / {subgenre_str}")
                print(f"    Free genres: {free_genres}")
                print(f"    Mood: {mood_str}")
                print(f"    Energy: {energy_level}")
                print(f"    Instruments: {instrument_tags}")
                print(f"    Voice: {voice}")

                what_has = instrument_tags[:] if instrument_tags else []

                all_instruments = ["percussion", "synth", "piano", "acousticGuitar",
                                   "electricGuitar", "strings", "bass", "bassGuitar",
                                   "brass", "woodwinds"]
                what_needs = [i for i in all_instruments if i not in instrument_tags]

                cyanite_to_profile = {
                    "rapHipHop": "trap/hip-hop", "trap": "trap/hip-hop",
                    "pop": "pop", "popRap": "trap/hip-hop",
                    "rnb": "r&b", "contemporaryRnB": "r&b", "neoSoul": "r&b",
                    "electronicDance": "edm/electronic", "house": "edm/electronic",
                    "techno": "edm/electronic", "techHouse": "edm/electronic",
                    "ambient": "lo-fi/chill",
                }
                mapped_genre = genre_str
                for tag in genre_tags + subgenre_tags:
                    if tag in cyanite_to_profile:
                        mapped_genre = cyanite_to_profile[tag]
                        break

                # Cross-check with measured audio features
                if latest_track_profile:
                    heuristic_genre = latest_track_profile.get("detected_genre", "")
                    if heuristic_genre and heuristic_genre != "default":
                        track_sb = latest_track_profile.get("frequency_bands", {}).get("sub_bass_20_80", 0)
                        if track_sb > 0.12 and mapped_genre in ("pop", "lo-fi/chill", "default", "unknown"):
                            print(f"  Genre override: Cyanite said '{mapped_genre}' but measured sub-bass={track_sb:.2f} → {heuristic_genre}")
                            mapped_genre = heuristic_genre

                return {
                    "source": "cyanite",
                    "genre": mapped_genre,
                    "genre_raw": genre_tags,
                    "subgenre": subgenre_tags,
                    "free_genres": free_genres,
                    "mood": mood_str,
                    "mood_tags": mood_tags,
                    "energy": energy_level or "medium",
                    "what_track_has": what_has,
                    "what_track_needs": what_needs,
                    "instruments": instrument_tags,
                    "voice": voice,
                    "cyanite_key": cyanite_key,
                    "cyanite_bpm": cyanite_bpm,
                    "summary": f"{mapped_genre} track — {mood_str}. Instruments: {', '.join(what_has) or 'minimal'}. Needs: {', '.join(what_needs[:4]) or 'refinement'}.",
                }

            elif v6.get("__typename") == "AudioAnalysisV6Failed":
                err = v6.get("error", {}).get("message", "unknown")
                print(f" ✗ Failed: {err}")
                return None

        print(" ✗ Timeout waiting for Cyanite analysis")
        return None

    except Exception as e:
        print(f"\n  Cyanite error: {e}")
        return None
    finally:
        if mp3_path and Path(mp3_path).exists():
            try:
                Path(mp3_path).unlink()
            except Exception:
                pass
        try:
            if track_id:
                _cyanite_graphql("""
                    mutation Delete($input: LibraryTracksDeleteInput!) {
                        libraryTracksDelete(input: $input) {
                            __typename
                            ... on LibraryTracksDeleteSuccess { __typename }
                            ... on Error { message }
                        }
                    }
                """, {"input": {"libraryTrackIds": [track_id]}})
                print(f"  Cleaned up Cyanite track {track_id}")
        except Exception:
            pass

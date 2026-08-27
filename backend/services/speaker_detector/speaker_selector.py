"""
Speaker Selector — WHO is speaking in this clip.

Orchestrates the service's answer end-to-end (moved here from the pipeline
orchestrator, where selection logic didn't belong):

    1. Candidate gate: EVERY track overlapping the speech window by at least
       min_track_speech_overlap enters the competition. NO top-N cap by
       default (max_candidate_tracks=None) — in wide shots of 5-speaker
       panels, a size-based cap could eliminate the real speaker (small face
       in the back) before TalkNet ever scores them. A numeric cap remains
       available purely as a speed knob.
    2. ONE decode of the clip → face crops for ALL candidates (TrackCropReader).
    3. MFCC slice from the per-clip cache (MfccExtractor).
    4. TalkNet scores every candidate.
    5. Winner = max(overlap × mean ASD score × average face size) — the
       exact selection formula the pipeline used before, unchanged.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from services.speaker_detector.active_speaker import ASDResult, TalkNetASD
from services.speaker_detector.audio_features import MfccExtractor
from services.speaker_detector.track_crops import CropRequest, TrackCropReader

# Crop specification TalkNet was trained on
ASD_CROP_SIZE = (112, 112)
ASD_CROP_MARGIN = 0.2


@dataclass
class SpeakerSelection:
    """The service's answer: the active speaker's track + its ASD evidence."""
    track: object            # FaceTrack of the active speaker
    asd_result: ASDResult


class SpeakerSelector:
    """Selects the active speaker among a clip's face tracks."""

    def __init__(
        self,
        talknet: TalkNetASD,
        crop_reader: TrackCropReader,
        mfcc_extractor: MfccExtractor,
        min_track_speech_overlap: float,
        max_candidate_tracks: Optional[int] = None,
    ):
        self.talknet = talknet
        self.crop_reader = crop_reader
        self.mfcc_extractor = mfcc_extractor
        self.min_track_speech_overlap = min_track_speech_overlap
        # None = no cap (default) — every overlapping track is scored
        self.max_candidate_tracks = max_candidate_tracks

    def select_active_speaker(
        self,
        video_path: Path,
        audio_path: Path,
        face_tracks: List,
        speech_start: float,
        speech_end: float,
        fps: float,
    ) -> Optional[SpeakerSelection]:
        """Pick the track whose lips match the clip's audio.

        Times are clip-file seconds (same coordinates as the wav).
        Returns None when no track meaningfully overlaps the speech —
        the caller drops the clip as 'no_track_overlaps_speech'.
        """
        candidates = self._gate_candidates(face_tracks, speech_start, speech_end, fps)
        if not candidates:
            return None

        logger.info(
            f"    ASD scoring {len(candidates)}/{len(face_tracks)} candidate tracks..."
        )

        # One decode for every candidate's overlap window
        requests = [
            CropRequest(
                track=track,
                start_frame=max(track.start_frame, int(speech_start * fps)),
                end_frame=min(track.end_frame + 1, int(speech_end * fps)),
            )
            for track in candidates
        ]
        crops_by_track = self.crop_reader.read(
            video_path, requests,
            target_size=ASD_CROP_SIZE, grayscale=True, bbox_margin=ASD_CROP_MARGIN,
        )

        # One MFCC computation per clip (cached), sliced to the speech window
        audio_features = self.mfcc_extractor.features_for_window(
            audio_path, speech_start, speech_end,
        )

        results: List[tuple] = []   # (track, ASDResult)
        for request in requests:
            crops = crops_by_track.get(request.track.track_id)
            if crops is None or len(crops) == 0:
                continue
            scores = self.talknet.score_crops(crops, audio_features)
            results.append((request.track, ASDResult(
                track_id=request.track.track_id,
                start_frame=request.start_frame,
                end_frame=request.end_frame,
                scores=scores,
                method=self.talknet.method,
            )))

        return self._pick_winner(results, speech_start, speech_end, fps)

    # ----------------------------------------------------------- internals

    def _gate_candidates(
        self,
        face_tracks: List,
        speech_start: float,
        speech_end: float,
        fps: float,
    ) -> List:
        """Tracks overlapping the speech window ≥ min_track_speech_overlap.

        With max_candidate_tracks set (speed knob), keeps the top-N by
        overlap × average face size; by default everyone competes.
        """
        gated = []
        for track in face_tracks:
            overlap = self._speech_overlap(track, speech_start, speech_end, fps)
            if overlap < self.min_track_speech_overlap:
                continue
            gated.append((overlap * self._average_face_size(track), track))

        gated.sort(key=lambda item: item[0], reverse=True)
        if self.max_candidate_tracks is not None:
            gated = gated[: self.max_candidate_tracks]
        return [track for _, track in gated]

    def _pick_winner(
        self,
        results: List[tuple],
        speech_start: float,
        speech_end: float,
        fps: float,
    ) -> Optional[SpeakerSelection]:
        """The exact pre-revision formula: overlap × mean ASD × face size."""
        best_selection = None
        best_score = 0.0
        for track, asd_result in results:
            overlap = self._speech_overlap(track, speech_start, speech_end, fps)
            score = overlap * asd_result.mean_score * self._average_face_size(track)
            if score > best_score:
                best_score = score
                best_selection = SpeakerSelection(track=track, asd_result=asd_result)
        return best_selection

    @staticmethod
    def _speech_overlap(track, speech_start: float, speech_end: float, fps: float) -> float:
        track_start_sec = track.start_frame / fps
        track_end_sec = (track.end_frame + 1) / fps
        return min(speech_end, track_end_sec) - max(speech_start, track_start_sec)

    @staticmethod
    def _average_face_size(track) -> float:
        if not track.detections:
            return 1.0
        return float(np.mean(
            [max(d.bbox.width, d.bbox.height) for d in track.detections]
        ))

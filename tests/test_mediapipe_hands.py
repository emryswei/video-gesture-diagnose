from types import SimpleNamespace

from video_analysis.mediapipe_hands import (
    HandFrameEvidence,
    finger_extension_score,
    is_bent_and_compact,
    is_extended_and_wide,
)


def make_landmarks(*, bent: bool):
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    for offset, (mcp, pip, dip, tip) in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        x = float(offset)
        landmarks[mcp] = SimpleNamespace(x=x, y=0.0, z=0.0)
        landmarks[pip] = SimpleNamespace(x=x, y=1.0, z=0.0)
        landmarks[dip] = SimpleNamespace(
            x=x + (1.0 if bent else 0.0), y=1.0 if bent else 2.0, z=0.0
        )
        landmarks[tip] = SimpleNamespace(
            x=x + (1.0 if bent else 0.0), y=0.0 if bent else 3.0, z=0.0
        )
    return landmarks


def test_finger_extension_score_separates_straight_and_bent_fingers():
    assert finger_extension_score(make_landmarks(bent=False)) > 0.95
    assert finger_extension_score(make_landmarks(bent=True)) < 0.25


def test_landmark_geometry_thresholds_are_explicit():
    extended = HandFrameEvidence(1, 1, 0.84, 0.62, 0.9)
    bent = HandFrameEvidence(2, 1, 0.60, 0.25, 0.9)

    assert is_extended_and_wide(extended)
    assert not is_bent_and_compact(extended)
    assert is_bent_and_compact(bent)
    assert not is_extended_and_wide(bent)

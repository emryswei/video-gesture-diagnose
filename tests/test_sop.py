import pytest

from video_analysis.sop import load_sop
from video_analysis.vlm import build_segment_prompt


def test_who_handrub_has_seven_auditable_steps():
    sop = load_sop()

    assert sop.id == "who_handrub"
    assert sop.standard_version == "WHO-2009"
    assert sop.definition_version == 2
    assert len(sop.steps) == 7
    assert [step.order for step in sop.steps] == list(range(1, 8))
    assert "hands_dry" not in {step.id for step in sop.steps}


def test_bilateral_requirements_match_v1_definition():
    sop = load_sop()
    bilateral = {step.id for step in sop.steps if step.requires_both_sides}

    assert bilateral == {
        "palm_over_dorsum",
        "backs_of_fingers",
        "thumbs",
        "fingertips_in_palm",
    }


def test_segment_prompt_contains_discriminating_pose_signatures():
    prompt = build_segment_prompt(load_sop())

    assert '"fingers_interlaced": true' in prompt
    assert '"fingers_bent": true' in prompt
    assert "Do not classify straight interlaced fingers as backs of fingers" in prompt


def test_sop_id_rejects_path_traversal():
    with pytest.raises(ValueError, match="Unknown SOP"):
        load_sop("../who_handrub")

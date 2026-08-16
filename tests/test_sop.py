import pytest

from video_analysis.sop import load_sop
from video_analysis.vlm import build_segment_prompt


def test_who_handrub_has_seven_auditable_steps():
    sop = load_sop("who_handrub")

    assert sop.id == "who_handrub"
    assert sop.standard_version == "WHO-2009"
    assert sop.definition_version == 2
    assert len(sop.steps) == 7
    assert [step.order for step in sop.steps] == list(range(1, 8))
    assert "hands_dry" not in {step.id for step in sop.steps}


def test_hk_chp_handrub_is_default_and_includes_wrists():
    sop = load_sop()

    assert sop.id == "hk_chp_handrub"
    assert sop.standard_version == "HK-CHP"
    assert sop.duration_max_seconds is None
    assert len(sop.steps) == 8
    assert [step.order for step in sop.steps] == list(range(1, 9))
    wrists = sop.steps[-1]
    assert wrists.id == "wrists"
    assert wrists.requires_both_sides is True


def test_bilateral_requirements_match_v1_definition():
    sop = load_sop("who_handrub")
    bilateral = {step.id for step in sop.steps if step.requires_both_sides}

    assert bilateral == {
        "palm_over_dorsum",
        "backs_of_fingers",
        "thumbs",
        "fingertips_in_palm",
    }


def test_segment_prompt_contains_discriminating_pose_signatures():
    prompt = build_segment_prompt(load_sop("who_handrub"))

    assert '"fingers_interlaced": true' in prompt
    assert '"fingers_bent": true' in prompt
    assert "Do not classify straight interlaced fingers as backs of fingers" in prompt


def test_sop_id_rejects_path_traversal():
    with pytest.raises(ValueError, match="Unknown SOP"):
        load_sop("../who_handrub")

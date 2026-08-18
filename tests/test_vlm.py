from video_analysis.models import StepStatus
from video_analysis.sop import load_sop
from video_analysis.video import SampledFrame
from video_analysis.vlm import AWSBedrockVLM, OpenAICompatibleVLM, create_vlm


def test_segment_timestamps_are_limited_to_the_supplied_window(monkeypatch):
    vlm = OpenAICompatibleVLM()
    monkeypatch.setattr(
        vlm,
        "_request",
        lambda *_args, **_kwargs: {
            "step_id": load_sop().steps[0].id,
            "status": "passed",
            "confidence": 0.9,
            "start_sec": 1,
            "end_sec": 99,
            "observation": "The action is visible.",
        },
    )
    frames = [
        SampledFrame(timestamp_sec=10, jpeg_bytes=b"jpeg"),
        SampledFrame(timestamp_sec=12, jpeg_bytes=b"jpeg"),
    ]

    result = vlm.analyze_segment(frames, load_sop())

    assert result.status == StepStatus.PASSED
    assert result.start_sec == 10
    assert result.end_sec == 12


def test_passed_segment_without_timestamps_becomes_uncertain(monkeypatch):
    vlm = OpenAICompatibleVLM()
    monkeypatch.setattr(
        vlm,
        "_request",
        lambda *_args, **_kwargs: {
            "step_id": load_sop().steps[0].id,
            "status": "passed",
            "confidence": 0.9,
            "start_sec": None,
            "end_sec": None,
            "observation": "The action may be visible.",
        },
    )

    result = vlm.analyze_segment(
        [SampledFrame(timestamp_sec=10, jpeg_bytes=b"jpeg")],
        load_sop(),
    )

    assert result.status == StepStatus.UNCERTAIN
    assert result.start_sec is None
    assert result.end_sec is None


def test_aws_bedrock_sends_timestamped_jpeg_frames(monkeypatch):
    class FakeClient:
        request = None

        def converse(self, **kwargs):
            self.request = kwargs
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": """{
                                  "step_id": "apply_product",
                                  "status": "passed",
                                  "confidence": 0.9,
                                  "start_sec": 1,
                                  "end_sec": 99,
                                  "observation": "Product is visibly applied."
                                }"""
                            }
                        ]
                    }
                }
            }

    fake_client = FakeClient()

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        def client(self, service_name, **_kwargs):
            assert service_name == "bedrock-runtime"
            return fake_client

    monkeypatch.setattr("video_analysis.vlm.boto3.Session", FakeSession)
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    vlm = AWSBedrockVLM("qwen.qwen3-vl-235b-a22b")
    vlm.set_frame_evidence({10: "MediaPipe evidence"})
    frames = [
        SampledFrame(timestamp_sec=10, jpeg_bytes=b"first-jpeg"),
        SampledFrame(timestamp_sec=12, jpeg_bytes=b"second-jpeg"),
    ]

    result = vlm.analyze_segment(frames, load_sop())

    assert result.status == StepStatus.PASSED
    assert result.start_sec == 10
    assert result.end_sec == 12
    assert fake_client.request["modelId"] == "qwen.qwen3-vl-235b-a22b"
    assert fake_client.request["inferenceConfig"]["temperature"] == 0
    content = fake_client.request["messages"][0]["content"]
    assert content[2]["image"]["source"]["bytes"] == b"first-jpeg"
    assert content[4]["image"]["source"]["bytes"] == b"second-jpeg"
    assert "MediaPipe evidence" in content[1]["text"]


def test_vlm_factory_keeps_openai_compatible_fallback(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compatible")

    assert isinstance(create_vlm("local-model"), OpenAICompatibleVLM)

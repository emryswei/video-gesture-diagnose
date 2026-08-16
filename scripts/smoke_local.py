from __future__ import annotations

import tempfile
import time
from pathlib import Path

import av
import httpx
from PIL import Image


APP_URL = "http://127.0.0.1:8000"


def create_test_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"
        for index in range(32):
            image = Image.new("RGB", (320, 240), (30 + index * 4, 60, 120))
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as output:
        video_path = Path(output.name)
    try:
        create_test_video(video_path)
        with httpx.Client(timeout=30) as client:
            with video_path.open("rb") as video:
                response = client.post(
                    f"{APP_URL}/api/analyses",
                    files={"video": ("smoke.avi", video, "video/x-msvideo")},
                    data={"sop_id": "hk_chp_handrub"},
                )
            response.raise_for_status()
            job = response.json()
            analysis_id = job["analysis_id"]
            print(f"created={analysis_id}")

            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                result_response = client.get(f"{APP_URL}/api/analyses/{analysis_id}")
                result_response.raise_for_status()
                result = result_response.json()
                print(f"status={result['status']}")
                if result["status"] in {"succeeded", "failed"}:
                    if result["status"] != "succeeded":
                        raise RuntimeError(result.get("error", "analysis failed"))
                    step_count = len(result["result"]["steps"])
                    if step_count != 8:
                        raise RuntimeError(f"Expected 8 HK CHP steps, got {step_count}")
                    print(f"steps={step_count}")
                    return
                time.sleep(2)
        raise TimeoutError("analysis did not finish within 20 minutes")
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

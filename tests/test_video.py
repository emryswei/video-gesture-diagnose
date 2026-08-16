from fractions import Fraction

import av
from PIL import Image

from video_analysis.video import sample_video


def create_test_video(path, duration_seconds=16, fps=4):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 160
        stream.height = 120
        stream.pix_fmt = "yuv420p"
        for index in range(duration_seconds * fps):
            frame = av.VideoFrame.from_image(Image.new("RGB", (160, 120), "blue"))
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_sample_video_uses_two_fps_and_preserves_timestamps(tmp_path):
    path = tmp_path / "sample.mp4"
    create_test_video(path)

    sampled = sample_video(path, sample_fps=2, max_frames=96, max_image_edge=160)

    assert sampled.duration_sec == 16
    assert len(sampled.frames) == 32
    assert sampled.frames[0].timestamp_sec > 0
    assert sampled.frames[-1].timestamp_sec < sampled.duration_sec

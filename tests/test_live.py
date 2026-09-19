"""Sanity check for bronchotrack/live.py's background-threaded reader.

Doesn't require an actual RTSP source: writes a tiny synthetic video file
and points LiveVideoStream at it (cv2.VideoCapture treats a file path and a
stream URL identically), which is enough to exercise the connect / read /
stop lifecycle and confirm frames actually arrive.

Run: python3 tests/test_live.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _make_synthetic_video(path: str, n_frames: int = 30, size=(64, 64), fps: float = 20.0) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_live_stream_reads_frames():
    from bronchotrack.live import LiveVideoStream

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "synthetic.mp4")
        _make_synthetic_video(video_path, n_frames=30)

        stream = LiveVideoStream(video_path, reconnect=False, connect_timeout_sec=10.0)
        stream.start()
        try:
            frame, idx = stream.read_new(-1, timeout_sec=5.0)
            assert frame is not None, "expected at least one frame from the synthetic video"
            assert idx >= 0

            # give the background thread a moment to advance further
            time.sleep(0.3)
            frame2, idx2 = stream.read_new(idx, timeout_sec=2.0)
            # either more frames arrived, or the short synthetic video already ended --
            # both are fine, we're only checking the mechanism doesn't crash/hang
            assert idx2 >= idx
        finally:
            stream.stop()

    print("[live] synthetic video read OK")


def _run_all():
    tests = [test_live_stream_reads_frames]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()

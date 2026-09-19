"""Live video support: read frames from an RTSP/RTMP/HTTP stream in a
background thread so the main processing loop always works on the *latest*
available frame instead of falling behind a buffered queue (the standard
failure mode with `cv2.VideoCapture` on network streams: if you call
`.read()` once per processed frame and processing is slower than the
stream's frame rate, frames pile up in the socket buffer and your "live"
view drifts further and further behind reality).

Also handles the other standard RTSP annoyance: streams drop and need to be
reconnected without killing the whole pipeline.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class LiveVideoStream:
    """Background-threaded reader that always exposes the most recent frame.

    Usage
    -----
    >>> stream = LiveVideoStream("rtsp://192.168.1.50:8554/bronch")
    >>> stream.start()
    >>> frame = stream.read()   # non-blocking; None until the first frame arrives
    >>> stream.stop()
    """

    def __init__(
        self,
        source: str,
        reconnect: bool = True,
        max_reconnect_attempts: int = 10,
        reconnect_delay_sec: float = 2.0,
        connect_timeout_sec: float = 15.0,
        fast_retry_limit: int = 15,
        fast_retry_delay_sec: float = 0.02,
    ):
        """
        `fast_retry_limit` / `fast_retry_delay_sec` control tolerance for
        isolated dropped/corrupt frames (common on WiFi MJPEG sources like
        DroidCam) *without* tearing down and reopening the whole capture:
        up to `fast_retry_limit` consecutive failed reads are just retried
        after a short sleep. Only once that many reads fail in a row does
        this fall back to the slower release-and-reconnect path -- that
        distinction matters because a full reopen is comparatively
        expensive and, for a stream that's basically fine but drops one
        frame in a hundred, doing it on every glitch would tank your
        effective frame rate for no reason.
        """
        self.source = source
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self.connect_timeout_sec = connect_timeout_sec
        self.fast_retry_limit = fast_retry_limit
        self.fast_retry_delay_sec = fast_retry_delay_sec

        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_idx = -1
        self._running = False
        self._connected_event = threading.Event()
        self._fatal_error: Optional[str] = None

    # ------------------------------------------------------------------
    def start(self) -> "LiveVideoStream":
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._connected_event.wait(timeout=self.connect_timeout_sec):
            self.stop()
            raise ConnectionError(
                f"Timed out connecting to live source '{self.source}' "
                f"after {self.connect_timeout_sec}s"
            )
        if self._fatal_error:
            raise ConnectionError(self._fatal_error)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame (or None if nothing has arrived
        yet). Non-blocking. Repeated calls between new frames arriving
        return the same frame -- callers wanting "wait for a new frame"
        should use `read_new(last_idx)` instead."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def read_new(self, last_idx: int, poll_interval_sec: float = 0.005, timeout_sec: float = 5.0):
        """Block (briefly) until a frame newer than `last_idx` is available.
        Returns (frame, frame_idx) or (None, last_idx) on timeout."""
        start = time.time()
        while time.time() - start < timeout_sec:
            with self._lock:
                if self._latest_frame_idx > last_idx and self._latest_frame is not None:
                    return self._latest_frame.copy(), self._latest_frame_idx
            time.sleep(poll_interval_sec)
        return None, last_idx

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._connected_event.is_set()

    # ------------------------------------------------------------------
    def _open(self) -> bool:
        import cv2

        cap = cv2.VideoCapture(self.source)
        # keep OpenCV's internal buffer as small as possible so we don't
        # silently accumulate latency even between our own read() calls
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    def _run(self) -> None:
        reopen_attempts = 0
        consecutive_failures = 0
        if not self._open():
            self._fatal_error = f"Could not open live source: {self.source}"
            self._connected_event.set()  # unblock start(); it will raise
            return

        self._connected_event.set()
        frame_idx = 0

        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1

                if consecutive_failures <= self.fast_retry_limit:
                    # likely just an isolated dropped/corrupt frame -- retry
                    # without tearing down the connection
                    time.sleep(self.fast_retry_delay_sec)
                    continue

                # enough consecutive failures in a row that this looks like
                # a real disconnect, not a glitch -- fall back to reopening
                reopen_attempts += 1
                if not self.reconnect or reopen_attempts > self.max_reconnect_attempts:
                    self._fatal_error = (
                        f"Live source '{self.source}' disconnected and "
                        f"reconnect gave up after {reopen_attempts} attempts"
                    )
                    break
                time.sleep(self.reconnect_delay_sec)
                if self._cap is not None:
                    self._cap.release()
                if self._open():
                    reopen_attempts = 0
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            reopen_attempts = 0
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_idx = frame_idx
            frame_idx += 1

        self._running = False

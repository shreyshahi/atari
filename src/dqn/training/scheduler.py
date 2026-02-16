from __future__ import annotations


class EpsilonScheduler:
    def __init__(self, start: float, end: float, decay_frames: int):
        self.start = float(start)
        self.end = float(end)
        self.decay_frames = max(1, int(decay_frames))

    def value(self, frame: int) -> float:
        if frame >= self.decay_frames:
            return self.end
        frac = frame / self.decay_frames
        return self.start + frac * (self.end - self.start)

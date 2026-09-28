import numpy as np


class Car:
    def __init__(s, x=0.0, y=0.0, h=0.0, v=0.0, L=2.8):
        s.x, s.y, s.h, s.v, s.L = x, y, h, v, L
        s.d = 0.0

    def step(s, d, a, dt):
        dm = 0.6
        rt = 0.5 * dt
        s.d = float(np.clip(np.clip(d, s.d - rt, s.d + rt), -dm, dm))
        a = float(np.clip(a, -6.0, 3.0))
        s.x += s.v * np.cos(s.h) * dt
        s.y += s.v * np.sin(s.h) * dt
        s.h += s.v / s.L * np.tan(s.d) * dt
        s.v = max(0.0, s.v + a * dt)

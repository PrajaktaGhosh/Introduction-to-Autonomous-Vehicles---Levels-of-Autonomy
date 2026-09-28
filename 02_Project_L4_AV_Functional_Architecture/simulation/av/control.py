import numpy as np


class Lateral:
    def __init__(s, rt, L=2.8, k0=4.0, k1=0.6):
        s.rt, s.L, s.k0, s.k1 = rt, L, k0, k1

    def cmd(s, car, es):
        ld = s.k0 + s.k1 * car.v
        tx, ty = s.rt.xy(es + ld)
        al = np.arctan2(ty - car.y, tx - car.x) - car.h
        return float(np.arctan2(2 * s.L * np.sin(al), ld))


class Longitudinal:
    def __init__(s, kp=0.8, ki=0.1, b=2.0, jmax=2.5):
        s.kp, s.ki, s.b, s.j = kp, ki, b, jmax
        s.i, s.a = 0.0, 0.0

    def cmd(s, es, v, p, dt):
        if p["mode"] in ("MRM", "MRC_REACHED"):
            a = -3.0 if v > 0.05 else -0.5
            s.a = max(a, s.a - s.j * 2 * dt)
            return s.a, 0.0
        vr = p["vref"]
        if p["stop"] is not None:
            r = p["stop"] - es
            vr = min(vr, np.sqrt(max(2 * s.b * max(r - 0.3, 0.0), 0.0)))
        e = vr - v
        s.i = float(np.clip(s.i + e * dt, -5, 5))
        a = s.kp * e + s.ki * s.i
        if p["acc"] is not None:
            a = min(a, p["acc"])
        if p["stop"] is not None:
            r = p["stop"] - es - 0.3
            ff = -v * v / (2 * max(r, 0.05))
            if ff < -0.3 or r < 5:
                a = min(a, ff)
            if vr < 1.0:
                s.i = min(s.i, 0.0)
        if p["stop"] is not None and p["stop"] - es < 0.4 and v < 0.3:
            a, s.i = -2.0, 0.0
        a = float(np.clip(a, -6.0, 2.0))
        s.a = float(np.clip(a, s.a - s.j * dt, s.a + s.j * dt))
        return s.a, vr

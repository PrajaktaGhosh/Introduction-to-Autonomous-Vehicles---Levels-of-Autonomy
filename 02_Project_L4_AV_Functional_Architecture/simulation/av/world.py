import numpy as np


class Route:
    def __init__(s, pts):
        s.p = np.asarray(pts, float)
        d = np.linalg.norm(np.diff(s.p, axis=0), axis=1)
        s.s = np.concatenate([[0], np.cumsum(d)])
        s.len = s.s[-1]
        h = np.arctan2(np.diff(s.p[:, 1]), np.diff(s.p[:, 0]))
        s.h = np.concatenate([h, [h[-1]]])
        dh = np.diff(np.unwrap(s.h))
        s.k = np.concatenate([dh / np.maximum(d, 1e-6), [0]])

    def xy(s, q):
        q = np.clip(q, 0, s.len)
        return np.interp(q, s.s, s.p[:, 0]), np.interp(q, s.s, s.p[:, 1])

    def hd(s, q):
        return np.interp(np.clip(q, 0, s.len), s.s, np.unwrap(s.h))

    def frenet(s, x, y):
        i = int(np.argmin(np.hypot(s.p[:, 0] - x, s.p[:, 1] - y)))
        h = s.h[i]
        dx, dy = x - s.p[i, 0], y - s.p[i, 1]
        return s.s[i] + dx * np.cos(h) + dy * np.sin(h), -dx * np.sin(h) + dy * np.cos(h)

    def kmax(s, q0, q1):
        m = (s.s >= q0) & (s.s <= q1)
        return float(np.max(np.abs(s.k[m]))) if m.any() else 0.0


def make_route(r=30.0, a=150.0, b=250.0, ds=0.5):
    p = [(x, 0.0) for x in np.arange(0, a, ds)]
    n = int(np.pi / 2 * r / ds)
    p += [(a + r * np.sin(t), r - r * np.cos(t)) for t in np.linspace(0, np.pi / 2, n)]
    p += [(a + r, r + y) for y in np.arange(ds, b, ds)]
    return Route(p)


class Scenario:
    def __init__(s, rt):
        s.rt = rt
        s.lead = {"s": 45.0, "v": 6.0, "out": 140.0}
        s.light = {"s": 260.0, "red_until": 44.0}
        s.ped = {"s": 330.0, "d": -6.0, "vd": 1.2, "trig": 268.0, "on": False}
        s.dest = 400.0

    def step(s, t, dt, es):
        if s.lead is not None:
            s.lead["s"] += s.lead["v"] * dt
            if s.lead["s"] > s.lead["out"]:
                s.lead = None
        if es > s.ped["trig"]:
            s.ped["on"] = True
        if s.ped["on"] and s.ped["d"] < 6.0:
            s.ped["d"] += s.ped["vd"] * dt

    def truth(s, t):
        o = []
        if s.lead is not None:
            o.append({"id": 1, "cls": "car", "s": s.lead["s"], "d": 0.0, "vs": s.lead["v"], "vd": 0.0})
        o.append({"id": 2, "cls": "ped", "s": s.ped["s"], "d": s.ped["d"], "vs": 0.0, "vd": s.ped["vd"] if s.ped["on"] and s.ped["d"] < 6 else 0.0})
        g = "red" if t < s.light["red_until"] else "green"
        return o, {"s": s.light["s"], "state": g}

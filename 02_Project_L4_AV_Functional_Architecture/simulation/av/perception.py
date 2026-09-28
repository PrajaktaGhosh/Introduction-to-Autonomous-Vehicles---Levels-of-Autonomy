import numpy as np


class Tracker:
    def __init__(s, al=0.5, be=0.2):
        s.al, s.be, s.tr = al, be, {}

    def up(s, dets, dt):
        seen = set()
        for z in dets:
            k = z["id"]
            seen.add(k)
            if k not in s.tr:
                s.tr[k] = {**z, "vs": 0.0, "vd": 0.0}
                continue
            t = s.tr[k]
            for p, v in (("s", "vs"), ("d", "vd")):
                pr = t[p] + t[v] * dt
                r = z[p] - pr
                t[p] = pr + s.al * r
                t[v] = t[v] + s.be * r / dt
        for k in list(s.tr):
            if k not in seen:
                del s.tr[k]
        return [dict(v) for v in s.tr.values()]


class Perception:
    def __init__(s, rng=80.0, sd=0.25, seed=7):
        s.rng, s.sd = rng, sd
        s.r = np.random.default_rng(seed)
        s.trk = Tracker()
        s.ok = True

    def run(s, es, objs, light, dt):
        if not s.ok:
            return None
        dets = []
        for o in objs:
            if 0 < o["s"] - es < s.rng:
                dets.append({"id": o["id"], "cls": o["cls"], "s": o["s"] + s.r.normal(0, s.sd), "d": o["d"] + s.r.normal(0, s.sd * 0.6)})
        tl = light if 0 < light["s"] - es < 70 else None
        return {"tracks": s.trk.up(dets, dt), "light": tl}

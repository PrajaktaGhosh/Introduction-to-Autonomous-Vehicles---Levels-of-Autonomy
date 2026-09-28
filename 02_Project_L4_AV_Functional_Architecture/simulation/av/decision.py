import numpy as np

LF, VF, SS, YV, PO, MRM, DONE = "LANE_FOLLOW", "VEHICLE_FOLLOW", "STOP_SIGNAL", "YIELD_VRU", "PULL_OVER", "MRM", "MRC_REACHED"


class Behaviour:
    def __init__(s, rt, dest, vmax=12.0, alat=2.0):
        s.rt, s.dest, s.vmax, s.alat = rt, dest, vmax, alat
        s.m = LF

    def conflict(s, t, es, v):
        tta = max((t["s"] - es) / max(v, 1.0), 0.0)
        ts = np.linspace(0, min(tta + 2.0, 6.0), 25)
        return np.min(np.abs(t["d"] + t["vd"] * ts)) < 3.0 or abs(t["d"]) < 3.0

    def plan(s, es, v, wm, sup):
        if s.m in (MRM, DONE):
            if s.m == MRM and v < 0.05:
                s.m = DONE
            return {"mode": s.m, "vref": 0.0, "stop": None, "acc": None}
        if sup or wm is None:
            s.m = MRM
            return {"mode": s.m, "vref": 0.0, "stop": None, "acc": None}
        k = s.rt.kmax(es, es + 20 + 2 * v)
        vr = min(s.vmax, np.sqrt(s.alat / k) if k > 1e-4 else s.vmax)
        st, m, acc = None, LF, None
        lead = [t for t in wm["tracks"] if t["cls"] == "car" and abs(t["d"]) < 1.8 and t["s"] > es]
        if lead:
            l = min(lead, key=lambda t: t["s"])
            g = l["s"] - es - 4.5
            gd = 5.0 + 1.6 * v
            if g < gd + 25:
                m = VF
                acc = 0.25 * (g - gd) + 0.6 * (l["vs"] - v)
        lt = wm["light"]
        if lt is not None and lt["state"] == "red":
            st, m = lt["s"] - 2.0, SS
        for t in wm["tracks"]:
            if t["cls"] == "ped" and 0 < t["s"] - es < 60 and s.conflict(t, es, v):
                q = t["s"] - 6.0
                if st is None or q < st:
                    st, m = q, YV
        if s.dest - es < 45:
            q = s.dest
            if st is None or q < st:
                st, m = q, PO
            if s.dest - es < 0.6 and v < 0.05:
                m = DONE
        s.m = m
        return {"mode": m, "vref": vr, "stop": st, "acc": acc}

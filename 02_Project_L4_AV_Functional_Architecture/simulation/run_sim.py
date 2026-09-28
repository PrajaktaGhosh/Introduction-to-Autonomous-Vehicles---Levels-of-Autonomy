import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from av.world import make_route, Scenario
from av.vehicle import Car
from av.perception import Perception
from av.decision import Behaviour
from av.control import Lateral, Longitudinal

CL = {"LANE_FOLLOW": "#dcfce7", "VEHICLE_FOLLOW": "#dbeafe", "STOP_SIGNAL": "#fee2e2", "YIELD_VRU": "#fef3c7",
      "PULL_OVER": "#ede9fe", "MRM": "#fecaca", "MRC_REACHED": "#e5e7eb"}


def run(fault=None, dt=0.02, T=90.0):
    rt = make_route()
    sc = Scenario(rt)
    car = Car(v=0.0)
    pc = Perception()
    bh = Behaviour(rt, sc.dest)
    la, lo = Lateral(rt), Longitudinal()
    lg = {k: [] for k in ["t", "x", "y", "v", "vr", "a", "m", "ey", "gap", "dp"]}
    p, t, k = None, 0.0, 0
    while t < T:
        es, ey = rt.frenet(car.x, car.y)
        sc.step(t, dt, es)
        objs, light = sc.truth(t)
        if fault is not None and t >= fault:
            pc.ok = False
        if k % 5 == 0:
            wm = pc.run(es, objs, light, dt * 5)
            p = bh.plan(es, car.v, wm, sup=wm is None)
        a, vr = lo.cmd(es, car.v, p, dt)
        d = la.cmd(car, es)
        car.step(d, a, dt)
        ld = [o for o in objs if o["cls"] == "car"]
        pd = [o for o in objs if o["cls"] == "ped"][0]
        px, py = rt.xy(pd["s"])
        h = rt.hd(pd["s"])
        px, py = px - pd["d"] * np.sin(h), py + pd["d"] * np.cos(h)
        for kk, vv in zip(lg, [t, car.x, car.y, car.v, vr, a, p["mode"], ey,
                               ld[0]["s"] - es - 4.5 if ld else np.nan, np.hypot(px - car.x, py - car.y)]):
            lg[kk].append(vv)
        if p["mode"] == "MRC_REACHED" and car.v < 0.01:
            break
        t += dt
        k += 1
    return rt, sc, {k: np.array(v) for k, v in lg.items()}


def bands(ax, t, m):
    i0 = 0
    for i in range(1, len(m) + 1):
        if i == len(m) or m[i] != m[i0]:
            ax.axvspan(t[i0], t[i - 1], color=CL[m[i0]], alpha=0.8, lw=0)
            i0 = i


def plot(rt, sc, g, fn, ttl):
    f = plt.figure(figsize=(13, 8.5))
    gs = f.add_gridspec(3, 2, width_ratios=[1, 1.5], hspace=0.45, wspace=0.25)
    a0 = f.add_subplot(gs[:, 0])
    a0.plot(rt.p[:, 0], rt.p[:, 1], color="#9ca3af", lw=6, alpha=0.5, label="route (lane centre)")
    a0.plot(g["x"], g["y"], color="#1d4ed8", lw=1.6, label="ego trajectory")
    for q, c, lb in [(sc.light["s"], "#dc2626", "traffic light"), (sc.ped["s"], "#d97706", "pedestrian crossing"), (sc.dest, "#6d28d9", "destination")]:
        x, y = rt.xy(q)
        a0.scatter([x], [y], c=c, s=70, zorder=5, label=lb, edgecolors="black")
    a0.scatter([g["x"][-1]], [g["y"][-1]], marker="*", s=200, c="#111827", zorder=6, label="final position")
    a0.set_aspect("equal")
    a0.set_xlabel("x [m]")
    a0.set_ylabel("y [m]")
    a0.set_title("Top view")
    a0.legend(fontsize=8, loc="upper left")
    a0.grid(alpha=0.3)
    a1 = f.add_subplot(gs[0, 1])
    bands(a1, g["t"], g["m"])
    a1.plot(g["t"], g["v"], color="#1d4ed8", label="speed v")
    a1.plot(g["t"], g["vr"], color="#111827", ls="--", lw=1, label="reference v_ref")
    a1.set_ylabel("m/s")
    a1.set_title("Speed and behaviour state")
    a1.legend(fontsize=8, loc="upper right")
    a2 = f.add_subplot(gs[1, 1])
    bands(a2, g["t"], g["m"])
    a2.plot(g["t"], g["a"], color="#b91c1c")
    a2.set_ylabel("m/s²")
    a2.set_title("Longitudinal acceleration command")
    a3 = f.add_subplot(gs[2, 1])
    a3.plot(g["t"], g["ey"], color="#15803d", label="lateral error e_y")
    a3.set_ylabel("e_y [m]", color="#15803d")
    a3.set_ylim(-0.6, 0.6)
    a3.set_xlabel("time [s]")
    a3.set_title("Lateral tracking error and clearances")
    b3 = a3.twinx()
    b3.plot(g["t"], g["gap"], color="#1d4ed8", label="gap to lead car")
    b3.plot(g["t"], np.where(g["dp"] < 60, g["dp"], np.nan), color="#d97706", label="distance to pedestrian")
    b3.set_ylabel("clearance [m]")
    b3.set_ylim(0, 70)
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = b3.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3)
        a.set_xlim(0, g["t"][-1])
    from matplotlib.patches import Patch
    hs = [Patch(color=v, label=k) for k, v in CL.items() if k in set(g["m"])]
    f.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=8.5, frameon=False)
    f.suptitle(ttl, fontsize=12, fontweight="bold")
    f.savefig(fn, dpi=170, bbox_inches="tight")
    plt.close(f)


def summary(sc, g, nm):
    m = g["m"]
    st = [m[0]] + [m[i] for i in range(1, len(m)) if m[i] != m[i - 1]]
    gp = g["gap"][~np.isnan(g["gap"])]
    print(f"[{nm}]")
    print(f"  duration          : {g['t'][-1]:.1f} s")
    print(f"  state sequence    : {' -> '.join(st)}")
    print(f"  max |lateral err| : {np.max(np.abs(g['ey'])):.2f} m")
    print(f"  min gap to lead   : {gp.min():.1f} m" if len(gp) else "  min gap to lead   : n/a")
    print(f"  min dist to ped   : {g['dp'].min():.1f} m")
    print(f"  max decel         : {-g['a'].min():.2f} m/s^2")
    print(f"  final speed       : {g['v'][-1]:.2f} m/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", type=float, default=None)
    ap.add_argument("--out", default="results")
    ar = ap.parse_args()
    import os
    os.makedirs(ar.out, exist_ok=True)
    rt, sc, g = run()
    plot(rt, sc, g, f"{ar.out}/sim_nominal.png", "Scenario A: nominal L4 mission (lead car, red light, pedestrian, drop-off)")
    summary(sc, g, "nominal")
    ft = ar.fault if ar.fault is not None else 22.0
    rt, sc, g = run(fault=ft)
    plot(rt, sc, g, f"{ar.out}/sim_fault_mrm.png", f"Scenario B: perception failure at t = {ft:.0f} s triggers minimal-risk manoeuvre")
    summary(sc, g, "fault")

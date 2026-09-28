# L4 AV Architecture Prototype (Python)

Closed-loop prototype of the perception, decision and control modules described in the project report (Section 11).

## Structure

| File | Module |
|---|---|
| `av/world.py` | Route (lane centre), Frenet conversion, scripted scenario (lead car, traffic light, pedestrian, destination) |
| `av/perception.py` | Noisy range-gated detections + alpha-beta tracker, traffic-light visibility |
| `av/decision.py` | Behaviour planner FSM: LANE_FOLLOW, VEHICLE_FOLLOW, STOP_SIGNAL, YIELD_VRU, PULL_OVER, MRM, MRC_REACHED |
| `av/control.py` | Pure-pursuit lateral control, PI + feed-forward longitudinal control with jerk limit |
| `av/vehicle.py` | Kinematic bicycle model with steering-rate and acceleration limits |
| `run_sim.py` | Runs Scenario A (nominal) and Scenario B (perception failure -> minimal risk manoeuvre), prints metrics, saves plots |

## Run

```bash
pip3 install -r requirements.txt
python3 run_sim.py
python3 run_sim.py --fault 30
```

Plots are written to `results/`.

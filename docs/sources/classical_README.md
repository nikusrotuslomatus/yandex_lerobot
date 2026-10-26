# SO-101 Arm — Classical Robotics in MuJoCo

A complete classical robotics implementation for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) 6-DOF robot arm in MuJoCo simulation. The project covers the full pipeline from Forward Kinematics through to a physics-based pick-and-place task using torque control.

## Project Structure

```
so101_arm/
├── config/
│   └── robot_params.py          # DH params, link lengths, inertia from XML
├── utils/
│   └── transforms.py            # Rx, Ry, Rz, quat_to_rot, make_T, skew, ...
├── kinematics/
│   ├── fk.py                    # Forward kinematics (body-by-body chain)
│   └── ik.py                    # Damped least-squares IK, multi-start
├── diff_kinematics/
│   └── jacobian.py              # Full 6×N geometric Jacobian, manipulability
├── dynamics/
│   ├── rnea.py                  # Recursive Newton-Euler (gravity, inertia)
│   └── tmt.py                   # Transformation Matrix Transposition M(q)
├── control/
│   ├── trajectory.py            # quintic_interp, linear_interp, lerp, slerp
│   └── controllers.py           # FF+FB, gravity_pd, impedance, hybrid
├── model/
│   ├── assets/                  # STL mesh files
│   ├── so101_new_calib.xml      # Original robot model
│   ├── so101_torque.xml         # Motor actuators + graspframe + finger pads
│   ├── scene_torque.xml         # Torque-control scene (no objects)
│   └── scene_pickplace.xml      # Pick-and-place scene (cube + target)
└── simulation/
    ├── sim_pickplace.py         # Full pick-and-place (IK + FF+FB + physics)
    ├── sim_impedance.py         # Cartesian impedance control demo
    ├── sim_hybrid.py            # Hybrid force-position control demo
    └── mj_sim_control.py        # Trajectory tracking with FF+FB torque control
```

## Pipeline

Each phase builds on the previous, verified independently before moving to the next.

| Phase | Module | Method | Verified |
|-------|--------|--------|---------|
| Forward Kinematics | `kinematics/fk.py` | Body-by-body transform chain | 0.00 mm vs MuJoCo |
| Inverse Kinematics | `kinematics/ik.py` | Damped least-squares Jacobian iteration | < 1 mm, 5–13 iters |
| Differential Kinematics | `diff_kinematics/jacobian.py` | Geometric Jacobian, analytical vs numerical | 1.76e-7 max error |
| Dynamics — RNEA | `dynamics/rnea.py` | Recursive Newton-Euler | 0.033 N·m vs MuJoCo |
| Dynamics — TMT | `dynamics/tmt.py` | Composite Jacobian inertia matrix | matches RNEA 1.73e-18 |
| Trajectory | `control/trajectory.py` | Quintic polynomial, SLERP | zero vel/acc endpoints |
| FF+FB Control | `mj_sim_control.py` | Feedforward + feedback torque | max 1.9 mm EE error |
| Impedance Control | `sim_impedance.py` | Cartesian spring-damper | spike-and-decay at waypoints |
| Hybrid Force-Position | `sim_hybrid.py` | Force in y, position in x/z | F_y = 3 N constant |
| Pick and Place | `sim_pickplace.py` | IK + quintic + FF+FB + physics | cube lifted and placed |

## Key Design Decisions

### Actuator Model (`so101_torque.xml`)
The original `so101_new_calib.xml` uses `<position kp="998.22">` servo actuators — `data.ctrl` is interpreted as joint angle setpoints. For torque control, `so101_torque.xml` replaces these with `<motor>` actuators so `data.ctrl` becomes raw torque [N·m]. A separate `<position>` actuator is appended for the gripper finger only (mixed control: arm = torques, gripper = position setpoint).

### graspframe vs gripperframe
The robot has two end-effector sites:
- `gripperframe` — fingertip TCP, 9.8 cm from the wrist roll joint
- `graspframe` — midpoint **between** the fingers, 3.8 cm from the wrist roll joint

IK targets `graspframe` for grasping so the jaw centres on the object. Since `graspframe` is not in the analytical FK chain, an iterative correction loop reads the actual `graspframe` position from MuJoCo after each IK solve and shifts the target until the error falls below 8 mm (typically 2–4 outer iterations).

### Finger Collision Pads
Mesh-to-mesh contact generates unstable single contact points, causing the cube to wobble and bounce. Two small box geoms (`static_finger_pad`, `moving_finger_pad`, 2.5 mm) are added to the fingertips in `so101_torque.xml` to give stable flat contact surfaces. This follows the fix documented in [MuJoCo issue #239](https://github.com/google-deepmind/mujoco/issues/239) .

### Physics Settings (`scene_pickplace.xml`)
```xml
<option noslip_iterations="3" impratio="10" cone="elliptic"/>
<default><geom solref="0.001 1" solimp="0.99 0.99 0.001"/></default>
```
Elliptic friction cone + no-slip iterations + stiff contact prevents slip without requiring soft contact springs.

### Torque Controller
```
tau = M(q)*qdd_ref + C(q,qdot) + Kp*(q_ref - q) + Kv*(qdot_ref - qdot)
```
`M(q)` from TMT, `C(q,qdot)` (gravity + Coriolis) from RNEA. Dynamics cache updated every 10 steps for speed.

## Pick and Place Sequence

```
home → above grasp → pre-grasp → [APPROACH] → GRASP → lift →
transport → pre-place → [APPROACH] → PLACE → RELEASE → lift → home
```

- `[APPROACH]` is a short horizontal move from pre-grasp to grasp, keeping the gripper at cube height throughout — no diagonal descent that would collide with the cube face.
- Cube: 3 cm (half-size 0.015 m), mass 0.03 kg, wood friction, at `P_GRASP = [0.40, -0.10, 0.015]`.
- Target: `P_PLACE = [0.40, 0.10, 0.015]`.

## Setup

```bash
# Create environment
conda create -n mujoco_env python=3.11
conda activate mujoco_env
pip install mujoco numpy matplotlib

# Run pick and place
cd /path/to/so101_arm
python3 simulation/sim_pickplace.py

# Run torque trajectory tracking
python3 simulation/mj_sim_control.py

# Run impedance control
python3 simulation/sim_impedance.py

# Run hybrid force-position control
python3 simulation/sim_hybrid.py
```

## Results

| Metric | Value |
|--------|-------|
| FK accuracy vs MuJoCo | 0.00 mm |
| IK convergence | < 1 mm in 5–13 iterations |
| FF+FB trajectory tracking | max 1.9 mm EE error |
| graspframe accuracy after correction | < 8 mm |
| Pick-and-place placement XY error | ~24 mm |
| Cube lift height | ~0.12 m |

## References

- [SO-ARM100 / SO-101 Robot](https://github.com/TheRobotStudio/SO-ARM100) — hardware and simulation model
- [MuJoCo issue #239](https://github.com/google-deepmind/mujoco/issues/239) — finger pad collision fix


# Timed native experts

Three additional task classes inherit the native scene and success conditions:
`pick_dual_bottles_timed`, `scan_object_timed`, `place_dual_shoes_timed`.
Set `right_start_offset_s` in setup_demo (seconds of simulation time).
Positive values delay the right arm; negative values delay the left. Values are
rounded to the nearest physics step (4 ms at the native 250 Hz). Start delays do
not change the speed or sample count of any native planned action.

The two bottle lanes have no intermediate barrier. For scanning, the object
lane also establishes the scan pose independently; scanner alignment starts
only after both lanes report ready. The scanner/object roles remain randomized
by the original task; timing always names physical left/right arms.

Both shoe lanes grasp, lift, and approach from the outside. The shoe-box entry
corridor has capacity one: placement and withdrawal reserve the overlapping
swept workspace. An overlapping request inserts the explicit
`wait_outside_shared_workspace` local subtask, holding the grasped shoe outside
the box's transformed collision bounds plus 15 mm clearance. No overlapping
reservation means no waiting subtask. Ties are deterministic (left first), and
the first arm to reach the corridor otherwise goes first. This is conservative
geometric resource exclusion, not an exact time-dependent mesh collision
oracle. Per-step inter-arm/shoe penetration checks fail the rollout on a
penetration deeper than 2 mm. A successful demonstration is not a proof of
collision freedom for every scene, embodiment, or arbitrary timing choice.

Collect exactly one explicit episode (no automatic seed search or replay):

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/environment/bin/python script/collect_timed_demo.py \
  pick_dual_bottles --seed 0 --right-start-offset-s 2 --output /external/new-episode
```

Replace task with `scan_object` or `place_dual_shoes` as needed. The output must
not exist. The output includes native HDF5 with RGB, arm/gripper state, end poses
and simulation timestamps; a three-camera `preview.mp4` with phase labels;
and `result.json` containing seed, code/environment provenance, requested
start offset and exact subtask/wait/entry/exit timestamps. Native `video/` is
the upstream fixed-30-FPS export; `preview.mp4` uses the actual simulation clock.

Run CPU scheduler tests with `python -m pytest tests/test_timed_expert.py`.


## Required New FOV camera contract

All timed experts install the existing `parallel_vla.robotwin_wrist_camera`
`centered_fovy90` preset after scene setup and before recording any frame.
Both wrists use vertical FOV 90 degrees, 320x240, principal point (160,120),
and the preset's calibrated gripper-to-camera transform (manifest SHA-256
`bd7a9d918a775f3d59b70ca04d1a9f4151c9602557487947e2ee49d8a461b0e5`).
The calibrated transform is essential; changing only focal length is invalid.
For Coder A, expose the existing source dependency with
`PYTHONPATH=/home/coder/share/parallelVLA-piperx-sortletter-retime/src`.
Each paired-data frame verifies actual intrinsics and the measured mount;
HDF5 stores those measurements and the preset receipt for independent audit.
Data produced before this contract used native 37-degree wrists and must not
be included in the New FOV paired datasets.

Paired HDF5 distinguishes measured `observation/state` from native commanded
`joint_action/vector`. LeRobot states use measured joints and normalized actual
grippers; actions use the next sampled command target. Physical actor poses are
also captured. Source command hashes remain invariant across timing variants.

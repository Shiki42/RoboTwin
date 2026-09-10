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

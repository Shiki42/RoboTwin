# CTR expert variants

Display names: **scan object-CTR**, **Blocks Ranking RGB-CTR**.
Python task IDs: `scan_object_ctr`, `blocks_ranking_rgb_ctr`.
Use the measured-state/centered_fovy90 runtime described in TIMED_EXPERTS.md.

## Timing

`play_once` constructs a source program, planning and executing each arm's lane
sequentially. Source planning is not a delivered timing variant. Replay freezes
all joint position, velocity, gripper targets and internal phase boundaries.
Each independent stage is continuous per arm, without an intervening arm barrier.

Choose `--first left|right --delay-fraction p` with p in [0,1]. Each independent
stage delays the other arm by p times the first arm's **exact frozen stage length**.
At p=1 the other arm starts immediately upon first-arm completion. No average
length approximation is needed. The same physical first arm and p apply to all
independent stages. Replay consumes the frozen controls at 250 Hz without any
further speed changes or resampling.

Alternatively `--u u` selects the repository uniform offset for the first stage:
`delta = T_L - u*(T_L+T_R)`. Convert it to first arm and p, then reuse them for later
independent stages. Later stage offsets need not be uniformly distributed.
Omit timing arguments for concurrent replay. Source/variant identities and
per-stage raw/rounded delays are recorded; rounding is to nearest physics step.

For one source, review variants are left-first sequential, right-first sequential,
concurrent, and u={0,.2,.4,.6,.8}. The duplicate u=0 is intentional.

## Scan

Object is always left, scanner always right. The native object scan preparation
center is [-.03,-.02,.95] m. The arm bisector is x=0; sample an angle uniformly on
the 10 cm circle in the parallel y-z plane. Retain native object orientation and
functional-point scanning geometry. The right preparation pose is the native
scanner alignment for the unshifted object center, derived from that geometry.

1. Independent prepare: each arm grasps, lifts and moves to its preparation pose.
2. Barrier: both prepare lanes must finish. Left holds the object while the right
   aligns to its displaced functional point. Native scanning success turns the
   red cuboid green; failed geometry never changes the light.
3. Independent put_back: both arms withdraw, put their actor at x=-.24/+ .24,
   y=-.15 m on their own side, open, withdraw, and return home. These table positions
   are episode-invariant; actor center height depends on the asset's support height.
   Initial actor orientation is restored. Reuse first arm and p.

The indicator is at [-.032,-.20,1.155] m with half extents [.04,.0096,.0128]. Its collision
geometry must project entirely into the head image and not overlap the projected
initial object/scanner collision bounds. Scan success is latched separately from
whole-task success; both returned actors must be stationary and both arms home.

## Blocks

Four equal-size blocks: red/yellow left, blue/green right. Sample the two left
poses from separated x bands and mirror their positions/yaws through x=0 to the
right (red->blue, yellow->green). Left always handles red then yellow; right always
blue then green. Each arm's entire two-block sequence plus return home is one lane.

Fixed final centers: x={-.31,-.21,+.21,+.31}, y=-.17 m. Thus the groups have center
spacing .52 m. Left/right refers to world x, not the mirrored appearance of a
camera. Every physics step certifies **at least .10 m** between conservative
world-axis bounding boxes of opposing arm collision shapes and their blocks.
The fixed head camera uses fx=fy=260 px at 320x240 (FOVY approximately49.55°)
to include all four safety-separated blocks; wrist calibration is unchanged.
The fixed initial and target regions are intentionally separated enough to avoid
runtime arbitration. A violation rejects the rollout rather than adding waits.
Cross-arm penetration over 2 mm is also rejected for both tasks.

## Idle fields and output

This CTR variant uses the user's updated definition, separate from historical
paired dataset masks: `retime/left_idle` and `retime/right_idle` are true only when
the **entire observation-to-next-target action interval** belongs to an imposed
independent-stage start delay. Completed terminal holds, synchronization barriers,
cooperative holds and necessary settling are false. There is no `retime/overlap`
field. It is derivable from physics reasons/source advancement when needed.
The float32 `[left,right]` field `observation/arm_active_mask` is the inverse
of the delay-only idle flags and has N-1 rows, aligned with the action intervals.
Masks on Sequential/Concurrent previews are diagnostic metadata; those training
groups do not enable IdleMask. No training is part of this delivery.

Each variant includes `episode.hdf5`, `result.json`, and a synchronized 25 FPS
three-camera `preview.mp4` with stage and idle fields. HDF5 stores measured state,
command targets, actor poses, intrinsics/mount validation, physics reason codes,
source indices, and stage indices. The final observation is verification/target
material only: N observations produce N-1 action intervals. Native commanded
`joint_action/vector[1:]` is the action sequence for measured states `[:-1]`.

One explicit source (no seed search):

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/runtime/python script/collect_ctr_experts.py \
  scan_object_ctr --seed 0 --output /external/scan-source
```

One explicit replay (output must not exist):

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/runtime/python script/collect_ctr_experts.py \
  scan_object_ctr --seed 0 --source /external/scan-source \
  --first left --delay-fraction 1 --variant left_first --output /external/scan-left
```

Replace the task ID for blocks. Source failures require a stated correction or
bounded retry reason; no automatic retry, seed search or relaxed acceptance gates.
Results are reported/audit-pending until the user approves archival.

Audit a completed 16-episode review root (containing review-plan.json):

```bash
/path/to/runtime/python script/audit_ctr_experts.py /external/review-root
```

## Scan motion details

After grasp and lift, the right arm uses one native SE(3) move to reach the ready
position and horizontal orientation together. There is no in-place leveling
substep or additional outward preparation waypoint. Scan-align alone holds the
ready quaternion with [1,1,1,0,0,0] and checks <=2 degrees of actual orientation
drift and horizontal tilt at every physics step.

After scan success, return paths are computed once from the actual grasp geometry.
Both arms move outward, approach their fixed table return poses while restoring
actor orientation, lower, release, clear vertically by 6 cm, and return home.
The motion structure and lateral goals are mirrored; different grasp geometry
requires arm-specific tool orientations and heights. There is no initial upward
waypoint. The native planner remains responsible for collision-checked planning;
planning failure is terminal.

Return constraints use the robot base frame: base Z is held during outward retraction and approach, and base X/Y
plus orientation during lowering and post-release clearance. Held components are
anchored to current planner FK after native calibration. These are planner costs,
not guarantees of exact physical tool motion, so actual execution is audited.

Outward retraction also restores each actor's return orientation, avoiding the
large redundant wrist turn induced by holding the scan orientation. Both source
retractions use an explicit1.0-second duration (250 controls), with positions
interpolated and velocities scaled consistently. Other return segments retain
native planner speed. No whole-return slowdown is used. Concurrent starts and
finishes both retractions together; delayed variants preserve their original
independent-stage offsets and do not insert a new synchronization barrier.

Both rise above the scan-end height and any intermediate vertical rebound are
measured at every physics step until release. Qualification requires <=8 mm for
both arms in the source and all eight review replays. The generated review page
reports actual values, horizontal scan checks, and links to per-episode receipts.

## Shared50 qualified Scan cohort

`scan50_recipe.py` freezes candidate seeds0..199 and50 output slots. The collector
`collect_ctr_batch.py` tries each candidate once, in order, and accepts it only
when its source and all five prescribed physical timing replays pass. A rejected
candidate does not advance the output slot. Unexpected infrastructure errors stop
the run; exhausting200 candidates before50 accepted scenes also stops. There are
no source replans, threshold relaxations or unbounded seed searches. Native scene,
scan orientation and collision failures retain their checks and are recorded as
candidate rejections. Motion is the user-approved cd624178 expert.

All methods share the selected50 seeds and their frozen source controls.
Sequential contains50 left-first then50 right-first episodes; Concurrent has50.
CTR has100 in accepted-scene-major order: slot i uses u=i/100 and(i+50)/100.
Mixed selects left-first slots0..33, right-first0..32, and concurrent17..49.
Actual seeds and output slots are recorded separately; Mixed covers all50 scenes.
`export_ctr_batch.py` consumes the same plan, verifies multistage command identity,
exports masks only forCTR and computes exact global numeric quantiles. Artifacts
and candidate rejection traces stay outside Git. No publication is implied.

## Model-dependent Scan geometry correction (E801)

The scanner ready target uses an explicitly horizontal functional axis, preserving
native ready roll and the measured tool-to-scanner transform. Its functional
position is5cm on the scanner side of the unshifted actual object center. The
2cm shallower right grasp clears the calibrated wrist camera without modifying
camera mounts, collision geometry, masses or friction. Ready and scan motions cap
joint speed at0.8rad/s for grip stability; outward return stays exactly1second.
Preparation remains one combined orientation/translation operation; there is no
in-place leveling substage. Scanner checks remain2degrees; batch qualification
also enforces the existing8mm pre-release return rise/drawup limit. Failure traces
include measured ready transforms and contact bodies to distinguish grasp drift,
tracking error and self contact. Cross-model qualification improves some scenes,
but every source and timing replay still requires independent success checks.

## Wrist-clearance roll selection

Scanner preparation evaluates ten evenly spaced rolls about the horizontal scan
axis, all at the same functional-point position. It executes only the shortest
planned candidate whose whole right-wrist trajectory stays within1.4rad bend
and avoids equivalent multi-turn branches (joint coordinates and excursion within
pi). These are additional source-path rejection rules, not relaxed physics or
success checks. The same selected roll is held through scan-align; there is no
in-place leveling or physical candidate trial. The selected source is frozen
for every timing replay.

Candidate goals reuse the native single-goal cuRobo planner individually because
the installed batched retry path has mixed-IK-success tensor shape failures.
Unexpected planner errors propagate. Each path is trimmed using cuRobo's actual
interpolated trajectory length; padded terminal buffers are never executed.
Known physically infeasible candidates remain failures. No changes to robot
collision shapes, contact materials, camera calibration or success tolerances.

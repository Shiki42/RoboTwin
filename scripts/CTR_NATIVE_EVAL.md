# Native RoboTwin2 evaluation for CTR tasks

The native task IDs are `scan_object_ctr` and `blocks_ranking_rgb_ctr`.
They use the same `setup_demo`, `get_obs`, `take_action(action, action_type)`,
`check_success`, `set_instruction`, and `close_env` contract as `scan_object`
and `blocks_ranking_rgb`. No separate policy API or expert-stage callback is
required. Existing policies still implement `get_model`, `eval`, `reset_model`.

From the current RoboTwin main checkout, start your XPolicyLab policy server
with its qualified checkpoint, then use the native evaluation entry point:

```bash
bash scripts/eval_policy.sh --task_name scan_object_ctr \
  --policy_name YOUR_POLICY --host localhost --port YOUR_SERVER_PORT \
  --task_config demo_clean --seed 0 --test_num 100
```

Replace `scan_object_ctr` with `blocks_ranking_rgb_ctr` for four-block sorting.
`YOUR_POLICY` and `YOUR_SERVER_PORT` refer to an actual running policy service.
The main branch's `scripts/eval_policy_xpolicylab.py` owns policy transport and
rollout orchestration; these tasks expose the same task methods as native tasks.
New instruction templates include seen/unseen text and fixed physical-arm roles.
Native action limits are1200 forScan and1500 forBlocks. Wrist cameras use the
existing centered_fovy90 calibration; Blocks retains its wider fixed head view.

## Scan success

The scan event is latched from native scan geometry during policy execution,
requiring lifted items and a scanner-axis tilt no greater than2degrees. This
turns the indicator green. Final success additionally requires:

- object center within3.5cm XY of(-.24,-.15)m and scanner center within3.5cm XY
  of(+.24,-.15)m;
- center height within3.5cm of each asset's initial supported table height;
- each item has a physical tabletop contact, not merely a low center position;
- both measured grippers are at least80% open;
- each item's linear speed <=.02m/s and angular speed <=.1rad/s;
- all return conditions persist for.2seconds of native physics steps.

Scan success stays latched while returning the items. Neither the expert-owned
`return_complete` flag nor arm home poses gate policy success. Objects need not
recover a prescribed final orientation. Regions are episode-invariant in XY;
asset-specific support height accounts for different object models. The20cm and
10cm earlier design discussions do not alter the implemented fixed return centers.

## Blocks success

Four centers must occupy their existing targets: red(-.31,-.17), yellow(-.21,-.17),
blue(+.21,-.17), green(+.31,-.17)m. Each has2.5cm XY tolerance and1.5cm center-height
tolerance from table height plus its half-size. Their world-X ordering must be
red<yellow<blue<green. Table support, measured open grippers and the same.2second
stationarity condition are required. Expert completion/home flags are irrelevant.

## Clock and diagnostics

`Base_Task.take_action` increments `eval_physics_steps` once per actual physics
step. Repeated `check_success()` queries without stepping cannot satisfy the
stability interval. Episode setup resets the clock, scan latch and stability
window. `info['evaluation']` exposes scan/order prerequisite, per-item target
checks, measured release and success. The native evaluator's expert feasibility
pass runs before policy actions and uses its physically settled final state;
policy rollouts require the full timed stability interval.

Collection behavior (`eval_mode=False`) remains unchanged and continues to enforce
its expert-specific stage, home and frozen timing checks. Native policy evaluation
(`eval_mode=True`) uses the physical success contract above. These code paths are
separate so this interface update does not change the in-progress dataset source.

## Train-disjoint fixed evaluation seeds

`scripts/qualify_ctr_eval_seeds.py` consumes a CPU-frozen plan with the union of
all training seeds and at most500 ascending non-training candidates per task.
It accepts100 per task only after a real expert `play_once` and native physical
success with a0.2-second stable return, without the expert completion flag.
Each candidate executes once in an isolated process. Expected planning/geometry
failures are recorded; unexpected software errors stop the Run. No policy outcome
may change the accepted list. Full receipts, initial scene identities, control
hashes and source version accompany each list. These lists measure success on
expert-feasible scenes, not an unfiltered random-scene distribution.

Parallel seed screening may use different fixed worker counts per task. Results
are consumed in frozen candidate order, never completion order. Existing verified
receipts are reused by SHA-256; no seed is repeated during a concurrency change.
Already-running candidates beyond the100th accepted cutoff are allowed to finish
and retained as unused evidence, without entering or changing the evaluation list.

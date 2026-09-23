"""CTR staged frozen controls: no replanning or speed changes during retiming."""
import hashlib
import json
import math
from dataclasses import dataclass
import numpy as np
from .timed_expert import TimedExpert
from .paired_timing import SIDES, REASONS, ACTIVE, START, DONE, BARRIER, HOLD, CandidateRejected, encode_control, digest, start_steps

PHASES = ('grasp', 'lift', 'ready', 'scan_align', 'return', 'release', 'withdraw', 'home',
          'first_grasp', 'first_lift', 'first_place', 'first_withdraw',
          'second_grasp', 'second_lift', 'second_place', 'second_withdraw')

@dataclass
class StagedProgram:
    stages: list
    arrays: dict

    def hashes(self):
        return {key: digest(value) for key, value in self.arrays.items()}

    def save(self, path):
        np.savez_compressed(path, stages=np.array(json.dumps(self.stages)), **self.arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(json.loads(str(data['stages'])), {k:data[k] for k in data.files if k != 'stages'})

class StageRecorder(TimedExpert):
    def __init__(self, task):
        super().__init__(task)
        self.stages = []
        self.arrays = {}
        self.stage_name = None
        task.recorded_expert = self
        from .ctr_safety import CrossArmSafety
        self.safety = CrossArmSafety(task)

    def tick(self, controls, step):
        super().tick(controls, step)
        self.safety.check(step)

    def motion(self, label, actions):
        side = str(actions[0])
        try:
            for control in super().motion(label, actions):
                self.arrays[self.stage_name + '/' + side].append(encode_control(side, control, label, PHASES))
                yield control
        except RuntimeError as error:
            if str(error).startswith(('planning failed:', 'failed to construct')):
                raise CandidateRejected(str(error)) from error
            raise

    def stage(self, name, lanes, independent=True):
        self.stage_name = name
        self.stages.append(dict(name=name, independent=independent))
        for side in SIDES:
            self.arrays[name+'/'+side] = []
        # Plan one arm at a time; exported lanes are independently replayable.
        for side in SIDES:
            if side in lanes:
                self.run({side: lanes[side]})

    def program(self):
        return StagedProgram(self.stages, {k:np.asarray(v,dtype=np.float64).reshape(-1,16) for k,v in self.arrays.items()})

class StageClock:
    def __init__(self, program, *, u=None, first=None, delay_fraction=None):
        if u is not None and (not math.isfinite(u) or not 0 <= u <= 1):
            raise ValueError('uniform u must lie in [0,1]')
        if first is None and delay_fraction is not None:
            raise ValueError('delay_fraction requires an explicit first arm')
        self.program = program
        first_stage = program.stages[0]['name']
        lengths = {s:len(program.arrays[first_stage+'/'+s]) for s in SIDES}
        if first is None:
            _, raw, _ = start_steps(lengths['left'], lengths['right'], u=u)
            first = 'left' if raw >= 0 else 'right'
            delay_fraction = abs(raw)/lengths[first]
        elif u is not None:
            raise ValueError('choose uniform u or first/delay_fraction')
        if first not in SIDES or delay_fraction is None or not math.isfinite(delay_fraction) or not 0 <= delay_fraction <= 1:
            raise ValueError('first must name a physical arm; fraction must lie in [0,1]')
        self.first, self.fraction = first, delay_fraction
        self.step = 0
        self.stage_index = 0
        self.local_step = 0
        self.events = []
        self.schedules = []
        self.execution_hash = {k:hashlib.sha256() for k in program.arrays}
        self._enter()

    def _enter(self):
        stage = self.program.stages[self.stage_index]
        self.name = stage['name']
        self.rows = {s:self.program.arrays[self.name+'/'+s] for s in SIDES}
        self.index = dict.fromkeys(SIDES, 0)
        raw = self.fraction*len(self.rows[self.first]) if stage['independent'] else 0
        delay = int(math.floor(raw+0.5))
        self.starts = {s:delay if s != self.first else 0 for s in SIDES}
        self.schedules.append(dict(stage=self.name, independent=stage['independent'], start_step=self.step,
                                  first=self.first, fraction=self.fraction, raw_delay_steps=raw,
                                  delay_steps=delay, durations={s:len(v) for s,v in self.rows.items()}))
        self.events.append(dict(event='stage_start', stage=self.name, step=self.step))

    def finished(self):
        return self.stage_index == len(self.program.stages)

    def next(self):
        controls, reasons, indices = {}, [], []
        stage = self.program.stages[self.stage_index]
        for side in SIDES:
            index = self.index[side]
            if self.local_step < self.starts[side] and len(self.rows[side]):
                reason = START
            elif index == len(self.rows[side]):
                reason = (BARRIER if self.stage_index+1 < len(self.program.stages) else DONE) if stage['independent'] else HOLD
            else:
                reason = ACTIVE
                controls[side] = self.rows[side][index]
                self.execution_hash[self.name+'/'+side].update(controls[side].tobytes())
                if index == 0:
                    self.events.append(dict(event='arm_start', arm=side, stage=self.name, step=self.step))
                self.index[side] += 1
                if self.index[side] == len(self.rows[side]):
                    self.events.append(dict(event='arm_done', arm=side, stage=self.name, step=self.step+1))
            reasons.append(reason)
            indices.append(index if reason == ACTIVE else -1)
        self.step += 1
        self.local_step += 1
        end = all(self.index[s] == len(self.rows[s]) for s in SIDES)
        return controls, reasons, indices, end

    def advance_stage(self):
        if any(self.index[s] != len(self.rows[s]) for s in SIDES):
            raise RuntimeError('stage barrier released before both arms complete')
        self.events.append(dict(event='stage_done',stage=self.name,step=self.step))
        self.stage_index += 1
        self.local_step = 0
        if not self.finished():
            self._enter()

    def close(self):
        actual = {k:v.hexdigest() for k,v in self.execution_hash.items()}
        if actual != self.program.hashes():
            raise AssertionError('frozen control hash mismatch')
        return actual

def delay_masks(reasons, steps):
    """Only full action intervals spent in an imposed stage-start delay are idle."""
    reasons = np.asarray(reasons)
    return np.asarray([np.all(reasons[a:b] == START, axis=0) for a,b in zip(steps[:-1],steps[1:])],dtype=bool)

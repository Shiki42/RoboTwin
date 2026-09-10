"""Frozen-control paired datasets. Rendering and planning are outside the clock."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .timed_expert import TimedExpert

SIDES = ('left', 'right')
PHASES = ('grasp', 'lift', 'carry_to_target', 'outside_approach',
          'place_in_box', 'withdraw_from_box', 'object_ready', 'scan')
REASONS = ('active', 'start_delay', 'finished', 'workspace_wait',
           'scan_barrier', 'cooperative_hold', 'settle')
ACTIVE, START, DONE, WORKSPACE, BARRIER, HOLD, SETTLE = range(7)


class CandidateRejected(RuntimeError):
    """A scene/control combination fails qualification; never an environment error."""


def start_steps(left_steps, right_steps, *, u=None):
    raw = 0.0 if u is None else left_steps - float(u)*(left_steps+right_steps)
    delta = int(np.floor(raw+0.5)) if raw >= 0 else -int(np.floor(-raw+0.5))
    return dict(left=max(0,-delta), right=max(0,delta)), raw, delta


def encode_control(arm, control, phase):
    row = np.zeros(16, dtype=np.float64)
    row[1] = PHASES.index(phase)
    if arm+'_arm' in control:
        plan = control[arm+'_arm']
        row[0] = 0
        row[2:8] = plan['position'][0]
        row[8:14] = plan['velocity'][0]
    else:
        plan = control[arm+'_gripper']
        row[0] = 1
        row[14] = plan['result'][0]
        row[15] = plan['per_step']
    return row


def apply_control(task, side, row):
    if row[0] == 0:
        task.robot.set_arm_joints(row[2:8], row[8:14], side)
    else:
        task.robot.set_gripper(row[14], side, row[15])


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class Program:
    left: np.ndarray
    right: np.ndarray
    tail: np.ndarray
    gate: dict

    def arrays(self):
        return dict(left=self.left, right=self.right, tail=self.tail)

    def hashes(self):
        return {k:digest(v) for k,v in self.arrays().items()}

    def save(self, path):
        np.savez_compressed(path, **self.arrays(), gate=np.array(json.dumps(self.gate)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(a['left'],a['right'],a['tail'],json.loads(str(a['gate'])))


class RecordExpert(TimedExpert):
    def __init__(self, task):
        super().__init__(task)
        self.rows = {s:[] for s in SIDES}
        self.tail_rows = []
        task.recorded_expert = self

    def motion(self, label, actions):
        arm = str(actions[0])
        try:
            for control in super().motion(label, actions):
                row = encode_control(arm, control, label)
                if label in ('object_ready', 'scan'):
                    self.tail_rows.append(np.r_[SIDES.index(arm), row])
                else:
                    self.rows[arm].append(row)
                yield control
        except RuntimeError as e:
            if str(e).startswith(('planning failed:', 'failed to construct')):
                raise CandidateRejected(str(e)) from e
            raise

    def program(self):
        arrays = {s:np.asarray(self.rows[s], dtype=np.float64).reshape(-1,16) for s in SIDES}
        gates = {}
        for side,a in arrays.items():
            indices = np.flatnonzero(a[:,1] == PHASES.index('place_in_box'))
            if len(indices):
                gates[side] = int(indices[0])
        return Program(arrays['left'], arrays['right'],
                       np.asarray(self.tail_rows,dtype=np.float64).reshape(-1,17), gates)


class Clock:
    """Pure scheduler. The source index pauses only at the shoe-box boundary."""
    def __init__(self, program, task_name, u=None):
        self.p = program
        self.task_name = task_name
        self.starts,self.raw_delta,self.delta = start_steps(len(program.left),len(program.right),u=u)
        self.index = dict.fromkeys(SIDES,0)
        self.owner = None
        self.arrivals = {}
        self.step = 0
        self.tail_index = 0
        self.events = []
        self.previous = None
        self.execution_hash = {s:hashlib.sha256() for s in (*SIDES,'tail')}

    def emit(self,event,arm=None,**kw):
        self.events.append(dict(event=event,arm=arm,step=self.step,**kw))

    def finished(self):
        return all(self.index[s] == len(getattr(self.p,s)) for s in SIDES) and self.tail_index == len(self.p.tail)

    def next(self):
        controls = {}
        reasons = [ACTIVE,ACTIVE]
        indices = [self.index[s] for s in SIDES]
        # Release is resolved before admission at the same physics boundary.
        if self.owner and self.index[self.owner] == len(getattr(self.p,self.owner)):
            self.emit('workspace_exit',self.owner)
            self.owner = None
        for s in SIDES:
            if self.step >= self.starts[s] and s in self.p.gate and self.index[s] == self.p.gate[s]:
                self.arrivals.setdefault(s,self.step)
        if self.owner is None and self.arrivals:
            self.owner = min(self.arrivals,key=lambda s:(self.arrivals[s],s))
            self.arrivals.pop(self.owner)
            self.emit('workspace_enter',self.owner)
        front_done = all(self.index[s] == len(getattr(self.p,s)) for s in SIDES)
        if front_done and self.tail_index < len(self.p.tail):
            if self.tail_index == 0:
                self.emit('scan_barrier_release')
            row = self.p.tail[self.tail_index]
            s = SIDES[int(row[0])]
            controls[s] = row[1:]
            reasons = [HOLD,HOLD]; reasons[SIDES.index(s)] = ACTIVE
            self.execution_hash['tail'].update(row.tobytes())
            indices = [-1,-1]; indices[SIDES.index(s)] = self.tail_index
            self.tail_index += 1
        else:
            for a,s in enumerate(SIDES):
                i = self.index[s]; trajectory = getattr(self.p,s)
                if self.step < self.starts[s]:
                    reasons[a] = START; indices[a] = -1
                elif i == len(trajectory):
                    reasons[a] = BARRIER if self.task_name=='scan_object' else DONE
                elif s in self.p.gate and i == self.p.gate[s] and self.owner != s:
                    reasons[a] = WORKSPACE
                else:
                    if i == 0:
                        self.emit('arm_start',s)
                    controls[s] = trajectory[i]
                    self.execution_hash[s].update(trajectory[i].tobytes())
                    self.index[s] += 1
                    if self.index[s] == len(trajectory):
                        self.events.append(dict(event='arm_done',arm=s,step=self.step+1))
        for a,s in enumerate(SIDES):
            previous = None if self.previous is None else self.previous[a]
            if previous != reasons[a]:
                self.emit('state_change',s,reason=REASONS[reasons[a]])
        self.previous = reasons
        self.step += 1
        return controls, reasons, indices

    def close(self):
        if self.owner:
            self.emit('workspace_exit',self.owner)
            self.owner = None
        actual = {k:v.hexdigest() for k,v in self.execution_hash.items()}
        if actual != self.p.hashes():
            raise AssertionError(f'frozen control hash mismatch: {actual} != {self.p.hashes()}')
        return actual


def action_masks(reasons, sample_steps):
    reasons = np.asarray(reasons)
    masks, overlap = [],[]
    for a,b in zip(sample_steps[:-1],sample_steps[1:]):
        segment = reasons[a:b]
        if len(segment)==0:
            raise ValueError('empty action interval')
        masks.append(np.all(np.isin(segment,[START,DONE]),axis=0))
        overlap.append(bool(np.any(np.all(segment==ACTIVE,axis=1))))
    masks = np.asarray(masks,dtype=bool)
    if np.any(np.all(masks,axis=1)):
        raise AssertionError('new both-arms-idle training interval')
    return masks,np.asarray(overlap,dtype=bool)

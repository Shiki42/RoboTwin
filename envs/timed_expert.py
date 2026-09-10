"""Independent native expert lanes on one physics clock.

Positive right_start_offset_s delays the right arm; negative delays the left.
The shoe-box entry corridor has capacity one. A reservation predicts a conflict
between overlapping placement/retreat sweeps; it is released after withdrawal.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Acquire:
    resource: str


@dataclass(frozen=True)
class Release:
    resource: str


class Timeline:
    """Cooperative scheduler; generators advance only after their last sample."""
    def __init__(self, dt, tick, event=None, validate_wait=None):
        self.dt = dt
        self.tick = tick
        self.step = 0
        self.events = []
        self.owners = {}
        self.event_callback = event
        self.validate_wait = validate_wait

    def emit(self, name, arm=None, **fields):
        row = dict(event=name, arm=arm, step=self.step, time_s=self.step*self.dt, **fields)
        self.events.append(row)
        if self.event_callback:
            self.event_callback(row)

    def run(self, lanes, right_start_offset_s=0):
        offset = float(right_start_offset_s)
        if not math.isfinite(offset):
            raise ValueError('start offset must be finite')
        # Round to nearest physics step, with ties away from zero.
        delay = int(math.floor(abs(offset)/self.dt + 0.5))
        starts = {arm: (delay if (arm == 'right') == (offset >= 0) else 0)
                  for arm in lanes}
        start_step = self.step
        pending, started, waiting = {}, set(), set()
        active = dict(lanes)
        while active:
            controls = {}
            for arm in sorted(tuple(active), key=lambda a: (starts[a], a)):
                if self.step - start_step < starts[arm]:
                    continue
                if arm not in started:
                    self.emit('lane_start', arm)
                    started.add(arm)
                while True:
                    token = pending.pop(arm, None)
                    if token is None:
                        token = next(active[arm], None)
                    if token is None:
                        self.emit('lane_done', arm)
                        del active[arm]
                        break
                    if isinstance(token, Acquire):
                        owner = self.owners.get(token.resource)
                        if owner is not None and owner != arm:
                            if self.validate_wait:
                                self.validate_wait(arm)
                            if arm not in waiting:
                                self.emit('wait_outside_shared_workspace_start', arm,
                                          resource=token.resource, blocking_arm=owner,
                                          reason='overlapping_entry_and_placement_corridors')
                                waiting.add(arm)
                            pending[arm] = token
                            break
                        if arm in waiting:
                            self.emit('wait_outside_shared_workspace_end', arm, resource=token.resource)
                            waiting.remove(arm)
                        self.owners[token.resource] = arm
                        self.emit('shared_workspace_enter', arm, resource=token.resource)
                        continue
                    if isinstance(token, Release):
                        if self.owners.get(token.resource) != arm:
                            raise RuntimeError('resource released by non-owner')
                        del self.owners[token.resource]
                        self.emit('shared_workspace_exit', arm, resource=token.resource)
                        continue
                    controls.update(token)
                    break
            if not active:
                break
            self.tick(controls, self.step)
            self.step += 1
        if self.owners:
            raise RuntimeError('expert finished with an unreleased workspace')


class TimedExpert:
    def __init__(self, task):
        self.task = task
        self.timeline = Timeline(task.scene.get_timestep(), self.tick,
                                 event=self.on_event, validate_wait=self.validate_wait)
        self.frames = []
        self.contacts = []
        self.phase = {'left': 'start_delay', 'right': 'start_delay'}
        self.last_capture = -1

    def on_event(self, event):
        if event['event'] == 'wait_outside_shared_workspace_start':
            self.phase[event['arm']] = 'wait_outside_shared_workspace'
        elif event['event'] == 'wait_outside_shared_workspace_end':
            self.phase[event['arm']] = 'hold'

    def capture(self):
        t = self.task
        if t.save_data:
            t._update_render()
            t.validate_wrist_cameras()
            t._take_picture()
            self.frames.append(dict(frame=t.FRAME_IDX-1, step=self.timeline.step,
                                    time_s=self.timeline.step*self.timeline.dt,
                                    phases=dict(self.phase)))
            self.last_capture = self.timeline.step

    def tick(self, controls, step):
        t = self.task
        if t.save_data and t.save_freq is not None and step % t.save_freq == 0 and step != self.last_capture:
            self.capture()
        if controls:
            seq = dict.fromkeys(('left_arm', 'right_arm', 'left_gripper', 'right_gripper'))
            seq.update(controls)
            t.take_dense_action(seq, save_freq=None)
        else:
            t.scene.step()
        if t.render_freq and step % t.render_freq == 0:
            t._update_render()
            t.viewer.render()
        if hasattr(t, 'left_shoe'):
            names = {t.left_shoe.actor: 'left', t.right_shoe.actor: 'right'}
            for contact in t.scene.get_contacts():
                entity_a, entity_b = (body.entity for body in contact.bodies)
                a, b = entity_a.name, entity_b.name
                side_a = names.get(entity_a) or ('left' if a.startswith('fl_') else 'right' if a.startswith('fr_') else None)
                side_b = names.get(entity_b) or ('left' if b.startswith('fl_') else 'right' if b.startswith('fr_') else None)
                if side_a and side_b and side_a != side_b:
                    for p in contact.points:
                        if p.separation < -0.002:
                            row = dict(step=step, bodies=[a,b], separation=float(p.separation))
                            self.contacts.append(row)
                            raise RuntimeError(f'inter-arm/object collision: {row}')

    def motion(self, label, actions):
        t = self.task
        arm, sequence = actions
        arm = str(arm)
        if not t.plan_success or arm not in ('left', 'right'):
            raise RuntimeError(f'failed to construct {label}')
        self.phase[arm] = label
        self.timeline.emit('subtask_start', arm, subtask=label)
        for action in sequence:
            if action.action == 'move':
                plan = getattr(t, arm+'_move_to_pose')(
                    action.target_pose, constraint_pose=action.args.get('constraint_pose'))
                if not t.plan_success:
                    raise RuntimeError(f'planning failed: {arm} {label}')
                n = len(plan['position'])
                for i in range(n):
                    yield {arm+'_arm': {'position': plan['position'][i:i+1],
                                        'velocity': plan['velocity'][i:i+1]}}
            else:
                plan = t.set_gripper(set_tag=arm, **{arm+'_pos': action.target_gripper_pos})
                for i in range(plan['num_step']):
                    yield {arm+'_gripper': dict(result=plan['result'][i:i+1],
                                               num_step=1, per_step=plan['per_step'])}
        self.timeline.emit('subtask_end', arm, subtask=label)
        self.phase[arm] = 'hold'

    def validate_wait(self, arm):
        import numpy as np
        t = self.task
        shoe = getattr(t, arm+'_shoe')
        # Mesh-derived world bounds, including the shape's local transform/scale.
        points = []
        import sapien.physx as physx
        for component in shoe.actor.get_components():
            if isinstance(component, physx.PhysxRigidBodyComponent):
                for shape in component.get_collision_shapes():
                    vertices = np.asarray(shape.get_vertices()) * np.asarray(shape.get_scale())
                    transform = (shoe.get_pose() * shape.get_local_pose()).to_transformation_matrix()
                    points.append(vertices @ transform[:3,:3].T + transform[:3,3])
        if not points:
            raise RuntimeError('shoe has no collision mesh for waiting-region validation')
        x = np.concatenate(points)[:,0]
        edge = self.box_half_width + 0.015
        outside = x.max() < -edge if arm == 'left' else x.min() > edge
        if not outside:
            raise RuntimeError(f'{arm} waiting shoe is not outside box corridor: {x.min()}, {x.max()}')

    def run(self, lanes, offset=0):
        self.timeline.run(lanes, offset)
        if self.last_capture != self.timeline.step:
            self.capture()

    def finish(self):
        self.task.info['timing'] = dict(
            right_start_offset_s=self.task.right_start_offset_s,
            physics_dt_s=self.timeline.dt, events=self.timeline.events,
            frames=self.frames, cross_arm_collisions=self.contacts)
        return self.task.info


def validate_camera_geometry(intrinsic, width, height, actual_mount, expected_mount):
    import numpy as np
    if (width,height)!=(320,240):
        raise ValueError('centered_fovy90 requires 320x240 wrist images')
    k=np.asarray(intrinsic)
    expected_k=np.array([[height/2,0,width/2],[0,height/2,height/2],[0,0,1]])
    if not np.allclose(k,expected_k,atol=1e-3,rtol=0):
        raise ValueError(f'wrist camera does not satisfy centered FOVY90 intrinsics: {k}')
    error=float(np.max(np.abs(np.asarray(actual_mount)-expected_mount)))
    if error>2e-6:
        raise ValueError(f'wrist camera optical-axis mount mismatch: {error}')
    return dict(fovy_deg=math.degrees(2*math.atan(height/(2*k[1,1]))),
                cx=float(k[0,2]),cy=float(k[1,2]),mount_max_abs_error=error)


class TimedSetup:
    expert_driver_type = TimedExpert

    def setup_scene(self, **kwargs):
        super().setup_scene(**kwargs)
        # Select raster rendering before cameras are constructed. The deployed
        # OIDN backend fails on this host; rendering does not change physics.
        import sapien
        sapien.render.set_camera_shader_dir('default')
        sapien.render.set_ray_tracing_denoiser('none')

    def setup_demo(self, **kwargs):
        offset = float(kwargs.pop('right_start_offset_s', 0.0))
        if not math.isfinite(offset):
            raise ValueError('right_start_offset_s must be finite')
        save_data=kwargs.get("save_data",False)
        kwargs["save_data"]=False
        super().setup_demo(**kwargs)
        self.right_start_offset_s = offset
        from parallel_vla.robotwin_wrist_camera import install_wrist_camera_preset
        self.wrist_camera_receipt = install_wrist_camera_preset(self, 'centered_fovy90')
        self.validate_wrist_cameras()
        self.save_data=save_data

    def validate_wrist_cameras(self):
        import numpy as np
        import sapien
        expected=np.asarray(self.wrist_camera_receipt['gripper_to_camera_matrix'])
        report={}
        for side in ('left','right'):
            camera=getattr(self.cameras,side+'_camera')
            ee=self.get_arm_pose(side)
            ee_matrix=sapien.Pose(ee[:3],ee[3:]).to_transformation_matrix()
            actual=np.linalg.inv(ee_matrix) @ camera.entity.get_pose().to_transformation_matrix()
            report[side]=validate_camera_geometry(camera.get_intrinsic_matrix(),camera.width,camera.height,actual,expected)
            report[side]['actual_mount']=actual.tolist()
        return report

"""scan object-CTR: fixed roles, offset preparation, scan barrier, and return."""
import numpy as np
import sapien
from .scan_object import scan_object
from .timed_expert import TimedSetup
from .ctr_timing import StageRecorder
from .utils import ArmTag, create_box

class ScanRecorder(StageRecorder):
    def tick(self, controls, step):
        super().tick(controls, step)
        if self.phase['right'] == 'scan_align':
            self.task.validate_scan_translation()


class scan_object_ctr(TimedSetup, scan_object):
    record_actor_names = ('object', 'scanner')
    display_name = 'scan object-CTR'

    def sample_scanner_side(self):
        return 1

    def load_actors(self):
        super().load_actors()
        self.scan_angle = float(np.random.uniform(0, 2*np.pi))
        self.scan_offset = np.array([0, .1*np.cos(self.scan_angle), .1*np.sin(self.scan_angle)])
        self.scan_complete = False
        self.return_complete = False
        self.scan_indicator = create_box(self.scene, sapien.Pose([-.032, -.20, 1.155]),
                                        (.04, .0096, .0128), color=(1,0,0),
                                        is_static=True, name='SCAN_indicator')

    def setup_demo(self, **kwargs):
        super().setup_demo(**kwargs)
        self.validate_indicator()
        self.return_poses = {}
        for side,name in (('left','object'),('right','scanner')):
            pose = getattr(self,name).get_pose()
            self.return_poses[side] = sapien.Pose([-.24 if side=='left' else .24, -.15, pose.p[2]], pose.q)

    def validate_indicator(self):
        from .ctr_safety import CrossArmSafety
        geometry = CrossArmSafety(self)
        geometry.add(self.scan_indicator.actor, 'left')
        camera = self.cameras.get_config()['head_camera']
        intrinsic, extrinsic = camera['intrinsic_cv'], camera['extrinsic_cv']
        rectangles = {}
        for name in (*self.record_actor_names, 'scan_indicator'):
            actor = getattr(self,name).actor
            points = []
            for shapes in geometry.shapes.values():
                for entity,corners in shapes:
                    if entity == actor:
                        matrix = entity.get_pose().to_transformation_matrix()
                        points.extend(corners@matrix[:3,:3].T+matrix[:3,3])
            if not points:
                raise RuntimeError(f'missing projection geometry for {name}')
            points = np.asarray(points)
            projected = (np.c_[points,np.ones(len(points))]@extrinsic.T)@intrinsic.T
            if np.any(projected[:,2] <= 0):
                raise RuntimeError('indicator or initial actor is behind the head camera')
            uv = projected[:,:2]/projected[:,2:]
            rectangles[name] = np.array([uv.min(axis=0),uv.max(axis=0)])
        indicator = rectangles['scan_indicator']
        if np.any(indicator[0] < [0,0]) or np.any(indicator[1] >= [320,240]):
            raise RuntimeError(f'scan indicator outside head image: {indicator}')
        for name in self.record_actor_names:
            rectangle = rectangles[name]
            if np.all(indicator[0] < rectangle[1]) and np.all(rectangle[0] < indicator[1]):
                raise RuntimeError(f'scan indicator obscures initial {name} region')
        self.indicator_projection = {name:rectangle.tolist() for name,rectangle in rectangles.items()}

    def begin_scan_translation(self):
        self.scan_reference_quaternion = np.array(self.scanner.get_pose().q)
        self.scan_translation_audit = dict(max_orientation_error_deg=0.0, max_tilt_deg=0.0,
                                           tolerance_deg=2.0, physics_steps=0)
        self.validate_scan_translation()

    def validate_scan_translation(self):
        q = np.array(self.scanner.get_pose().q)
        cosine = np.clip(abs(np.dot(q,self.scan_reference_quaternion))/(np.linalg.norm(q)*np.linalg.norm(self.scan_reference_quaternion)),0,1)
        angle = float(np.degrees(2*np.arccos(cosine)))
        direction = self.scanner.get_functional_point(0,'matrix')[:3,:3]@np.array([0,0,-1])
        tilt = float(np.degrees(np.arcsin(np.clip(abs(direction[2])/np.linalg.norm(direction),0,1))))
        audit = self.scan_translation_audit
        audit['max_orientation_error_deg'] = max(audit['max_orientation_error_deg'],angle)
        audit['max_tilt_deg'] = max(audit['max_tilt_deg'],tilt)
        audit['physics_steps'] += 1
        if angle > audit['tolerance_deg'] or tilt > audit['tolerance_deg']:
            raise RuntimeError(f'scanner failed horizontal translation: {audit}')

    def translate_scanner(self, position):
        pose = np.array(self.get_arm_pose('right'))
        pose[:3] = position
        arm,actions = self.move_to_pose(ArmTag('right'),pose)
        actions[0].args['constraint_pose'] = [1,1,1,0,0,0]
        return arm,actions

    def complete_scan(self):
        if not scan_object.check_success(self):
            raise RuntimeError('scanner failed native alignment geometry')
        self.scan_complete = True
        for component in self.scan_indicator.actor.get_components():
            if isinstance(component, sapien.render.RenderBodyComponent):
                for shape in component.render_shapes:
                    shape.material.base_color = [0,1,0,1]

    def play_once(self):
        driver = ScanRecorder(self)
        def prepare(side, actor):
            arm = ArmTag(side)
            yield from driver.motion('grasp', self.grasp_actor(actor, arm, pre_grasp_dis=.08))
            yield from driver.motion('lift', self.move_by_displacement(arm, x=-.05 if side=='left' else .05, z=.13))
            if side == 'left':
                target = np.array(self.left_object_target_pose)
                target[:3] += self.scan_offset
                actions = self.place_actor(actor, arm, target, pre_dis=0, dis=0, is_open=False)
            else:
                # Source left lane has finished. Undo only its sampled translation;
                # retain the native object's actual scan orientation/functional point.
                target = np.array(self.object.get_functional_point(1))
                target[:3] -= self.scan_offset
                self.scanner_base_functional_target = target.tolist()
                _, ready_actions = self.place_actor(actor, arm, target, functional_point_id=0,
                                           pre_dis=.05, dis=.05, is_open=False)
                ready_pose = np.array(ready_actions[-1].target_pose)
                # Establish horizontal scan orientation while clear of the center,
                # then retain that orientation throughout both translations.
                orient_pose = np.array(self.get_arm_pose(arm))
                orient_pose[3:] = ready_pose[3:]
                yield from driver.motion('ready', self.move_to_pose(arm,orient_pose))
                actions = self.translate_scanner(ready_pose[:3])
            yield from driver.motion('ready', actions)
        driver.stage('prepare', {'left':prepare('left',self.object), 'right':prepare('right',self.scanner)})
        self.begin_scan_translation()
        destination = np.array(self.get_arm_pose('right')[:3]) + self.scan_offset
        driver.stage('scan_align', {'right':driver.motion('scan_align',
            self.translate_scanner(destination))}, independent=False)
        self.complete_scan()
        def put_back(side, actor):
            arm = ArmTag(side)
            # Clear the interaction area before rotating back toward the table.
            yield from driver.motion('withdraw', self.move_by_displacement(arm, x=-.12 if side=='left' else .12, z=.04))
            ee = self.get_arm_pose(arm)
            grasp = actor.get_pose().inv() * sapien.Pose(ee[:3],ee[3:])
            target = self.return_poses[side]
            above = sapien.Pose(target.p + np.array([0,0,.14]), target.q) * grasp
            low = sapien.Pose(target.p + np.array([0,0,.012]), target.q) * grasp
            yield from driver.motion('return', self.move_to_pose(arm, np.r_[above.p,above.q]))
            yield from driver.motion('return', self.move_to_pose(arm, np.r_[low.p,low.q]))
            yield from driver.motion('release', self.open_gripper(arm))
            yield from driver.motion('withdraw', self.move_by_displacement(arm,z=.12))
            yield from driver.motion('home', self.back_to_origin(arm))
        driver.stage('put_back', {'left':put_back('left',self.object),'right':put_back('right',self.scanner)})
        self.return_complete = True
        return driver.finish()

    def check_success(self):
        if not self.scan_complete or not self.return_complete:
            return False
        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            return False
        for side,name in (('left','object'),('right','scanner')):
            actor = getattr(self,name)
            if np.linalg.norm(actor.get_pose().p[:2]-self.return_poses[side].p[:2]) > .035:
                return False
            if abs(actor.get_pose().p[2]-self.return_poses[side].p[2]) > .035:
                return False
            if np.linalg.norm(np.array(self.get_arm_pose(side)[:3])-np.array(getattr(self.robot,side+'_original_pose')[:3])) > .015:
                return False
            for component in actor.actor.get_components():
                if isinstance(component,sapien.physx.PhysxRigidDynamicComponent):
                    if np.linalg.norm(component.linear_velocity) > .02 or np.linalg.norm(component.angular_velocity) > .1:
                        return False
        return True

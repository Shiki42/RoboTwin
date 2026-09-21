"""Blocks Ranking RGB-CTR: mirrored pairs, independent fixed-order arms."""
import numpy as np
import sapien
from .blocks_ranking_rgb import blocks_ranking_rgb
from .timed_expert import TimedSetup
from .ctr_timing import StageRecorder
from .utils import ArmTag, create_box

class blocks_ranking_rgb_ctr(TimedSetup, blocks_ranking_rgb):
    record_actor_names = ('red','yellow','blue','green')
    display_name = 'Blocks Ranking RGB-CTR'

    def load_actors(self):
        self.block_half_size = float(np.random.uniform(.015,.025))
        # Disjoint initial bands avoid unbounded rejection sampling.
        left = [(-np.random.uniform(.16,.20), np.random.uniform(-.04,.04)),
                (-np.random.uniform(.30,.33), np.random.uniform(-.04,.04))]
        yaw = np.random.uniform(-.75,.75,2)
        colors = ((1,0,0),(1,1,0),(0,0,1),(0,1,0))
        for index,(name,color) in enumerate(zip(self.record_actor_names,colors)):
            pair = index % 2
            x,y = left[pair]
            angle = yaw[pair]
            if index >= 2:
                x,angle = -x,-angle
            actor = create_box(self, sapien.Pose([x,y,.74+self.block_half_size],
                               [np.cos(angle/2),0,0,np.sin(angle/2)]),
                               (self.block_half_size,)*3, color=color, name=name)
            setattr(self,name,actor)
            self.add_prohibit_area(actor,padding=.05)
        # 0.52 m group-center spacing; certify >=0.10 m geometric clearance.
        self.targets = {name:[x,-.17,.74+self.table_z_bias,0,1,0,0]
                        for name,x in zip(self.record_actor_names,(-.31,-.21,.21,.31))}
        self.return_complete = False

    def play_once(self):
        driver = StageRecorder(self)
        def lane(side, names):
            arm = ArmTag(side)
            for ordinal,name in zip(('first','second'),names):
                actor = getattr(self,name)
                yield from driver.motion(ordinal+'_grasp', self.grasp_actor(actor,arm,pre_grasp_dis=.09))
                yield from driver.motion(ordinal+'_lift', self.move_by_displacement(arm,z=.09))
                yield from driver.motion(ordinal+'_place', self.place_actor(actor,arm,self.targets[name],
                    functional_point_id=0,pre_dis=.09,dis=.02,constrain='align'))
                yield from driver.motion(ordinal+'_withdraw', self.move_by_displacement(arm,z=.10))
            yield from driver.motion('home',self.back_to_origin(arm))
        driver.stage('sort', {'left':lane('left',('red','yellow')),'right':lane('right',('blue','green'))})
        self.return_complete = True
        return driver.finish()

    def check_success(self):
        if not self.return_complete or not self.is_left_gripper_open() or not self.is_right_gripper_open():
            return False
        for name,target in self.targets.items():
            actor = getattr(self,name)
            if np.linalg.norm(actor.get_pose().p[:2]-np.array(target[:2])) > .025:
                return False
            if abs(actor.get_pose().p[2]-(.74+self.table_z_bias+self.block_half_size)) > .015:
                return False
            for component in actor.actor.get_components():
                if isinstance(component,sapien.physx.PhysxRigidDynamicComponent):
                    if np.linalg.norm(component.linear_velocity) > .02 or np.linalg.norm(component.angular_velocity) > .1:
                        return False
        for side in ('left','right'):
            if np.linalg.norm(np.array(self.get_arm_pose(side)[:3])-np.array(getattr(self.robot,side+'_original_pose')[:3])) > .015:
                return False
        return self.red.get_pose().p[0] < self.yellow.get_pose().p[0] < self.blue.get_pose().p[0] < self.green.get_pose().p[0]

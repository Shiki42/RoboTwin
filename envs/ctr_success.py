"""Physical-state success conditions shared by native CTR task evaluation."""
from dataclasses import dataclass
import numpy as np

@dataclass
class StableWindow:
    duration_s: float = .2
    since_step: int | None = None

    def update(self, satisfied, step, dt):
        if not satisfied:
            self.since_step=None
            return False
        if self.since_step is None:
            self.since_step=step
        return (step-self.since_step)*dt >= self.duration_s-1e-9


def resting_in_target(position,target,linear_velocity,angular_velocity,table_contact,xy_radius,z_tolerance):
    return bool(table_contact
        and np.linalg.norm(np.asarray(position)[:2]-np.asarray(target)[:2])<=xy_radius
        and abs(position[2]-target[2])<=z_tolerance
        and np.linalg.norm(linear_velocity)<=.02
        and np.linalg.norm(angular_velocity)<=.1)


class NativeCtrEvaluation:
    def setup_demo(self,**kwargs):
        super().setup_demo(**kwargs)
        self._native_success_window=StableWindow()

    def native_targets_success(self,targets,xy_radius,z_tolerance,prerequisite=True):
        import sapien.physx
        released=bool(np.all(np.asarray(self.robot.get_measured_gripper_val())>=.8))
        supported=set()
        for contact in self.scene.get_contacts():
            entities=[body.entity for body in contact.bodies]
            if self.table in entities and any(point.separation<=.002 for point in contact.points):
                supported.update(entity for entity in entities if entity is not self.table)
        items={}
        for name,target in targets.items():
            actor=getattr(self,name);body=actor.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if body is None:
                raise RuntimeError(f'missing dynamic body for {name}')
            items[name]=resting_in_target(actor.get_pose().p,target,body.linear_velocity,
                body.angular_velocity,actor.actor in supported,xy_radius,z_tolerance)
        physical=bool(prerequisite and released and all(items.values()))
        # Native expert-feasibility checking calls play_once before policy actions.
        # It has already settled the actors; policy success requires a timed hold.
        if self.take_action_cnt==0:
            success=physical
        else:
            success=self._native_success_window.update(physical,self.eval_physics_steps,self.scene.get_timestep())
        self.info['evaluation']=dict(prerequisite=bool(prerequisite),released=released,
            target_items=items,stable_duration_s=.2,success=bool(success))
        return bool(success)

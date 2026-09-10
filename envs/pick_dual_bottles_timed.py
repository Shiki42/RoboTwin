from .pick_dual_bottles import pick_dual_bottles
from .timed_expert import TimedExpert, TimedSetup
from .utils import ArmTag


class pick_dual_bottles_timed(TimedSetup, pick_dual_bottles):
    def play_once(self):
        driver = TimedExpert(self)

        def lane(side, bottle, target):
            arm = ArmTag(side)
            yield from driver.motion('grasp', self.grasp_actor(bottle, arm, pre_grasp_dis=0.08))
            yield from driver.motion('lift', self.move_by_displacement(arm, z=0.1))
            yield from driver.motion('carry_to_target', self.place_actor(
                bottle, arm, target, functional_point_id=0, pre_dis=0, dis=0, is_open=False))

        driver.run({'left': lane('left', self.bottle1, self.left_target_pose),
                    'right': lane('right', self.bottle2, self.right_target_pose)}, self.right_start_offset_s)
        self.info['info'] = {'{A}': '001_bottle/base13', '{B}': '001_bottle/base16'}
        return driver.finish()

from .place_dual_shoes import place_dual_shoes
from .timed_expert import TimedExpert, TimedSetup, Acquire, Release
from .utils import ArmTag
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC


class place_dual_shoes_timed(TimedSetup, place_dual_shoes):
    def play_once(self):
        driver = TimedExpert(self)
        # Reuse the native transformed box bounds for the exclusive corridor.
        self.add_prohibit_area(self.shoe_box, padding=0)
        box_bounds = self.prohibited_area.pop()
        driver.box_half_width = max(abs(box_bounds[0]), abs(box_bounds[2]))

        def lane(side, shoe, target_id):
            arm = ArmTag(side)
            sign = -1 if side == 'left' else 1
            yield from driver.motion('grasp', self.grasp_actor(shoe, arm, pre_grasp_dis=0.1))
            yield from driver.motion('lift', self.move_by_displacement(arm, z=0.15))
            yield from driver.motion('outside_approach', self.move_by_displacement(
                arm, x=sign*0.35-float(shoe.get_pose().p[0]), y=-0.05,
                quat=GRASP_DIRECTION_DIC['top_down']))
            driver.validate_wait(side)
            # This is a conditional local waiting subtask, not an unconditional delay.
            # The scheduler emits wait_outside_shared_workspace only if occupied.
            yield Acquire('shoe_box_entry_corridor')
            yield from driver.motion('place_in_box', self.place_actor(
                shoe, arm, self.shoe_box.get_functional_point(target_id),
                functional_point_id=0, pre_dis=0.07, dis=0.02, constrain='align'))
            yield from driver.motion('withdraw_from_box', self.back_to_origin(arm))
            yield Release('shoe_box_entry_corridor')

        driver.run({'left': lane('left', self.left_shoe, 0),
                    'right': lane('right', self.right_shoe, 1)}, self.right_start_offset_s)
        # A real three-second settling interval, sampled on the same physics clock.
        def settle():
            for _ in range(round(3/driver.timeline.dt)):
                yield {}
        driver.run({'left': settle()})
        self.info['info'] = {'{A}': f'041_shoe/base{self.shoe_id}', '{B}': '007_shoe-box/base0'}
        return driver.finish()

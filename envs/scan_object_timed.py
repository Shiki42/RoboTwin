from .scan_object import scan_object
from .timed_expert import TimedExpert, TimedSetup
from .utils import ArmTag


class scan_object_timed(TimedSetup, scan_object):
    def play_once(self):
        driver = self.expert_driver_type(self)
        scanner_arm = ArmTag('left' if self.scanner.get_pose().p[0] < 0 else 'right')
        object_arm = scanner_arm.opposite

        def pickup(arm, actor):
            yield from driver.motion('grasp', self.grasp_actor(actor, arm, pre_grasp_dis=0.08))
            yield from driver.motion('lift', self.move_by_displacement(
                arm, x=0.05 if arm == 'right' else -0.05, z=0.13))
            driver.timeline.emit('scan_ready', str(arm))

        driver.run({str(scanner_arm): pickup(scanner_arm, self.scanner),
                    str(object_arm): pickup(object_arm, self.object)}, self.right_start_offset_s)
        driver.timeline.emit('scan_barrier_released', prerequisites=['left_ready', 'right_ready'])
        target = self.right_object_target_pose if object_arm == 'right' else self.left_object_target_pose
        driver.run({str(object_arm): driver.motion('object_ready', self.place_actor(
            self.object, object_arm, target, pre_dis=0, dis=0, is_open=False))})
        driver.run({str(scanner_arm): driver.motion('scan', self.place_actor(
            self.scanner, scanner_arm, self.object.get_functional_point(1),
            functional_point_id=0, pre_dis=0.05, dis=0.05, is_open=False))})
        self.info['info'] = {'{A}': f'112_tea-box/base{self.object_id}',
                             '{B}': f'024_scanner/base{self.scanner_id}',
                             '{a}': str(object_arm), '{b}': str(scanner_arm)}
        return driver.finish()

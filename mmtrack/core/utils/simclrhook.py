from mmcv.runner import HOOKS, Hook
@HOOKS.register_module()
class SimCLRHook(Hook):

    def __init__(self, value: int, num_step: int):
        self.value = value
        self.num_step = num_step
        self.step_size = value / num_step

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter <= self.num_step:
            runner.model.module.auxiliary.loss_decode.loss_weight = self.value - cur_iter * self.step_size

    def after_iter(self, runner):
        runner.model.module.auxiliary.iter += 1
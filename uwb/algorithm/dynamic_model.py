import numpy as np


class DynamicModel:
    def __init__(self, std=1):
        self.std = std

    def step(self, pos, vel):
        return DynamicModel.transition_function(pos, vel, std=self.std)

    @staticmethod
    def transition_function(current_pos, current_vel, std=1):
        next_state = np.copy(current_pos)
        next_state += current_vel
        next_vel = np.copy(current_vel)
        next_vel += (np.random.randn(np.prod(current_vel.shape)) * std).reshape(
            current_vel.shape
        )
        return next_state, next_vel

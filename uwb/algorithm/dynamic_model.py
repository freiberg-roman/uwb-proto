import numpy as np


class DynamicModel:
    def __init__(self, init_pos, init_vel, std=1):
        self.pos = init_pos
        self.vel = init_vel
        self.std = std

    def step(self):
        return DynamicModel.transition_function(self.pos, self.vel, std=self.std)

    @staticmethod
    def transition_function(current_pos, current_vel, std=1):
        next_state = np.copy(current_pos)
        next_state += current_vel
        next_vel = np.copy(current_vel)
        next_vel += np.random.randn(current_vel.shape[0]) * std
        return next_state, next_vel

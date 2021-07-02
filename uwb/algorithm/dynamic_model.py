import numpy as np


class DynamicModel:
    """Updates positions according to dynamic model.

    Updates positions according to dynamic model described in paper of
    `W. Suski <https://ieeexplore.ieee.org/document/6514113>`

    Attributes:
      std: Standard deviation used in dynamic model for updating velocities.
    """

    def __init__(self, std=1):
        """Initializes standard deviation."""
        self.std = std

    def step(self, pos, vel):
        """Performs one time step for the positions according to dynamics."""
        return DynamicModel.transition_function(pos, vel, std=self.std)

    @staticmethod
    def transition_function(current_pos, current_vel, std=1):
        """Performs one time step for the position according to dynamics."""
        next_state = np.copy(current_pos)
        next_state += current_vel
        next_vel = np.copy(current_vel)
        next_vel += (np.random.randn(np.prod(current_vel.shape)) * std).reshape(
            current_vel.shape
        )
        return next_state, next_vel

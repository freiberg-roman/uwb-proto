class ParticleFilter:
    """Base class for particle filters.

    Attributes:
        init_particles: initial particle positions
        init_weights: initial particle weights
    """

    def __init__(self, init_particles, init_weights):
        """Initializes particles and weights"""
        self.particles = init_particles
        self.weights = init_weights

    def update_weights(self, z):
        """Updates weights of particles"""
        pass

    def resample(self):
        """Resamples particles."""
        pass

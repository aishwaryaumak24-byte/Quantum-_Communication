from .noise_models import NoiseModel

class QuantumEnvironment:
    def __init__(self, distance_km):
        """
        distance_km: distance between sender and receiver (in km)
        """
        self.distance = distance_km
        self.time_step = 0
        self.noise_model = NoiseModel()

    def update_environment(self):
        """
        Update time and get current noise
        """
        self.time_step += 1
        noise = self.noise_model.dynamic_noise(self.time_step)
        return noise

    def get_state(self, noise):
        """
        State representation for RL agent
        """
        return {
            "distance": self.distance,
            "noise": noise
        }


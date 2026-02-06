import math
import random

class NoiseModel:
    def __init__(self, base_noise=0.05):
        """
        base_noise: minimum noise present in the channel
        """
        self.base_noise = base_noise

    def static_noise(self):
        """
        Static noise: noise does not change with time
        """
        return self.base_noise

    def dynamic_noise(self, time_step):
        """
        Dynamic noise: noise changes with time
        """
        noise = self.base_noise + 0.02 * math.sin(time_step)
        noise += random.uniform(-0.01, 0.01)

        # Ensure noise stays between 0 and 1
        noise = max(0.0, min(noise, 1.0))
        return noise

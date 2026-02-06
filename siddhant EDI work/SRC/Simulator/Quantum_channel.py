import math

class QuantumChannel:
    def __init__(self, loss_factor=0.02):
        """
        loss_factor: how fast signal weakens with distance
        """
        self.loss_factor = loss_factor

    def transmission_success(self, distance):
        """
        Calculates probability of successful transmission
        """
        success_probability = math.exp(-self.loss_factor * distance)

        # keep value between 0 and 1
        success_probability = max(0.0, min(success_probability, 1.0))
        return success_probability

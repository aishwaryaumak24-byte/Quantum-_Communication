class Metrics:
    @staticmethod
    def calculate_fidelity(success_probability, noise):
        """
        Calculates communication fidelity
        """
        fidelity = success_probability * (1 - noise)

        # keep fidelity between 0 and 1
        fidelity = max(0.0, min(fidelity, 1.0))
        return fidelity

    @staticmethod
    def calculate_error(fidelity):
        """
        Calculates error rate
        """
        return 1 - fidelity
if __name__ == "__main__":
    fidelity = Metrics.calculate_fidelity(0.8, 0.1)
    error = Metrics.calculate_error(fidelity)
    print("Fidelity:", fidelity)
    print("Error:", error)

class Neuron:
    """A placeholder spiking neuron class."""

    def __init__(self, threshold=1.0):
        self.potential = 0.0
        self.threshold = threshold

    def step(self, input_current: float) -> bool:
        """Update neuron state and return True if spike occurs."""
        self.potential += input_current
        if self.potential >= self.threshold:
            self.potential = 0.0
            return True
        return False

class Probability(float):
    """A numeric value in the range [0, 1]."""

    def __new__(cls, *args, **kwargs):
        result = super().__new__(cls, *args, **kwargs)
        if not 0.0 <= result <= 1.0:
            raise ValueError(float(result))
        return result

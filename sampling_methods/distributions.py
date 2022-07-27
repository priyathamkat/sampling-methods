import numpy as np
import numpy.typing as npt
from scipy.special import erf, erfinv

FloatArray = npt.NDArray[np.float64]


class Distribution:
    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        pass

    @property
    def mean(self):
        raise NotImplementedError(f"{self.__class__} does not implement mean")

    @property
    def mode(self):
        raise NotImplementedError(f"{self.__class__} does not implement mode")

    @property
    def variance(self):
        raise NotImplementedError(f"{self.__class__} does not implement variance")

    @property
    def stddev(self):
        raise NotImplementedError(f"{self.__class__} does not implement stddev")

    def sample(self, n: int) -> FloatArray:
        raise NotImplementedError(f"{self.__class__} does not implement sample")

    def pdf(self, x: FloatArray) -> FloatArray:
        raise NotImplementedError(f"{self.__class__} does not implement pdf")

    def log_pdf(self, x: FloatArray) -> FloatArray:
        raise NotImplementedError(f"{self.__class__} does not implement log_pdf")

    def cdf(self, x: FloatArray) -> FloatArray:
        raise NotImplementedError(f"{self.__class__} does not implement cdf")

    def icdf(self, x: FloatArray) -> FloatArray:
        raise NotImplementedError(f"{self.__class__} does not implement icdf")


class Normal(Distribution):
    def __init__(self, loc: float, scale: float) -> None:
        super().__init__()

        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self.stddev ** 2

    @property
    def stddev(self):
        return self.scale

    def sample(self, n: int) -> FloatArray:
        return self._rng.normal(loc=self.mean, scale=self.stddev, size=n)

    def pdf(self, x: FloatArray) -> FloatArray:
        z = (x - self.mean) / self.stddev
        return np.exp(-(z ** 2) / 2) / (np.sqrt(2 * np.pi) * self.stddev)

    def log_pdf(self, x: FloatArray) -> FloatArray:
        z = (x - self.mean) / self.stddev
        return -(z ** 2) / 2 - 0.5 * np.log(2 * np.pi * self.variance)

    def cdf(self, x: FloatArray) -> FloatArray:
        z = (x - self.mean) / self.stddev
        return 0.5 * (1 + erf(z / np.sqrt(2)))

    def icdf(self, x: FloatArray) -> FloatArray:
        return self.stddev * np.sqrt(2) * erfinv(2 * x - 1) + self.mean


class Categorical(Distribution):
    def __init__(self, probs: FloatArray) -> None:
        super().__init__()

        self.probs = probs
        self.support = np.arange(self.probs.size)

    @property
    def mean(self):
        return np.sum(self.probs * self.support)

    @property
    def mode(self):
        return np.argmax(self.probs)

    @property
    def variance(self):
        return np.sum(self.probs * self.support ** 2) - self.mean ** 2

    @property
    def stddev(self):
        return np.sqrt(self.variance)

    def sample(self, n: int) -> FloatArray:
        return self._rng.choice(self.probs.size, size=n, p=self.probs)

    def cdf(self, x: FloatArray) -> FloatArray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        supports = np.tile(self.support, (x.size, 1))
        indicator = np.where(supports <= x, 1, 0)
        return np.sum(self.probs * indicator, keepdims=False)

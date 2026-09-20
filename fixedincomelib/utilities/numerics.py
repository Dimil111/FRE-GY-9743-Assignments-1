import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                          InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                          extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def _bucket_index(self, x: float) -> int:
        """Index i such that f(x) = values[i], per the convention above."""
        idx = np.searchsorted(self.axis1_, x, side='left')
        return min(idx, self.length_ - 1)

    def interpolate(self, x: float) -> float:
        return float(self.values_[self._bucket_index(x)])

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        grad = np.zeros(self.length_)
        grad[self._bucket_index(x)] = 1.0
        return grad

    def _segment_overlaps(self, start_x: float, end_x: float) -> np.ndarray:
        """Length of overlap between [l, u] (l <= u) and each constant-value
        region: (-inf, x_0], (x_0, x_1], ..., (x_{N-2}, x_{N-1}], (x_{N-1}, +inf).
        The first and last regions share values[0] / values[-1] respectively
        with flat extrapolation, so this returns exactly `length_` overlaps,
        one per node/value.
        """
        x = self.axis1_
        n = self.length_
        overlaps = np.zeros(n)

        # left wing: (-inf, x_0], attributed to values[0]
        lo, hi = start_x, min(end_x, x[0])
        if hi > lo:
            overlaps[0] += hi - lo

        # interior buckets: (x_{i-1}, x_i], attributed to values[i]
        for i in range(1, n):
            lo, hi = max(start_x, x[i - 1]), min(end_x, x[i])
            if hi > lo:
                overlaps[i] += hi - lo

        # right wing: (x_{N-1}, +inf), attributed to values[-1]
        lo, hi = max(start_x, x[-1]), end_x
        if hi > lo:
            overlaps[-1] += hi - lo

        return overlaps

    def integrate(self, start_x: float, end_x: float) -> float:
        if start_x > end_x:
            return -self.integrate(end_x, start_x)
        overlaps = self._segment_overlaps(start_x, end_x)
        return float(np.dot(overlaps, self.values_))

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        if start_x > end_x:
            return -self.gradient_of_integrated_value_wrt_ordinate(end_x, start_x)
        return self._segment_overlaps(start_x, end_x)


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                                values: np.ndarray | List,
                                interpolation_method: InterpMethod,
                                extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')

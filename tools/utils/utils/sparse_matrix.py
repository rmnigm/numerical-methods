import numpy as np

import typing as tp


class SparseMatrix:
    """Sparse matrix class based on dict of nonzero values"""

    def __init__(self,
                 data: tp.Optional[tp.Dict[tp.Tuple[int, int], float]] = None,
                 n: int = 2):
        self.data = data or {}
        self.n = n

    def from_dense_matrix(self, matrix: np.ndarray) -> "SparseMatrix":
        self.n = len(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] != 0:
                    self[(i, j)] = matrix[i][j]
        return self

    def dense(self) -> np.ndarray:
        res = np.zeros((self.n, self.n))
        for (i, j), v in self.data.items():
            res[i, j] = v
        return res

    def __getitem__(self, key: tp.Tuple[int, int]) -> float:
        return self.data.get(key, 0)

    def __setitem__(self, key: tp.Tuple[int, int], value: float):
        self.data[key] = value

    def __iter__(self) -> tp.Iterator[tp.Tuple[int, int]]:
        items: tp.Iterable = self.data.items()
        return iter(items)

    def __len__(self):
        return self.n

    def __add__(self, other: "SparseMatrix") -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] += self[key]
        for key in other:
            result[key] += other[key]
        return result

    def __sub__(self, other: "SparseMatrix") -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] = self[key]
        for key in other:
            result[key] -= other[key]
        return result

    def __mul__(self, other: "SparseMatrix") -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            for key2 in other:
                result[(key[0], key2[1])] += self[key] * other[key2]
        return result

    def __rmul__(self, other: float) -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] = self[key] * other
        return result

    def __truediv__(self, other: float) -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] = self[key] / other
        return result

    def __neg__(self) -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] = -self[key]
        return result

    def __pow__(self, other: float) -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] = self[key] ** other
        return result

    def __rpow__(self, other: float) -> "SparseMatrix":
        result = SparseMatrix()
        for key in self:
            result[key] = other ** self[key]
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        return self.data == other.data

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        return self.data == other.data

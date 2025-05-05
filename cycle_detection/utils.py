# MIT License

# Copyright (c) 2025 JRC-ISIA

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

def default(val, default_val):
	if val is not None:
		return val
	return default_val

def normalize(x):
	return (x - x.min()) / (x.max() - x.min())

def delay_embedding(X: np.array, d: int = 1) -> np.array:
	return np.array([X[i : i + d] for i in range(X.shape[0] - d + 1)])

def distance_mat_(X: np.array) -> np.array:
	if len(X.shape) == 1 or X.shape[0] == 1:
		X = X.reshape(-1, 1)
	n = X.shape[0]
	M = (X @ X.T).reshape(n, n)
	D = np.diag(M).reshape(-1, 1) + np.diag(M).reshape(1, -1) - 2 * M
	return np.sqrt(D)

def distance_mat(X: np.array, delay: int = None) -> np.array:
	if delay:
		X = delay_embedding(X, delay).reshape(X.shape[0] - delay + 1, -1)
	return distance_mat_(X)

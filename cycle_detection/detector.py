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

from scipy.signal import correlate

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use("ggplot")

from typing import Literal
from functools import partial

from . import get_persistent_homology
from .utils import *



def surrogate_func(
	X: np.array, 
	delay: int = 0, 
	embedding_length: int = 1, 
	x_start: int = None
) -> np.array:

	# delay embedding
	if embedding_length > 1 and delay > 0:
		n = embedding_length
		X = np.array(
			[X[i : i + n * delay : delay] for i in range(X.shape[0] - delay * (n - 1))]
		)
		X = X.reshape(X.shape[0], -1)

	# repetetive or reocurring
	norms = (X**2).sum(axis=-1)
	if x_start is not None:
		return np.sqrt(
			np.fmax(norms[x_start] + norms - 2 * X @ X[x_start], 0)
		)

	# periodic
	N = X.shape[0]
	ncum_fwd = np.cumsum(norms)
	ncum_bwd = np.cumsum(norms[::-1])
	rxx = correlate(np.concatenate([X, np.zeros_like(X)]), X, mode="valid").reshape(-1)

	r = np.empty(N)
	for k in range(N):
		m = N - k
		r[k] = (ncum_fwd[-(k + 1)] + ncum_bwd[-(k + 1)]) - 2 * rxx[k]
		r[k] *= 1 / m

	return r


class CycleDetection:
	def __init__(
		self,
		behaviour: Literal["periodic", "repetitive", "recurring"],
		epsilon: float,
		delta: float,
		x_start: int = None,
		delay: int = 0,
		embedding_length: int = 1,
		max_plot_size=2000,
	):
		"""
		Topology-driven identification of repetitions in multi-variate time series
		arXiv: ...

		Parameters
		----------
				behaviour : Literal['periodic', 'repetitive', 'recurring']
						A string indicating which type of cyclicity to look for:

						``periodic`` 	(Definiton 1)
								One period repeats everty T time steps.
								``self.predict`` returns all period lengths. 
						``repetitive``	(Definiton 2)
								The trajectory is the same for each period,
								but the time of traversal is in different.
								Uses a reference point `x_start`, as well as
								delay embedding of length `delay` and 
								`embedding_length` number of samples.
								`self.predict` returns all cycle lengths.
						``recurring`` 	(Definiton 3)
								The point `x_start` is visited repeatedly.
								``self.predict`` returns all recurrence times.
				x_start : int, optional
						Reference point for 'repetitive' and  'recurring'.
						Defaults to None (i.e. 'periodic').
				delay : int, optional
						Number of delay embedding samples for 'repetitive'.
						Defaults to None / 1 (i.e. no delay embedding).
				epsilon: float 
						Maximum distance threshold in [0, 1].
						Distance to reference point for 'repetitive' and 'recurring',
						maximum pointwise distance for 'periodic'.
				delta : float
						Minimum persistence threshold in [0, 1].
				max_plot_size: int, optional
						The maximum number of elements considered in the distance vector
						for plotting. Also applies to the rows and cols of the
						distance matrix.
						The input time series X will be downsampled to this number.
						Defaults to 2000 (16MB with float32).

		Examples
		--------
		>>> import numpy as np
		>>> from cycle_detection import CycleDetection

		>>> t = np.linspace(0, 4*np.pi, 200)
		>>> x = np.sin(t)

		>>> CycleDetection('periodic', epsilon=0.2, delta=0.7).predict(x)
		[100  99   1]

		>>> CycleDetection('repetitive', epsilon=0.2, delta=0.7, x_start=0,
		>>>					delay=10, embedding_length=2).predict(x)
		[100  90]

		>>> CycleDetection('recurring', epsilon=0.2, delta=0.7, x_start=0).predict(x)
		[50 49 50 50  1]

		"""

		assert behaviour in [
			"periodic",
			"repetitive",
			"recurring",
		], f"""behaviour is one of [periodic, repetitive, recurring]
				Got '{behaviour}'."""
		self.behaviour = behaviour

		assert 0. <= delta and delta <= 1., f'Expected delta in [0, 1], but got {delta}.'
		assert 0. <= epsilon and epsilon <= 1., f'Expected epsilon in [0, 1], but got {epsilon}.'
		self.delta = delta
		self.epsilon = epsilon

		if behaviour == "periodic":
			self.x_start = None
			# we don't require delay embeddings here, but they can be used
			self.delay = default(delay, 0)
			self.embedding_length = default(embedding_length, 1)
			self.method = partial(
				surrogate_func,
				x_start=self.x_start,
				delay=self.delay,
				embedding_length=self.embedding_length,
			)

		if behaviour == "repetitive":
			assert isinstance(x_start, int), f'Expected integer value for x_start, but got {x_start}.'
			# we require delay embeddings here, else this method would be equivalent to 'recurring'
			assert delay > 0, f'Expected delay > 0, but got {delay}.' 
			assert embedding_length > 1, f'Expected embedding_length > 1, but got {embedding_length}.' 
			self.x_start = x_start
			self.delay = delay
			self.embedding_length = embedding_length

			self.method = partial(
				surrogate_func, 
				x_start=self.x_start, 
				delay=delay,
				embedding_length=self.embedding_length
			)

		if behaviour == "recurring":
			assert isinstance(x_start, int), f'Expected integer value for x_start, but got {x_start}.'
			self.x_start = x_start
			# we don't allow delay embeddings here, else this method would be equivalent to 'repetitive'
			self.delay = 0
			self.embedding_length = 1
			self.method = partial(
				surrogate_func,
				x_start=self.x_start,
				delay=self.delay,
				embedding_length=self.embedding_length,
			)

		self.max_plot_size = default(max_plot_size, 2000)

		self.X: np.array						# input signal
		self.D: np.array						# distance matrix
		self.d: np.array						# surrogate function
		self.persistence_diagram: np.array
		self.clusters: np.array					# binary epsilon delta criteron mask
		self.T: np.array						# recurrence times
		self.tau: np.array						# cycle lengths

	def get_persistent_minima(self, d=None):
		x = default(d, self.d)
		idcs, pers = get_persistent_homology(-x)

		pd = np.array([[idx, x[idx], x[idx] + p] for (idx, p) in zip(idcs, pers)])

		persistent_idcs = np.argwhere(
			(pers >= self.delta) & (pd[:, 1] <= self.epsilon)).reshape(-1)
		mask = np.zeros_like(idcs)
		mask[persistent_idcs] = 1
		return np.array(idcs[persistent_idcs]), pd, mask

	def predict(self, X: np.array) -> np.array:
		self.X = X
		if len(X.shape) == 1 or X.shape[0] == 1:
			self.X = self.X.reshape(-1, 1)

		self.d = self.method(X=self.X)
		self.d = normalize(self.d)

		idcs, self.persistence_diagram, self.clusters = self.get_persistent_minima()
		idcs = sorted(idcs)
		idcs = np.append(idcs, self.d.shape[0])

		self.T = idcs
		self.tau = idcs[1:] - idcs[:-1]

		return self.tau

	def downsample_plot(self, x: np.array) -> np.array:
		ds = int(np.ceil(x.shape[0] / self.max_plot_size))
		if ds > 1:
			return x[::ds]
		else:
			return x

	def build_matrix_(self):
		self.D = distance_mat(self.downsample_plot(self.X), delay=self.delay)

	def plot(self, plot_data=False, plot_matrix=True, export_path=None):
		if export_path:
			import matplotlib as mlp
			mlp.use("pgf")
			plt.rcParams.update(
				{
					"font.family": "serif",
				}
			)

		cols = 2 + plot_data + plot_matrix
		_, axes = plt.subplots(1, cols, figsize=[6 * (cols), 5])

		d = self.downsample_plot(self.d)
		_, pd, clusters = self.get_persistent_minima(d)
		pd[0, 2] = 1.08

		colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
		colors = [colors[i] for i in [3,1,0]]


		if plot_data:
			X = self.downsample_plot(self.X)
			if X.shape[1] == 2:
				axes[0].plot(*X.T, c=colors[-1])
				axes[0].set_xlabel(r"$x_1$")
				axes[0].set_ylabel(r"$x_2$")
			else:
				axes[0].plot(X, c=colors[-1])
				axes[0].set_xlabel("t")
			axes[0].set_title("Data")

		if plot_matrix:
			self.build_matrix_()
			axes[int(plot_data)].imshow(self.D)
			axes[int(plot_data)].set_title("Distance Matrix")
			axes[int(plot_data)].grid(False)
			axes[int(plot_data)].set_xlabel("t")
			axes[int(plot_data)].set_ylabel("t")

		# persistence diagram
		axes[-1].scatter(*pd.T[1:], s=30, c=clusters, cmap=ListedColormap(colors[:2]))
		axes[-1].plot(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), "k")
		axes[-1].plot(
			np.linspace(-1, 2, 2),
			np.linspace(-1, 2, 2) + self.delta,
			"grey",
			ls=":",
			label=rf"$\delta={self.delta}$",
		)
		axes[-1].axvline(
			self.epsilon, -1, 2, c="grey", ls="--", label=rf"$\epsilon={self.epsilon}$"
		)
		axes[-1].set_xlim([-0.1, 1.1])
		axes[-1].set_ylim([-0.1, 1.1])
		axes[-1].set_title("Persistence Diagram")
		axes[-1].set_xlabel("Birth")
		axes[-1].set_ylabel("Death")
		axes[-1].legend(loc="lower right")

		
		# surrogate function
		# reduce the number of non-relevant local minima for plotting
		clusters = clusters[:clusters.sum() + 
					         min(10 * clusters.sum(), (clusters == 0).sum())]
		pd = pd[:len(clusters)]

		axes[-2].plot(d, colors[-1])
		axes[-2].vlines(*pd[clusters.astype(bool)].T,  colors[1])
		axes[-2].vlines(*pd[~clusters.astype(bool)].T, colors[0])

		axes[-2].set_ylim([0, 1.05])
		axes[-2].set_xlabel("$\\tau$")
		axes[-2].set_title("Surrogate Function")

		if export_path:
			plt.savefig(export_path)
		else:
			plt.show()

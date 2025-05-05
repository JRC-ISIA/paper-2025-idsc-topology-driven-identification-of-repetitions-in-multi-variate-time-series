## Official implementation of ["Topology-driven identification of repetitions in multi-variate time series"](arXiv).

> **Abstract:** 
> Many multi-variate time series obtained in the natural sciences and engineering possess a repetitive behavior, as for instance statespace trajectories of industrial machines in discrete automation. 
> Recovering the times of recurrence from such a multi-variate time series is of a fundamental importance for many monitoring and control tasks. 
> For a periodic time series this is equivalent to determining its period length. 
> In this work we present a persistent homology framework to estimate recurrence times in multi-variate time series with different generalizations of cyclic behavior (periodic, repetitive, and recurring). 
> To this end, we provide three specialized methods within our framework that are provably stable and validate them using real-world data, including a new benchmark dataset from an injection molding machine.


### Install

```bash
git clone https://github.com/JRC-ISIA/paper-2025-idsc-topology-driven-identification-of-repetitions-in-multi-variate-time-series.git cycle_detection
cd cycle_detection
pip install .
```


### Usage

```python
import numpy as np
from cycle_detection import CycleDetection

t = np.linspace(0, 4*np.pi, 200)
x = np.sin(t)

detector = CycleDetection('repetitive', epsilon=0.2, delta=0.7, x_start=0,
				          delay=10, embedding_length=2).predict(x)
tau = detector.predict(x)
print(tau)
detector.plot()

```

### Citation
```bibtex
@inproceedings{SRMH25,
  keywords      = {cdg},
  title         = {{Topology-driven identification of repetitions in multi-variate time series}},
  author        = {Schindler, Simon and Reich, Elias Steffen and Messineo, Saverio and Huber, Stefan},
  booktitle     = Proc # " 7th " # iDSC # " (iDSC'25)",
  year          = {2025},
}
```
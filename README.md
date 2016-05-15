# graphBIX
Graph clustering by Bayesian inference with cross-validation model assessment.

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or above. See LICENSE.txt for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


* sbm.jl
	Bayesian inference of the stochastic block model using EM algorithm + belief propagation with the leave-one-out cross-validation.

USAGE
============

To start, the following package needs to be imported:
```
using DocOpt
```

For a given edgelist file, e.g. `edgelist.txt`,
```
julia sbmBIX.jl edgelist.txt
```
generates the following outputs:

* metadata.txt

* hyperparameters.txt  
	Values of the cluster sizes and the affinity matrices learned.

* assignment.dat:  
	(i,q)-element indicates the cluster assignments of vertex `i` with the input number of clusters `q`.

* FE_sbm.dat, cvBayesPrediction_sbm.dat, cvGibbsPrediction_sbm.dat, cvGibbsTraining_sbm.dat, cvMAP_sbm.dat:  
	(q,1)-element indicates the input number of clusters `q`, (q,2) and (q,3)-elements indicate the errors and standard errors with `q`, respectively.

OPTIONS
============

```
julia sbmBIX.jl -help
```
shows the options and more details.



REFERENCE
============
arxiv:xxxxxxx


============
Tatsuro Kawamoto: kawamoto.tatsuro@gmail.com

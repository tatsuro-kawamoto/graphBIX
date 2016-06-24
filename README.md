# graphBIX
Graph clustering by Bayesian inference with cross-validation model assessment.

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or above. See LICENSE.txt for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


* sbm.jl
	Bayesian inference of the stochastic block model using EM algorithm + belief propagation with the leave-one-out cross-validation.
* mod.jl
	Bayesian inference of the stochastic block model with a restricted affinity matrix (which corresponds to modularity maximization) using EM algorithm + belief propagation with the leave-one-out cross-validation.

USAGE
============
### sbm.jl
To start, the following package needs to be imported:
```
using DocOpt
```
For a given edgelist file, e.g. `edgelist.txt`,
```
julia sbm.jl edgelist.txt
```
generates the following outputs:

* summary.txt
* hyperparameters.txt  
	Values of the cluster sizes and the affinity matrices learned.
* assignment.dat:  
	(i,q)-element indicates the cluster assignments of vertex `i` with the input number of clusters `q`.
* BetheFreeEnergy_sbm.dat, cvBayesPrediction_sbm.dat, cvGibbsPrediction_sbm.dat, cvGibbsTraining_sbm.dat, cvMAP_sbm.dat:  
	(q,1)-element indicates the input number of clusters `q`, (q,2) and (q,3)-elements indicate the errors and standard errors with `q`, respectively.

### mod.jl
To start, the following package needs to be imported:
```
using DocOpt
using PyPlot
```
```
julia mod.jl edgelist.txt
```
generates the following outputs:

* Summary of model assessments (summary.txt)
* Cluster assignments (assignment.txt)
* Detailed results of model assessments (assessment.txt)
* Plot of model assessments (plot_model_assessment.pdf)
* [optional] `.smap` files for the alluvial diagram

OPTIONS
============

```
julia sbm.jl -help
```
shows the options and more details.



REFERENCE
============
Tatsuro Kawamoto and Yoshiyuki Kabashima, arXiv:1605.07915 (2016).


============  
Author: Tatsuro Kawamoto: kawamoto.tatsuro@gmail.com

# graphBIX
Graph clustering by Bayesian inference with cross-validation model assessment.

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or above. See LICENSE.txt for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


* sbm.jl
	Bayesian inference of the stochastic block model (with full degrees of freedom) using EM algorithm + belief propagation with the leave-one-out cross-validation.
* mod.jl
	Bayesian inference of the stochastic block model restricted to community structure using EM algorithm + belief propagation with the leave-one-out cross-validation.

USAGE
============
### sbm.jl, mod.jl
To start, the following package needs to be imported:
```
using DocOpt
using PyPlot
```
For a given edgelist file, e.g. `edgelist.txt`,
```
julia sbm.jl edgelist.txt
```
generates the following outputs:

* Summary of model assessments (summary.txt)  
	Input parameters / actual number of clusters & the number of iteration until convergence for each `q`.
* Detailed results of model assessments (assessment.txt):  
	Values of the cluster sizes and the affinity matrices learned.
* Cluster assignments (assignment.txt):  
	(i,q)-element indicates the cluster assignments of vertex `i` with the input number of clusters `q`.
* Plot of model assessments (assessment_"dataset".pdf)
* [optional] `.smap` files for the alluvial diagram


OPTIONS
============

```
julia sbm.jl -help
```
shows the options and more details.



REFERENCE
============
sbm.jl: Tatsuro Kawamoto and Yoshiyuki Kabashima, "Cross-validation estimate of the number of clusters in a network", Scientific Reports, 7, 3327 (2017).

mod.jl: Tatsuro Kawamoto and Yoshiyuki Kabashima, "Comparative analysis on the selection of number of clusters in community detection", Phys. Rev. E 97, 022315 (2018).

labeled_sbm: Tatsuro Kawamoto, "Algorithmic detectability threshold of the stochastic block model", Phys. Rev. E 97, 032301 (2018).

============  
Author: Tatsuro Kawamoto: kawamoto.tatsuro@gmail.com

# graphBIX
Graph clustering by Bayesian inference with cross-validation model assessment implemented in Julia.

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or above. See LICENSE.txt for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


* sbmBIX.jl
	Bayesian inference of the standard stochastic block model using EM algorithm + belief propagation with the leave-one-out cross-validation. 
+ dcsbmBIX.jl
	Bayesian inference of the degree-corrected stochastic block model using EM algorithm + belief propagation with the leave-one-out cross-validation. 

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

* assignment.dat

* FE_sbm.dat

* cvBayesPrediction_sbm.dat

* cvGibbsPrediction_sbm.dat

* cvGibbsTraining_sbm.dat

* cvMAP_sbm.dat


OPTIONS
============

```
julia sbmBIX.jl -help
```
shows the options: 
```
Usage:
  sbmBIX.jl [-h] <filename> [--Bmax=<Bmax>] [--init=partition...] [--samples=<samples>]
  sbmBIX.jl -h | --help
  sbmBIX.jl --version
  

Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --Bmax=<Bmax>             Maximum number of clusters. [default: 6]
  --init=partition...       Initial partition. [default: normalizedLaplacian]
  --samples=<samples>       Number of samples for each initial partition. [default: 10]
```

Note that the result varies depending on the initial values of the hyperparameters (fractions of clusters & affinity matrix). 
To be cautious, try multiple `--init` and many `samples`. 

To select the initial values of the hyperparameters, specify `--init`. 
The options for `--init` are 
- normalizedLaplacian: Spectral clustering with k-means algorithm. 
- random: Equal size clusters & randomly polarized affinity matrix. 
- uniformAssortative: Equal size clusters & equal size assortative block structure. 
- uniformDisassortative: Equal size clusters & equal size disassortative block structure. 

Example: 
```
julia sbmBIX.jl edgelist.txt --Bmax=10 --init={normalizedLaplacian,random} --samples=5
```



REFERENCE
============
arxiv:xxxxxxx


============
Tatsuro Kawamoto: kawamoto.tatsuro@gmail.com
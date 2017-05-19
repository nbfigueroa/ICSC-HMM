
===============================================================================
Matlab Software for:
* HDP-HMM (hierarchical Dirichlet process hidden Markov model)
* HDP-AR-HMM (hierarchical Dirichlet process autoregressive hidden Markov model)
* HDP-SLDS (hierarchical Dirichlet process switching linear dynamical system)
===============================================================================

Copyright (C) 2009, Emily B. Fox and Erik B. Sudderth.
(ebfox[at]alum[dot]mit[dot]edu and sudderth[at]cs[dot]brown[dot]edu)

This software package includes several Matlab scripts and
auxiliary functions, which implement Gibbs sampling algorithms 
for the models described in the following publications:
  An HDP-HMM for Systems with State Persistence
  E. B. Fox, E. B. Sudderth, M. I. Jordan, and A. S. Willsky
  Proc. Int. Conf. on Machine Learning, July, 2008.
Please cite this paper in any publications using the HDP-HMM package.
  Nonparametric Bayesian Learning of Switching Dynamical Systems
  E. B. Fox, E. B. Sudderth, M. I. Jordan, and A. S. Willsky
  Advances in Neural Information Processing Systems, vol. 21, pg. 457-464, 2009.
Please cite this paper in any publications using the HDP-AR-HMM or HDP-SLDS package.

See also:
  The Sticky HDP-HMM: Bayesian Nonparametric Hidden Markov Models with Persistent States
  E. B. Fox, E. B. Sudderth, M. I. Jordan, and A. S. Willsky
  MIT LIDS TR #2777, November, 2007.

  Bayesian Nonparametric Learning of Complex Dynamical Phenomena
  E. B. Fox
  Ph.D. Thesis, July, 2009.
	

HDP-HMM/HDP-AR-HMM/HDP-SLDS supports time series analysis with the following model families:
* HMM with single or DP mixture of Gaussian emissions with unknown mean and covariance.
	- conjugate NIW or non-conjugate N-IW prior
* HMM with multinomial emissions.
	- conjugate Dirichlet prior.
* AR-HMM of fixed order r and zero-mean noise
	- conjugate MNIW prior
* AR-HMM of unknown order, but maximal order r and zero-mean noise
	- non-conjugate sparsity inducing ARD prior
* AR-HMM of fixed order r and non-zero-mean noise
	- non-conjugate MNIW-N prior (MNIW prior on A,\Sigma and N prior on mu), or N-IW-N prior (N prior on A, IW on \Sigma, N on mu), or Afixed-IW-N (A is fixed by user, IW prior on \Sigma, and N prior on mu)
* SLDS with fixed state dimension d and zero mean noises
	- conjugate MNIW prior on A,\Sigma, and IW prior on measurement noise R
* SLDS with unknown state dimension, but maximal dimension d and zero mean noises
	- non-conjugate ARD prior on A,\Sigma, and IW prior on measurement noise R
* SLDS with other combinations of non-zero-mean noises and prior settings, though not thoroughly tested.

========================================================================
Package Organization and Documentation
========================================================================

Summary of HDP-HMM/HDP-AR-HMM/HDP-SLDS package contents:

HDPHMMDPinference.m:
  Main inference script.
/utilities:  
  Script runstuff.m with example inputs to main inference script, along
  with various other functions used to create necessary structures, etc.
/relabeler:  
  Code to perform optimal mapping between true and estimated mode sequences.

========================================================================
Setup and Usage Examples
========================================================================

For an example of sparse feature extraction, see runstuff.m.
To use the HDP-HMM/HDP-AR-HMM/HDP-SLDS code, you must first take two steps:
1) Install Minka's lightspeed toolbox and add directory to path:
	http://research.microsoft.com/~minka/software/lightspeed/
2) Add /relabeler and /utilities directory to path

========================================================================
Acknowledgments
========================================================================

Portions of the package were adapted from Yee Whye Teh's
"Nonparametric Bayesian Mixture Models" package, release 1.
Available from:  http://www.gatsby.ucl.ac.uk/~ywteh

========================================================================
Copyright & License
========================================================================

Copyright (C) 2009, Emily B. Fox and Erik B. Sudderth.

http://web.mit.edu/ebfox/www/

Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education, provided this copyright notice is retained, and note is
made of any changes that have been made.

These programs and documents are distributed without any warranty,
express or implied.  As the programs were written for research
purposes only, they have not been tested to the degree that would be
advisable in any important application.  All use of these programs is
entirely at the user's own risk.


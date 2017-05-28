# ICSC-HMM
ICSC-HMM : IBP Coupled SPCM-CRP Hidden Markov Model for Transform-Invariant Time-Series Segmentation  
Website: https://github.com/nbfigueroa/ICSC-HMM  
Author: Nadia Figueroa (nadia.figueroafernandez AT epfl.ch) 

This is a toolbox for inference of the ICSC-HMM (IBP Coupled SPCM-CRP Hidden Markov Model) [1]. The ICSC-HMM is a segmentation and action recognition algorithm that solves for three challenges in HMM-based segmentation: 

**(1) Unknown cardinality:** The typical model selection problem, number of hidden states is unknown. This can be solved by formulating an HMM with the Bayesian Non-Parametric treatment, i.e. placing an infinite prior on the transition distributions.  
**(2) Fixed dynamics:** For BNP analysis of ***multiple*** time-series with the HDP prior, the time series are tied together with the same set of transition and emission parameters.  
**(3) Transform-invariance:** For ***any*** type of HMM, the emission models are unique, there is no way to handle transformations within or throughout time-series.  

This is done by coupling the IBP-HMM which solves for challenges **(1-2)** with the SPCM-CRP mixture model for Covariance matrices which addresses challenge **(3)**. The underlying IBP-HMM code was forked from [NPBayesHMM](https://github.com/michaelchughes/NPBayesHMM) and modified accordingly. 

#### Reference
[1] [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) and Aude Billard, "Transform-Invariant Non-Parametric Clustering of Covariance Matrices and its Application to Unsupervised Joint Segmentation and Action Discovery." *In preparation for Pattern Recognition*. 

---

### Dependencies
- [LightSpeed Matlab Toolbox](https://github.com/tminka/lightspeed): Tom Minka's library which includes highly optimized versions of mathematical functions.
- [Eigen3 C++ Matrix Library](http://eigen.tuxfamily.org/index.php?title=Main_Page): C++ Linear Algebra Library.
- [SPCM-CRP](https://github.com/nbfigueroa/SPCM-CRP.git): Transform Invariant Chinese Restaurant Process Mixture Model for Covariance Matrices

---
### Installation
Before trying out anything, you must first compile the MEX functions for fast HMM dynamic programming, to do so run the following:
```
~/ICSC-HMM/CompileMEX.sh
```

You're ready! Now run demos..

---
### Illustrative Example

...

---


### Run Demo

...

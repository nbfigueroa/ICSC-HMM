# ICSC-HMM
ICSC-HMM : IBP Coupled SPCM-CRP Hidden Markov Model for Transform-Invariant Time-Series Segmentation  
Website: https://github.com/nbfigueroa/ICSC-HMM  
Author: Nadia Figueroa (nadia.figueroafernandez AT epfl.ch) 

This is a toolbox for inference of the ICSC-HMM (IBP Coupled SPCM-CRP Hidden Markov Model) [1]. The ICSC-HMM is a segmentation and action recognition algorithm that solves for three challenges in HMM-based segmentation and action recognition: 

**(1) Unknown cardinality:** The typical model selection problem, number of hidden states is unknown. This can be solved by formulating an HMM with the Bayesian Non-Parametric treatment. This is done by placing an infinite prior on the transition distributions, typically the Hierarchical Dirichlet Process (HDP).  
**(2) Fixed dynamics:** For BNP analysis of ***multiple*** time-series with the HDP prior, the time series are tied together with the same set of transition and emission parameters.  This problem was alleviated by the Indian Buffet Process (IBP) prior, which relaxes the assumption of the multiple time-series following the same transition parameters and allowing for only a sub-set of states to be active.  
**(3) Transform-invariance:** For ***any*** type of HMM, the emission models are always assumed to be unique, there is no way to handle transformations within or throughout time-series.   

We tackle these challenges by coupling the IBP-HMM which solves for challenges **(1-2)** with the SPCM-CRP mixture model for Covariance matrices which addresses challenge **(3)**. The underlying IBP-HMM code was forked from [NPBayesHMM](https://github.com/michaelchughes/NPBayesHMM) and modified accordingly. 

#### Reference
[1] [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) and Aude Billard, "Transform-Invariant Non-Parametric Clustering of Covariance Matrices and its Application to Unsupervised Joint Segmentation and Action Discovery." *Submitted to Journal of Machine Learning Research (JMLR)* [Under Review](https://arxiv.org/abs/1710.10060). 

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
### Motivating Example
***In which applications would someone need to tackle the three challenges listed above?*** 
Imagine a set of time-series, whose variables correspond to motion or interaction signals, such as position, velocity, orientation, forces and torques of an end-effector or a hand; representing a human (or robot) executing a complex sequence of actions in an LfD setting. These time-series might be subject to ***transformations***, as shown in the following figure: 
<p align="center">
<img src="https://github.com/nbfigueroa/ICSC-HMM/blob/master/figs/img/segmented_trajectories_light-1.png" width="500">
</p>
In this illustration, the point-mass is a simplification of an end-effector or hand. We have three time-series representing trajectories of the point-mass composed of a sequence of **common** actions: (i) approaching towards a surface, (ii) sliding on a surface and (iii) flying away. Given no **prior** knowledge on the number of actions present in the trajectories or what their sequencing might be, we would like to decompose them and discover the underlying transform-invariant actions (right). This translates to developing an unsupervised **joint segmentation and action discovery** framework; capable of decomposing the time-series into sequences of **transform-invariant** actions.

### Illustrative Example and Demos
In other words, assume we are given a set of 'M' 2D time-series with varying length $T= \{T^{(1)}, \cdots, T^{(M)}\}$  and switching dynamics $\pi= \{\pi^{(1)}, \cdots, \pi^{(M)}\}$, sampled from 2 \textbf{unique} Gaussian emission models  $\theta_1,\theta_2$ subject to transformations $f_1(\cdot),f_2(\cdot)$ resulting in a set of transform-dependent emission models  $\Theta = \{\theta_1,\theta_2,\theta_3 = f_1(\theta_2),\theta_4 = f_2(\theta_3)\}$,
<p align="center">
<img src="https://github.com/nbfigueroa/ICSC-HMM/blob/master/figs/img/transform_timeseries-1.png" width="425"><img src="https://github.com/nbfigueroa/ICSC-HMM/blob/master/figs/img/transform_emissionmodels-1.png" width="425">
</p>

### Run Demos
```
*** Hidden Markov Model Results*** 
 Optimal States: 4 
 Hamming-Distance: 0.385 (0.016) GCE: 0.026 (0.082) VO: 1.025 (0.229) 
 Purity: 0.977 (0.074) NMI: 0.679 (0.109) F: 0.752 (0.041)  

*** Sticky HDP-HMM Results*** 
 Optimal States: 4.000 (0.000) 
 Hamming-Distance: 0.379 (0.000) GCE: 0.000 (0.001) VO: 0.956 (0.006) 
 Purity: 1.000 (0.000) NMI: 0.713 (0.002) F: 0.765 (0.000)  

*** IBP-HMM Results*** 
 Optimal Feature States: 4.000 (0.000) 
 Hamming-Distance: 0.379 (0.000) GCE: 0.000 (0.000) VO: 0.953 (0.000) 
 Purity: 1.000 (0.000) NMI: 0.714 (0.000) F: 0.765 (0.000)  

 *** IBP-HMM + SPCM-CRP Results***
 Optimal Transform-Dependent States: 4.000 (0.000)  
 Hamming-Distance: 0.379 (0.000) GCE: 0.000 (0.000) VO: 0.953 (0.000) 
 Optimal Transform-Invariant States: 2.000 (0.000) 
 Hamming-Distance: 0.000 (0.000) GCE: 0.000 (0.000) VO: 0.000 (0.000) 
 Purity: 1.000 (0.000) NMI: 1.000 (0.000) F: 1.000 (0.000) 

 *** ICSC-HMM Results*** 
 Optimal Transform-Dependent States: 4.100 (0.316)  
 Hamming-Distance: 0.379 (0.000) GCE: 0.000 (0.000) VO: 0.953 (0.000) 
 Optimal Transform-Invariant States: 2.000 (0.000) 
 Hamming-Distance: 0.000 (0.000) GCE: 0.000 (0.000) VO: 0.000 (0.000) 
 Purity: 1.000 (0.000) NMI: 1.000 (0.000) F: 1.000 (0.000) 

---
```


### 3rd Party
- Classical HMM-EM implementation: ...
- Sampler for sticky HDP-HMM: ...
- Sampler for ibp-HMM: ...

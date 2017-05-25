# ICSC-HMM

Toolbox for inference of the ICSC-HMM (IBP Coupled SPCM-CRP Hidden Markov Mode). The underlying IBP-HMM code was forked from [NPBayesHMM](https://github.com/michaelchughes/NPBayesHMM), coupling of the SPCM-CRP and other modification are implemented on top of this fork.

---

### Dependencies
- [LightSpeed Matlab Toolbox](https://github.com/tminka/lightspeed): Tom Minka's library which includes highly optimized versions of mathematical functions.
- [Eigen3 C++ Matrix Library](http://eigen.tuxfamily.org/index.php?title=Main_Page): C++ Linear Algebra Library.
- [SPCM-CRP](https://github.com/nbfigueroa/SPCM-CRP.git): Transform Invariant Chinese Restaurant Process Mixture Model for Covariance Matrices
- [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox): Machine learning toolbox containing a plethora of dimensionality reduction, clustering, classification and regression algorithms accompanying the [Advanced Machine Learning](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/index.php) course imparted at EPFL by Prof. Aude Billard.

#### Installation
Before trying out anything, you must first compile the MEX functions for fast HMM dynamic programming, to do so run the following:
```
~/ICSC-HMM/CompileMEX.sh
```

You're ready! Now run demos..

---
### Run Demo

...

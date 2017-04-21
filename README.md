# tGau-BP-HMM

Toolbox for inference on heterogeneous sequential data. The underlying BP-HMM code was forked from [NPBayesHMM](http://michaelchughes.github.com/NPBayesHMM/), modifications for the transform Invariant Gaussian (tGau) formulation were implemented on top of this.

####Install required libraries
| Dependencies |
| ------------- |
|[Eigen3 C++ Matrix Library](http://eigen.tuxfamily.org/index.php?title=Main_Page)|
|[Lightspeed for Matlab](http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/)|
|[SPCM-CRP](https://github.com/nbfigueroa/SPCM-CRP.git)|

####Compilation
For systems other than 64bit Linux distro, you must compile MEX functions for fast HMM dynamic programming: ```~/tIG-BP-HMM/CompileMEX.sh```


####Configure Toolbox

- Create file a ```results``` folder. 
- Run: ``` ~/tGau-BP-HMM/ConfigToolbox.sh```

You're ready! Now run demos..

---
###Run Demo
...

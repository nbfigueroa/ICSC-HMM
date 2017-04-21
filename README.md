# ICSC-HMM

Toolbox for inference of the ICSC-HMM (IBP Coupled SPCM-CRP Hidden Markov Mode). The underlying IBP-HMM code was forked from [NPBayesHMM](http://michaelchughes.github.com/NPBayesHMM/), coupling of the SPCM-CRP is on top of this code.

#### Install required libraries
| Dependencies |
| ------------- |
|[Eigen3 C++ Matrix Library](http://eigen.tuxfamily.org/index.php?title=Main_Page)|
|[Lightspeed for Matlab](http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/)|
|[SPCM-CRP](https://github.com/nbfigueroa/SPCM-CRP.git)|

#### Compilation
For systems other than 64bit Linux distro, you must compile MEX functions for fast HMM dynamic programming: ```~/ICSC-HMM/CompileMEX.sh```

#### Configure Toolbox

- Create file a ```results``` folder. 
- Run: ``` ~/ICSC-HMM/ConfigToolbox.sh```

You're ready! Now run demos..

---
### Run Demo
...

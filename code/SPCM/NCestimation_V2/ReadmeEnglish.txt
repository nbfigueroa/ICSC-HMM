Help file of Toolbox for estimating the number of clusters (NCT)

 (Version 2.0)

Your comments are welcome at:
http://www.mathworks.com/matlabcentral/fileexchange/13916
E-mail: wangkjun@yahoo.com

(1) Contents of NCT
    The NCT includes 4 External validity indices and 8 internal validity indices, and the sub-routine "validity_Index.m" is designed to use them. 
    This tool is suitable for the research work such as the performance comparison of different indices on estimation of the number of clusters, algorithm design by improving part codes of this toolbox. 

i)  External validity indices when true class labels are known:
    Rand index
    Adjusted Rand index
    Mirkin index
    Hubert index
ii) Internal validity indices when true class labels are unknown:
    Silhouette
    Davies-Bouldin
    Calinski-Harabasz
    Krzanowski-Lai
    Hartigan
    weighted inter- to intra-cluster ratio
    Homogeneity
    Separation
iii) Others
    Error rate (compared with true labels)
    System Evolution: it is used to estimate the number of clusters and give separable degrees between clusters.

Note 1: The codes of Rand, Adjusted Rand, Mirkin, Hubert indices are from David Corney (D.Corney@cs.ucl.ac.uk), who holds the copyright.

Note 2: Statistics Toolbox of Matlab needs to be installed, since it contains routines such as K-means and Silhouette index.

Note 3: Error rate: The error rate might be inaccurate if the clustering solution under true NC has error rate >20%, since "valid_errorate" designed here can not deal with complex cases.

(2) Contents of main file "mainClusterValidationNC.m" 
    It is designed to use validity indices to estimate the number of clusters (NC) for PAM and K-means clustering algorithms. 
Part 1: Selection of a data set, initialization and computation of distance/dissimilarity matrix. 
Part 2: A clustering algorithm Runs (N-1 times) to yield k clusters (k=2,3,...,N). 
Part 3: Cluster validation for Estimating the number of clusters (NC). "validity_Index"

Note1: The programs will stop when the elements (or data points) are less than 4.
Note2: The programs are tested under Matlab 6.5 and 7.2.

(3) PAM & K-means clustering algorithms included in this program
  The K-means codes are from Mathworks. The initialization of K-means is to select K centroids from data at random, for other choices refer to the kmeans.m of Matlab (inner function of Matlab).

  The PAM (partitioning around medoids) is a robust clustering algorithm to minimize a sum of dissimilarities of data points to their closest medoids, and tends to be more robust than K-means, or a robust ¡°version¡± of K-means. PAM needs pre-assigned NC as input parameter, similar to K-means. It seems not suitable to large data sets, and might run slow for a data set with number of data points over such as 2000.

  The programs of PAM have been included in the Matlab library LIBRA (http://wis.kuleuven.be/stat/robust/LIBRA.html), the statistic analysis software S-plus (http://www.splus.com/) and the cluster package of R (http://cran.r-project.org/). The PAM codes in this program are from LIBRA.

(4) Pearson similarity/distance
  Pearson similarity/distance is the linear correlation coefficient between two vectors and has its value range from -1 to 1, and it is commonly used to measure the similarity/distances between genes. 
  For the correct computation of indices, in this program the correlation coefficient is normalized to [0,1] by R(i,j)=(1-R(i,j))/2 as distances, where 0 is the closest distance and 1 the farthest one, and it is easy to convert it back by 1-2R(i,j). 
  For example, assume that there be two genes g1 and g2, then R(g1,g2)=1 means that their distance is the farthest, and R(g1,g1)=0 means that g1 itself has the closest distance. 

(5) Input: a data file like "yourdata.txt"
  The input data file is the tab delimited text file with numeric tabular data or similar Matlab file format (e.g. rows denote data points/elements and columns denote dimensions),  and all the data should be numeric values and without missing values.

  If you use Euclidean distance, please put the data file before "case 21". If true class labels are known and in 1st column, put the data file before "case 11", otherwise in "case 11". 
  If you use Pearson distance, please put the data file after "case 20". If true class labels are known and in 1st column, put the data file between "case 21" and "case 40", otherwise after "case 40". 
 
(6) Output
   The PAM/K-means is first used to divide a data set into k clusters (k=1,2,3,¡­,N), resulting in N clustering solutions; and then the validity indices/methods estimate the optimal NC ko based on these solutions with seeking limit N=ko+6. The found ko indicated by a square symbol is shown in the figures.
   When a cluster has few elements (e.g.<4), the PAM/K-means will not go on (see rows in the clustering part)

(7) Demo data set (true class labels in 1st column of the data file)
Dataset             #class   #elements    dimension
leuk72_3k	3	72	39

---------------------------------------------------------------------------------------------------
Copyright (C) 2006-2007.
Last modified: July 1, 2009
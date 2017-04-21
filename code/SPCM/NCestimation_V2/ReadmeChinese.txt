估计聚类个数工具箱(NCT) （版本2.0） 帮助文件

    欢迎使用和评述此工具箱，您的意见是对我们工作的支持。E-mail: wangkjun@yahoo.com
下载地址: http://www.mathworks.com/matlabcentral/fileexchange/13916
    此工具适合于不同有效性指标的性能比较，改进代码用于不同的应用问题等等。

(1) NCT的内容
    NCT包括4个外部有效性指标和8种内部有效性指标，编制的程序文件"validity_Index.m"用于调用它们。
i)  外部有效性指标（当正确的类标已知时）
    Rand index
    Adjusted Rand index
    Mirkin index
    Hubert index
ii) 内部效性指标（当正确的类标未知时）
    Silhouette
    Davies-Bouldin
    Calinski-Harabasz
    Krzanowski-Lai
    Hartigan
    weighted inter- to intra-cluster ratio
    Homogeneity
    Separation
iii) 其它
    Error rate 错误率 (与正确的类标比较)
    System Evolution 估计类数与分析类间可分性

注1: Rand, Adjusted Rand, Mirkin, Hubert指标程序来源于 David Corney (程序作者拥有版权，D.Corney@cs.ucl.ac.uk)。

注2:  Matlab 的统计工具箱（Statistics Toolbox）需安装，它包含要用到的K-means聚类算法和Silhouette指标。

注3: Error rate（错误率）: 此错误率指标可能不准确（特别是聚类结果的错误率>20%,） 因为程序"valid_errorate"比较简单，不能处理复杂的情况。

注4: 若对聚类稳定性指标/方法感兴趣，涉及此程序的文章为
Smolkin, M. and Ghosh, D. (2003). Cluster stability scores for microarray data in cancer studies. BMC Bioinformatics 4, 36 - 42.。其文章与源程序可下载：http://www.sph.umich.edu/~ghoshd/COMPBIO/CSS/

(2) 主文件 "mainClusterValidationNC.m" 的内容
      主文件设计为如何使用PAM聚类算法、如何使用有效性指标和方法来估计聚类个数。
Part 1（第一部分）: 选择数据集、初始化和计算距离或不相似度矩阵。
Part 2: 运行聚类算法，产生个k聚类的聚类结果(k=2,3,...,N)。
Part 3: 聚类有效性评价和估计聚类个数(NC). "validity_Index"

注1: 当一个聚类包含的数据样本少于4个时，聚类算法将不再继续运行，参见Part 2 的程序。
注2: 程序已在 Matlab 6.5 和7.2下测试过。

(3) K-means与PAM聚类算法
    K均值聚类算法的程序是Matlab的内部函数。在这采用的是随机初始化，其它选择参见关于kmeans.m的Matlab帮助文件。
    PAM（Partitioning around medoid围绕中心点的划分）聚类算法与K均值聚类算法的工作原理有些类似，在聚类的过程中最小化各样本到其最近的代表样本的不相似度之和。PAM算法也需要预先给定类数才能工作，但PAM 算法对噪声不敏感且不受数据输入顺序的影响。因此，对PAM聚类的结果进行有效性分析和确定聚类个数是使用PAM算法进行聚类分析的必要步骤。 PAM算法似乎对大数据集不适合，例如若数据个数大于2000时它运行缓慢。
    以下这些软件包含PAM算法：统计软件S-plus (http://www.splus.com/)、Matlab库函数 LIBRA (http://wis.kuleuven.be/stat/robust/LIBRA.html)和R的聚类程序包 (http://cran.r-project.org/)。本工具箱的PAM程序来自于LIBRA（程序作者拥有版权）。

(4)  Pearson相似度和距离
    基因表达数据聚类分析中基因之间的相似性测度定义为Pearson相关系数，即两个样本/向量i和j之间的线性相关系数R(i,j)，其取值范围[-1,1]。
    为了便于各指标的计算，将R(i,j)进行这样的转换：R(i,j)=0.5-0.5R(i,j)，从而R(i,j)取值范围[0,1]。这样R(i,j)表示正的Pearson距离和不相似度，而相似度则为1-R(i,j)。 例如最远Pearson距离的两个基因 g1和 g2有R(g1,g2)=1 以及自身最近R(g1,g1)=0 。

(5) 输入：数据文件 "yourdata.txt"
    文件的每一行表示一个数据样本或基因，列表示维数，数据应为数值并无缺失数据。若类标已知，放类标于第一列；若类标未知，第一列则放数据。当使用欧式距离时，数据文件放于"case 21"之前；若类标已知，则放于"case 11"之前。当使用Pearson距离时，数据文件放于"case 20"之后；若类标已知，则放于"case 40"之前。
 
(6) 输出
    首先使用聚类算法对数据进行聚类，即将数据集划分为k个聚类（k=1,2,3,…,N），然后运用有效性评价指标和方法对这N个聚类结果进行评价和估计聚类个数。各种方法估计出的聚类个数在输出的图上由方形符号指明。

(7) 示例数据集 (数据文件的第一列为正确的类标，其余列为数据)
数据集             类数   样本数目    维数
leuk72_3k	3	72	39

---------------------------------------------------------------------------------------------------
Copyright (C) 2006-2007.
最后修改: 2009.7.1
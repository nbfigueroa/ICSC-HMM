% Function for finding basis of the intersection of N subspaces
% [out,d] = findIntersection(varargin)
%
% INPUT:    N matrices/vectors - matrices/vectors defining basis of the
%                                subspace (bases must be row vectors (!!!)
%                                and of the same dimension)
% OUTPUT:   intersection       - basis of the intersection space
%           dimension          - dimension of the intersection space
%-------------------------------------------------------------------------
% example: A = [1 0 1;  <-     B = [1 1 0];     C = [1 1 2;  <-
%               0 1 0;  <-                           0 0 1]; <-
%               0 2 3]; <-
%          -------------
%          [intersection,dimension] = findIntersect(A,B,C);
%          intersection = [1 1 0];
%          dimension    = 1;
%
% 04 Jul 2011   - created: Ondrej Sluciak <ondrej.sluciak@nt.tuwien.ac.at>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [intersection,dimension] = findIntersect(varargin)

N = nargin;

if (N<1)
    error('findIntersect:InvalidInput','Invalid input.');
end

if (N == 1)
    intersection = rref(varargin{1});
    return;
end

rnk1 = rank(varargin{1});

for i = 2:N
    rnk1 = rnk1 + rank(varargin{i});
    if (size(varargin{i},2) ~= size(varargin{i-1},2))
        error('findIntersect:InvalidInput','Sizes of the spaces must be equal.');
    end
end

rnk2      = rank([repmat(varargin{1},1,N-1); blkdiag(varargin{2:end})]); % formula by Yongge Tian: www.math-cs.ucmo.edu/~mjms/2002.2/ytian2.ps
dimension = rnk1 - rnk2;                                                 % dimension of the intersection

if (dimension < 1)
    intersection = zeros(1,size(varargin{i},2));
    return;
end

tmp = null([varargin{1}',varargin{2}'],'r');
tmp = rref(tmp(end-size(varargin{2},1)+1:end,:)'*varargin{2});

for i = 3:N
    tmp = null([tmp',varargin{i}'],'r');
    tmp = rref(tmp(end-size(varargin{i},1)+1:end,:)'*varargin{i});
end

intersection = tmp(1:dimension,:);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function for plotting planes in 3D
% plotPlanes(varargin)
%
% INPUT: N matrices/vectors - defining: 
%                             1. directional vector(s)
%                             2. normal vector              keyword 'Normal' must precede the normal vector! 
%                             3. plane shift    (optional), keyword 'd' must precede the vector defining the shift!
%                             4. colormap       (optional), keyword 'ColorMap' must preced the colormap!
% OUTPUT: --
%
% example: 
%           A = [1 1 0;  Ashift = [1 1 1];  B = [1 1 1]; CNormal = [1 1]
%                0 0 1];
%
%           plotPlanes(A,B)
%           plotPlanes(A,'Normal',CNormal)
%           plotPlanes(A,B,'Normal',CNormal)
%           plotPlanes(A,B,'Normal',CNormal,'d',Ashift)
%           plotPlanes(A,'d',Ashift,B,'Normal',CNormal,'ColorMap','gray')
%           
% 04 Jul 2011   - created: Ondrej Sluciak <ondrej.sluciak@nt.tuwien.ac.at>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotPlanes(varargin)

N = nargin;
if (iscell(varargin) && (N==1))
    N = size(varargin{1},2);
    varargin = varargin{1};
end

%%%%%%%%
% rough determining the size of the plotted area
mx = 0;
for i = 1:N
    a = varargin{i};
    if (~ischar(a) && (ismatrix(a) || isvector(a) ))
        mx = mx + max(abs(a));
    end
end

[s,t] = meshgrid(linspace(-mx(1),mx(1),50),linspace(-mx(1),mx(1),50));
%%%%%%%%

i = 1;
while i <= N
    
    a = varargin{i};

    if ischar(a)
       
        if strcmp(a,'ColorMap')
            colormap(varargin{i+1});
            i = i + 2;
            continue;
        else
            if strcmp(a,'Normal')   % normal vector
                a = varargin{i+1};
                if isvector(a)          
                   if (length(a)>3)
                       warning('plotPlanes:IncorrectInput','Too long vector.');
                       i = i + 1;
                       continue;
                    end

                    if (length(a)==2)
                        a(3) = 0;
                    end
                    if (size(a,1)>size(a,2))
                        a = a';    % input must be row vector
                    end

                    b = null(a,'r')';
                    
                    bDist = false;
                    dist = zeros(1,3);
                    
                    if (i<=(N-2))
                        if ischar(varargin{i+2})
                            if strcmp('d',varargin{i+2})
                                dist = varargin{i+3};
                                bDist = true;
                            end
                        end
                    end

                    x = b(1,1)*t + b(2,1)*s + dist(1);
                    y = b(1,2)*t + b(2,2)*s + dist(2);
                    z = b(1,3)*t + b(2,3)*s + dist(3);

                    surf(x,y,z,'LineStyle','none');
                    hold on;
                    if bDist
                        i = i + 4;
                    else
                        i = i + 2;
                    end
                    continue;
                else
                    error('plotPlanes:IncorrectInput','Normal vector must be a vector.');
                end
            else
                error('plotPlanes:IncorrectInput','Unknown input parameter.');
            end
        end
    end
    
    if isscalar(a)
        warning('plotPlanes:IncorrectInput','Scalar input.');
        i = i + 1;
        continue;
    end

    if isvector(a)    %directional vector
       if (length(a)>3)
           warning('plotPlanes:IncorrectInput','Too long vector.');
           i = i + 1;
           continue;
        end
        
        if (length(a)==2)
            a(3) = 0;
        end
        
        line([-mx(1),mx(1)]*a(1),[-mx(1),mx(1)]*a(2),[-mx(1),mx(1)]*a(3),'LineWidth',2);
        
        hold on;
        i = i + 1;
        continue;
    end
    
    if ismatrix(a)
        bDist = false;
        dist = zeros(1,3);
        if (i<=(N-2))
            if (ischar(varargin{i+1}))
                if strcmp('d',varargin{i+1})
                    dist = varargin{i+2};
                    bDist = true;
                end
            end
        end
        sz = size(a);
        if (sz(2)>3)
            warning('plotPlanes:IncorrectInput','Input matrix is of bigger dimension than 3.');
            if (bDist)
                i = i + 3;
            else
                i = i + 1;
            end
            continue;
        else
            if (sz(2) == 2)
                a(:,3) = zeros(sz(1),1);
            end
            switch sz(1)
                case 2      % 2 directional vectors
                     x = a(1,1)*t + a(2,1)*s + dist(1);
                     y = a(1,2)*t + a(2,2)*s + dist(2);
                     z = a(1,3)*t + a(2,3)*s + dist(3);
                otherwise
                    b = rref(a);
                    switch rank(b)
                        case 1
                            warning('plotPlanes:IncorrectInput','Matrix is of rank 1.');
                            if (bDist)
                                i = i + 3;
                            else
                                i = i + 1;
                            end
                            continue;
                        case 2
                            x = b(1,1)*t + b(2,1)*s + dist(1);
                            y = b(1,2)*t + b(2,2)*s + dist(2);
                            z = b(1,3)*t + b(2,3)*s + dist(3);
                        otherwise
                            warning('plotPlanes:IncorrectInput','Matrix is of incorrect rank.');
                            if (bDist)
                                i = i + 3;
                            else
                                i = i + 1;
                            end
                            continue;
                    end
            end
        end
        surf(x,y,z,'LineStyle','none');  
        hold on;
        if (bDist)
            i = i + 3;
        else
            i = i + 1;
        end
        continue;
    end
    
end

xlabel('x');
ylabel('y');
zlabel('z');

hold off;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

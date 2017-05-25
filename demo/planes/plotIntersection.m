% Function for plotting intersection of planes/lines in 3D. 
% plotIntersection(varargin)
% Planes are defined by either 2 directional vectors, or a normal vector. 
% The function uses files: 'plotPlanes.m','findIntersect.m','ismatrix.m'
%
% INPUT: N matrices/vectors - defining: 
%                             1. directional vector(s)
%                             2. normal vector              keyword 'Normal' must precede the normal vector! 
%                             3. colormap       (optional), keyword 'ColorMap' must preced the colormap!
%
% OUTPUT: --
%
% example:
%           A = [1 1 0;  B = [1 1 1]; C = [1 1 0]
%                0 0 1];
%
%           plotIntersection(A,B)
%           plotIntersection(A,B,'Normal',CNormal,'ColorMap','gray')
%
% 04 Jul 2011   - created: Ondrej Sluciak <ondrej.sluciak@nt.tuwien.ac.at>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotIntersection(varargin)

colormap = [0.32 0.376 0.404];

plotPlanes(varargin);
hold on;

c = cell(1,nargin);
j = 1;
i = 1;
mx = 0;
while i <= nargin
    a = varargin{i};
    if ischar(a)
        if (strcmp(a,'Normal'))
            b = null(varargin{i+1},'r');
            c{j} = b';
            mx = mx + max(abs(b'));
            j=j+1;
            i=i+1;
        else
            error('plotIntersection:InvalidInput','Unknown input parameter');
        end
    else
        if (ismatrix(a) || (isvector(a) && ~ischar(a)))
             c{j} = a;
             mx = mx + max(abs(a));
             j=j+1;
        else
             error('plotIntersection:InvalidInput','Unknown input parameter');
        end
    end
    i=i+1;
end    

c = c(1:end-(nargin-j+1));

[intersection,d] = findIntersect(c{:});

switch d
    case 0
        h = plot3(intersection(1),intersection(2),intersection(3),'kx','LineWidth',3,'Color',colormap);
        legend(h,'Intersection');
    case 1
        if (size(intersection,2)==2)
            intersection(3) = 0;
        end
        h = line([-mx(1)*1.1,mx(1)*1.1]*intersection(1),[-mx(1)*1.1,mx(1)*1.1]*intersection(2),[-mx(1)*1.1,mx(1)*1.1]*intersection(3),'LineWidth',3,'Color',colormap);
        legend(h,'Intersection');
    case 2
        plotPlanes(intersection,'ColorMap',colormap);
        legend('Intersection');
    otherwise        
        error('Unable to draw intersection of the spaces.');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
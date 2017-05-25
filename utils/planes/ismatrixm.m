% Simple function for determining if the input is a matrix NxM
% INPUT:  matrix
% OUTPUT: true/false
% 04 Jul 2011   - created: Ondrej Sluciak <ondrej.sluciak@nt.tuwien.ac.at>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function b = ismatrix(a)

b = false;

if isempty(a)
    return;
end
if isscalar(a)
    return;
end
if isvector(a)
    return;
end
if iscell(a)
    return;
end
if isstruct(a)
    return;
end

s = size(a);

if (length(s) > 2)
    return;
end

if (s(1) > 1 && s(2) > 1)
    b = true;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

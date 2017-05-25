function T = find_patterns(M)
%FIND_PATTERNS finds all repeating patterns in a vector or number.
% T = FIND_PATTERNS(M) returns an n-by-2 cell array.  For each row of the
% cell array, column one has the pattern, and column two has the number of
% times the pattern occurs.
% Takes as input either a vector of digits or charcters, or a scalar number  
% and finds all repeating patterns of length greater one. 
% If the input argument is numeric, the output argument will be numeric,
% and if the input argument is a vector of characters, the output argument
% will be a vector of characters.
%
% Examples:
%
%  M = 123412123562356; % A scalar number.
%  T = find_patterns(M);
%  T{8,:}  % Show that [2 3 5 6] occurs 2 times.
%
%  M = 'c31a234121a23562c356'; % A character array.
%  T = find_patterns(M);
%
%  M = round(rand(1,300)*3); % A vector of digits.
%  T = find_patterns(M);
%  T{1,:}
%
%
% Author: Matt Fig
if ~isscalar(M) && ~isvector(M)
    error('Only scalar and vector arguments are allowed.');
end
flg = 0;
if ~ischar(M)
    M = sprintf('%.0f',M);
    flg = 1;
else
    M = M(:).'; % Make sure we have a row vector...
end
cnt = 1;
L = length(M);
T = {};
for jj = 1:L-2
    for ii = jj+1:L-1
        I = strfind(M,M(jj:ii));
        if length(I)>1
            T{cnt} = sprintf('%s %i',M(jj:ii),length(I));%#ok
            cnt = cnt + 1;
        end
    end
end
if ~isempty(T)
    T = regexp(unique(T),'\ ','split');
    T = cat(1,T{:});
      if flg
          T(:,1) = cellfun(@(x) str2num(x.').',T(:,1),'Un',0);%#ok
          T(:,2) = cellfun(@str2double,T(:,2),'Un',0);
      end
      [J,J] = sort(cellfun('length',T(:,1)));
      T = T(J,:);
end 
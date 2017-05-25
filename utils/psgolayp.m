function [ SG0 ] = psgolayp ( A, k, n )
%psgolay compute x, x', x" from data A with k-polynomial over 
%   the preceding 2*n+1 points.
%   Differs from standard sgolay in that it introduces NO delay
%   2*n+1 is the number of past points that are used in the filter
%   k is the order of the polynomial 
%
%   Example use:
%   T=0.05;t=(0:99)*T; s = sin(2*pi*t); s_noise = s + 0.5*rand(size(t));
%   plot(s_noise); hold on; d=psgolayp(s_noise,2,20);plot(d,'g'); hold off;


if round(n) ~= n, 
    error(generatemsgid('MustBeInteger'),'Number of data points must be an integer.'), end
if round(k) ~= k,
    error(generatemsgid('MustBeInteger'),'Polynomial order must be an integer.'), end
if k > n-1, error(generatemsgid('InvalidRange'),'It is silly to have a polynomial degree higher than # data.'), end

% compute the SG parameters for the last item.  Based on code from
% Diederick C. Niehorster <dcniehorster@hku.hk>.  Lots of magic ;-)
x = [-n:n]';
df = cumprod([ones(2*n+1,1) x*ones(1,k)],2) \ eye(2*n+1);
df = df.';

hx = 1;
for i=1:k
    hx = [hx [n].^i];
end

% these are the SG filter coefficients
fc = df*hx';

% now do the smoothing!
for i = 2*n+1:length(A),  
    SG0(i) =   dot(fc, A(i-2*n:i));
end

end

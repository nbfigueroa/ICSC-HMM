function Ob = observabilityMatrix(A,C)

Ob = C;
for i=1:(length(A)-1)
    Ob = [Ob ; C*A^i];
end
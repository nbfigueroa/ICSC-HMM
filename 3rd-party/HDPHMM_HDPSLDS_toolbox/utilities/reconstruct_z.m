function reconstructed_z = reconstruct_z(z,blockSize,T)

blockEnd = cumsum(blockSize);

reconstructed_z = zeros(1,T);
reconstructed_z(1:blockEnd(1)) = z(1);
for i=2:length(z)
    reconstructed_z(blockEnd(i-1)+1:blockEnd(i))=z(i);
end
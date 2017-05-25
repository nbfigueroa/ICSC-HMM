function [spcm] = ComputeSPCMfunction(behavs_theta, use_log)

spcm = [];    
       
for i=1:length(behavs_theta)
  for j=1:length(behavs_theta)     
            
%         [Vi, Di] = eig(behavs_theta{i});
%         [Vj, Dj] = eig(behavs_theta{j});
%         
%         %Ensure eigenvalues are sorted in ascending order
%         [Vi, Di] = sortem(Vi,Di);
%         [Vj, Dj] = sortem(Vj,Dj);
%         
%         %Structural of Sprectral Polytope
%         Xi = Vi*Di^1/2;
%         Xj = Vj*Dj^1/2;
%                 
%         %Norms of Spectral Polytope Vectors
%         for k=1:length(Dj)
%             eig_i(k,1) = norm(Xi(:,k));
%             eig_j(k,1) = norm(Xj(:,k));
%         end
%         
%         %Homothetic factors
%         hom_fact_ij = eig_i./eig_j;
%         hom_fact_ji = eig_j./eig_i;
%         
%         %Magnif
%         if (mean(hom_fact_ji) > mean(hom_fact_ij)) || (mean(hom_fact_ji) == mean(hom_fact_ij))
%             dir = 1;
%             hom_fact = hom_fact_ji;
%         else
%             dir = -1;
%             hom_fact = hom_fact_ij;
%         end     
        
        [hom_fact dir] = ComputeHomFactorPair(behavs_theta{i},behavs_theta{j});

        if use_log
            spcm(i,j,1) = log(std(hom_fact)); 
        else
            spcm(i,j,1) = std(hom_fact);
%             spcm(i,j,1) = 1/(1+std(hom_fact));
        end
        spcm(i,j,2) = mean(hom_fact); 
        spcm(i,j,3) = dir; 
        
   end
end
end
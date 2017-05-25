function [spcm] = ComputeSPCMfunctionProb(behavs_theta, tau)

spcm = [];    
       
for i=1:length(behavs_theta)
  for j=1:length(behavs_theta)                   
        
        [p_sim s hom_fact dir] = ComputeSPCMPair(behavs_theta{i},behavs_theta{j}, tau);

        spcm(i,j,1) = s;
        spcm(i,j,2) = p_sim;
        spcm(i,j,3) = hom_fact; 
        spcm(i,j,4) = dir; 
        
   end
end

end
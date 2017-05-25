function [p_sim spcm mean_fact dir] = ComputeSPCMPair(Sigma_i,Sigma_j,tau)
       
        dim = size(Sigma_i,1);
        
        [Vi, Di] = eig(Sigma_i);
        [Vj, Dj] = eig(Sigma_j);
        
        %Ensure eigenvalues are sorted in ascending order
        [Vi, Di] = sortem(Vi,Di);
        [Vj, Dj] = sortem(Vj,Dj);
        
        %Structural of Sprectral Polytope
        Xi = Vi*Di^1/2;
        Xj = Vj*Dj^1/2;
                
        %Norms of Spectral Polytope Vectors
        for k=1:length(Dj)
            eig_i(k,1) = norm(Xi(:,k));
            eig_j(k,1) = norm(Xj(:,k));
        end
        
        %Homothetic factors
        hom_fact_ij = eig_i./eig_j;
        hom_fact_ji = eig_j./eig_i;
        
        %Force Magnification and set directionality
        if (mean(hom_fact_ji) > mean(hom_fact_ij)) || (mean(hom_fact_ji) == mean(hom_fact_ij))
            dir = 1;
            hom_fact = hom_fact_ji;
        else
            dir = -1;
            hom_fact = hom_fact_ij;
        end  
        
        % Homothetic mean factor 
        mean_fact = mean(hom_fact);
        
        %Spectral similarity value
        spcm = std(hom_fact);
        
        % Scaling function
        alpha = 10^(tau*exp(-dim));

        % B-SPCM p(theta_i~theta_j|Sigma_i,Sigma_j) = 
        % 1/( 1 + s(Sigma_i,Sigma_j)*alpha(tau,dim))
        p_sim = 1/(1+spcm*alpha);
        
end
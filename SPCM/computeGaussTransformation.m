function [W2 b homo_ratio J] = computeGaussTransformation(Theta_i,Theta_j, hom_ratio, hom_dir)

    % Compute translation
    mu1 = Theta_i.mu;
    mu2 = Theta_j.mu;
    b = -mu2 + mu1;

    Sigma_a = Theta_i.invSigma^-1;
    Sigma_b = Theta_j.invSigma^-1;
    [Va Da] = eig(Sigma_a);
    [Vb Db] = eig(Sigma_b);
  

    if hom_dir < 0
        homo_ratio = hom_ratio;    
    else
        homo_ratio = 1/hom_ratio;
    end
    
    Sigma_b_sc = Vb*(Db*homo_ratio)*inv(Vb);


    % Find transformation R that minimizes objective function J
    conv_thres = 1e-5;
    thres = 1e-3;
    max_iter = 10000;
    iter = 1;
    J_S = 1;

    tic
    S1 = Sigma_a;
    W2 = eye(size(Sigma_a));
    J = [];
    conv_flag = 0;

    while(J_S > thres)     
        S1 = Sigma_a;
        S2 = W2*Sigma_b_sc*W2';  

        % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||            
        if det(S1)<0
            disp('S1 not pos def.')        
        end

        if det(S2)<0
            disp('S2 not pos def.')
        end

        L1 = matrixSquareRoot(S1);
        L2 = matrixSquareRoot(S2);
        [U,D,V] = svd(L1'*L2);
        R_hat = V*U';    
        J_S = EuclideanNormMat(L1 - L2*R_hat);
        J(iter) = J_S;        

        %Check convergence
        if (iter>100) && (J(iter-10) - J(iter) < conv_thres)
            disp('Parameter Estimation Converged');
            conv_flag = 1;
            break;
        end

        if (iter>max_iter)
            disp('Exceeded Maximum Iterations');               
            break;
        end

        % Compute approx rotation
        W2 = W2 * R_hat^-1;
        iter = iter + 1;            
    end
    toc

end
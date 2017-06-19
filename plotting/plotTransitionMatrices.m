function [h, Psi] = plotTransitionMatrices(Psi)

h = figure('Color',[1 1 1]);
pi = [];
for i=1:length(Psi.Psi.Eta)   
    % Construct Transition Matrices for each time-series
    f_i   = Psi.Psi.Eta(i).availFeatIDs;
    eta_i = Psi.Psi.Eta(i).eta;
    
    % Normalize self-transitions with sticky parameter    
    pi_i  = zeros(size(eta_i));   
    for ii=1:size(pi_i,1);pi_i(ii,:) = eta_i(ii,:)/sum(eta_i(ii,:));end
    pi{i} = pi_i;    
    
    % Plot them
    subplot(length(Psi.Psi.Eta), 1, i)
    plotTransMatrix(pi{i},strcat('Time-Series (', num2str(i),')'),0, f_i);
end
Psi.pi = pi;

end
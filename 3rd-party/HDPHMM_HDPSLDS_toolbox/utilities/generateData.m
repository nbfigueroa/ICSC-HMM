function data_struct = generateData(model,settings,T);


resample_kappa = settings.resample_kappa;
Kz = settings.Kz;
Ks = settings.Ks;

d = size(model.obsModel.params.nu_delta,1);
dy = d;
if isfield(model.obsModel,'y_params')
    dy = size(model.obsModel.y_params.nu_delta,1);
end

data_struct.obs = zeros(dy,1);

[theta Ustats stateCounts hyperparams data_struct model S] = initializeStructs(model,data_struct,settings);
data_struct = rmfield(data_struct,'blockSize');
data_struct = rmfield(data_struct,'blockEnd');

obsModel = model.obsModel;  % structure containing the observation model parameters
obsModelType = obsModel.type;   % type of emissions including Gaussian, multinomial, AR, and SLDS.
HMMhyperparams = model.HMMmodel.params; % hyperparameter structure for the HMM parameters
HMMmodelType = model.HMMmodel.type; % type of HMM including finite and HDP

hyperparams = sample_hyperparams(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
dist_struct = sample_dist(stateCounts,hyperparams,model);
theta = sample_theta(theta,Ustats,obsModel);

%%

pi_z = dist_struct.pi_z;
pi_s = dist_struct.pi_s;
pi_init = dist_struct.pi_init;


labels = zeros(1,T);
slabels = zeros(1,T);

for t=1:T
    if t>1
        p_z = cumsum(pi_z(labels(t-1),:));
        labels(t) = 1 + sum(p_z(end)*rand(1) > p_z);
        p_s = cumsum(pi_s(labels(t),:));
        slabels(t) = 1 + sum(p_s(end)*rand(1) > p_s);
    else
        p_z = cumsum(pi_init);
        labels(t) = 1 + sum(p_z(end)*rand(1) > p_z);
        p_s = cumsum(pi_s(labels(t),:));
        slabels(t) = 1 + sum(p_s(end)*rand(1) > p_s);
    end           
end


switch obsModelType
    
    case 'Gaussian'
        
        mu = theta.mu;
        invSigma = theta.invSigma;
        
        cholSigma = zeros(d,d,Kz,Ks);
        for kz=1:Kz
            for ks=1:Ks
                cholSigma(:,:,kz,ks) = chol(inv(invSigma(:,:,kz,ks)));
            end
        end
        
        for t=1:T
            Y(:,t) = mu(:,labels(t),slabels(t)) + cholSigma(:,:,labels(t),slabels(t))'*randn(d,1);
        end
        
    case 'AR'
        
        r = model.obsModel.r;
        
        A = theta.A;
        if isfield(theta,'mu')
            mu = theta.mu;
        else
            mu = zeros(d,Kz,Ks);
        end
        invSigma = theta.invSigma;
        
        cholSigma = zeros(d,d,Kz);
        for kz=1:Kz
            cholSigma(:,:,kz) = chol(inv(invSigma(:,:,kz)));
        end
        
        tmp = cholSigma(:,:,labels(1))'*randn(d,r);
        x = tmp(:);
        for t=1:T
            Y(:,t) = A(:,:,labels(t))*x + mu(:,labels(t)) + cholSigma(:,:,labels(t))'*randn(d,1);
            x = [Y(:,t);x(1:(end-d),:)];
        end
        
    case 'SLDS'
        
        Kr = length(stateCounts.Nr);
        pi_r = dist_struct.pi_r;
        
        for t=1:T
            p_r = cumsum(pi_r);
            r(t) = 1 + sum(p_r(end)*rand(1) > p_r);
        end
        
        C = model.obsModel.params.C;
        P0 = model.obsModel.params.P0;
        
        dy = size(C,1);
                
        Lambda_r = theta.theta_r.invSigma;
        if isfield(theta.theta_r,'mu')
            mu_r = theta.theta_r.mu;
        else
            mu_r = zeros(dy,Kr);
        end
        
        cholSigma_r = zeros(dy,dy,Kz);
        for kr=1:Kr
            cholSigma_r(:,:,kr) = chol(inv(Lambda_r(:,:,kr)));
        end
        
        A = theta.A;
        if isfield(theta,'mu')
            mu = theta.mu;
        else
            mu = zeros(d,Kz,Ks);
        end
        invSigma = theta.invSigma;
        
        cholSigma = zeros(d,d,Kz);
        for kz=1:Kz
            cholSigma(:,:,kz) = chol(inv(invSigma(:,:,kz)));
        end
        
        x = chol(P0)'*randn(d,1);
        for t=1:T
            X(:,t) = A(:,:,labels(t))*x + mu(:,labels(t)) + cholSigma(:,:,labels(t))'*randn(d,1);
            x = X(:,t);
            Y(:,t) = C*X(:,t) + cholSigma_r(:,:,r(t))'*randn(dy,1);
        end
        
end



%%

data_struct.obs = Y;
if exist('X')
    data_struct.state = X;
end
data_struct.true_labels = labels;
data_struct.true_sublabels = slabels;
if exist('r')
    data_struct.true_rlabels = r;
end
data_struct.true_params.theta = theta;
data_struct.true_params.dist_struct = dist_struct;

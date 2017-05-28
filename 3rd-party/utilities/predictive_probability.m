function [predictive_prob data_sim] = predictive_probability(saveDirs,trial_iter_mat,Nsims,data_sim)
% 
% load(['../savedStats/HDPSLDSjournal/synth_data/AR2/original_data.mat'])
% 
% obsModelType = model.obsModel.type;
% 
% T = length(data_struct.true_labels);
% pi_s = data_struct.true_params.pi_s;
% pi_z = data_struct.true_params.pi_z;
% % Kz = size(pi_z,2);
% % Ks = size(pi_s,2);
% 
% switch obsModelType
%     case 'Multinomial'
%         P = data_struct.true_params.emissionProbs;
%     case 'Gaussian'
%         mu = data_struct.true_params.mu;
%         Sigma = data_struct.true_params.Sigma;
%     case 'AR'
%         A = data_struct.true_params.A;
%         Sigma = data_struct.true_params.Sigma;
% end


load(['../savedStats/HDPSLDSjournal/synth_data/SLDS/original_data.mat'])

obsModelType = model.obsModel.type;

T = length(data_struct.true_labels);
numStates = 3;
p_self = 0.98;
pi_z = ((1-p_self)/(numStates-1))*ones(numStates);
pi_z(find(eye(numStates))) = p_self;
pi_s = ones(size(pi_z,1),1);
% Kz = size(pi_z,2);
% Ks = size(pi_s,2);

A = model.trueparams.A;
Sigma = model.trueparams.Sigma;
dimu = size(data_struct.obs,1);

if nargin<4
    for sim=1:Nsims
        
        clear Y;
        Y = zeros(dimu,T);
        labels = zeros(1,T);
        labels(1) = 1;
        slabels(1) = 1;
        
        for t=1:T
            if t>1
                p_z = cumsum(pi_z(labels(t-1),:));
                labels(t) = 1 + sum(p_z(end)*rand(1) > p_z);
                p_s = cumsum(pi_s(labels(t),:));
                slabels(t) = 1 + sum(p_s(end)*rand(1) > p_s);
            end
        end
        
        switch obsModelType
            case 'Multinomial'
                for t=1:T
                    P_zs = cumsum(P{labels(t),slabels(t)});
                    Y(:,t) = 1 + sum(P_zs(end)*rand(1) > P_zs);
                end
            case 'Gaussian'
                for t=1:T
                    Y(:,t) = mu{labels(t),slabels(t)} + chol(Sigma{labels(t),slabels(t)})'*randn(dimu,1);
                end
            case 'AR'
                
                d = size(A{1},1);
                r = size(A{1},2)/size(A{1},1);
                
                tmp = chol(Sigma{labels(1)})'*randn(d,r);
                x = tmp(:);
                X = [];
                for t=1:T
                    Y(:,t) = A{labels(t)}*x + chol(Sigma{labels(t)})'*randn(d,1);
                    x = [Y(:,t);x(1:(end-d),:)];
                end
                %data_sim{sim}.X = makeDesignMatrix(Y,model.obsModel.r);
        end
        
        data_sim{sim}.zlength = T;
        data_sim{sim}.obs = Y;
        data_sim{sim}.true_labels = labels;
        data_sim{sim}.blockEnd = cumsum(ones(1,T));
        
    end
end

for dirCount = 1:length(saveDirs)
    load([saveDirs{dirCount} 'info4trial1.mat'])
    saveEvery = settings.saveEvery;
    storeEvery = settings.storeEvery;
    storeStateSeqEvery = settings.storeStateSeqEvery;
    Kz = settings.Kz;
    predProb = zeros(Nsims,size(trial_iter_mat{dirCount},2));
    trial_iter_mat_temp = trial_iter_mat{dirCount};
    store_count = 1;
    for trial_iter_count = 1:size(trial_iter_mat_temp,2)
        trial = trial_iter_mat_temp(1,trial_iter_count);
        iter = trial_iter_mat_temp(2,trial_iter_count);
        
        if rem(iter,settings.saveEvery)==0
            n_save = iter;
            n_store = saveEvery/storeEvery;
            n_storeSeq = saveEvery/storeStateSeqEvery;
        else
            iters_until_save = saveEvery - mod(iter,saveEvery);
            n_save = iter + iters_until_save;
            n_store = mod(iter,saveEvery)/storeEvery;
            n_storeSeq = mod(iter,saveEvery)/storeStateSeqEvery;
        end
        
        filename = [saveDirs{dirCount} 'HDPHMMDPstatsiter' num2str(n_save) 'trial' num2str(trial) '.mat'];
        load(filename)
        
        z = S.stateSeq(n_storeSeq).z;
        counts = histc(z,[1:Kz]);
        counts = counts/sum(counts);
        uniqueZ = find(counts>0.05);
        
        pi_init_temp = S.dist_struct(n_store).pi_init;
        pi_z_temp = S.dist_struct(n_store).pi_z;
        pi_init_temp(setdiff([1:Kz],uniqueZ)) = 0;
        for ii=1:Kz
            pi_z_temp(ii,setdiff([1:Kz],uniqueZ)) = 0;
        end
        
        theta_temp = S.theta(n_store);
        if isfield(S.theta,'ARDhypers')
            ARDhypers = S.theta(n_store).ARDhypers;
            for jj=uniqueZ
                smallLags = find(ARDhypers(:,jj)>100);
                d = size(theta_temp.A,1);
                r = size(theta_temp.A,2)/d;
                if r==1 %SLDS
                    theta_temp.A(:,smallLags,jj)=0;
                else
                    theta_temp.A(:,d*(smallLags-1)+1:d*(smallLags-1)+d,jj)=0;
                end
            end
        end
        
        dist_struct_temp = S.dist_struct(n_store);
        dist_struct_temp.pi_z = pi_z_temp;
        dist_struct_temp.pi_init = pi_init_temp;
        
        for sim = 1:Nsims
            data_sim_temp = data_sim{sim};
            if strcmp(obsModelType,'AR') || strcmp(obsModelType,'SLDS')
                [X,valid] = makeDesignMatrix(data_sim_temp.obs,model.obsModel.r);
                
                data_sim_temp.obs = data_sim_temp.obs(:,find(valid));
                data_sim_temp.X = X(:,find(valid));
                data_sim_temp.true_labels = data_sim_temp.true_labels(find(valid));
            end
            predProb(sim,store_count) = observation_likelihood(data_sim_temp,obsModelType,dist_struct_temp,theta_temp);
        end
        
        store_count = store_count + 1;
    end
    predictive_prob{dirCount}.predProb = predProb;
    
end

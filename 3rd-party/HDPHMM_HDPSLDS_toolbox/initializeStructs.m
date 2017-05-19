function [theta Ustats stateCounts hyperparams data_struct model S] = initializeStructs(model,data_struct,settings)

Kz = settings.Kz;
Ks = settings.Ks;

prior_params = model.obsModel.params;
if ~isfield(data_struct(1),'blockSize')
    data_struct(1).blockSize = [];
end

if ~isfield(data_struct(1),'test_cases')
    data_struct(1).test_cases = 1;
end

if isempty(data_struct(1).test_cases)
    data_struct(1).test_cases = 1;
end

switch model.obsModel.type

    case 'Gaussian'

        dimu = size(data_struct(1).obs,1);
        
        for ii=1:length(data_struct)
            if isempty(data_struct(ii).blockSize)
                data_struct(ii).blockSize = ones(1,size(data_struct(ii).obs,2));
            end
            data_struct(ii).blockEnd = cumsum(data_struct(ii).blockSize);
        end

        theta = struct('invSigma',zeros(dimu,dimu,Kz,Ks),'mu',zeros(dimu,Kz,Ks));
        
        Ustats = struct('card',zeros(Kz,Ks),'YY',zeros(dimu,dimu,Kz,Ks),'sumY',zeros(dimu,Kz,Ks));
        
    case 'Multinomial'

        for ii=1:length(data_struct)
            if size(data_struct(ii).obs,1)>1
                error('not multinomial obs')
            end
            if isempty(data_struct(ii).blockSize)
                data_struct(ii).blockSize = ones(1,size(data_struct(ii).obs,2));
            end
            data_struct(ii).blockEnd = cumsum(data_struct(ii).blockSize);
        end
        
        data_struct(1).numVocab = length(prior_params.alpha);
        
        theta = struct('p',zeros(Kz,Ks,data_struct(1).numVocab));

        Ustats = struct('card',zeros(data_struct(1).numVocab,Kz,Ks));

    case {'AR','SLDS'}

        if settings.Ks~=1
            error('Switching linear dynamical models only defined for Gaussian process noise, not MoG')
        end
        
        switch model.obsModel.priorType
            case 'MNIW'
                
                dimu = size(prior_params.M,1);
                dimX = size(prior_params.M,2);
                
                theta = struct('invSigma',zeros(dimu,dimu,Kz,Ks),'A',zeros(dimu,dimX,Kz,Ks));
                
            case {'MNIW-N','N-IW-N'}
                
                dimu = size(prior_params.M,1);
                dimX = size(prior_params.M,2);
                
                theta = struct('invSigma',zeros(dimu,dimu,Kz,Ks),'A',zeros(dimu,dimX,Kz,Ks),'mu',zeros(dimu,Kz,Ks));
                
            case {'ARD'}
                
                dimu = size(prior_params.M,1);
                dimX = size(prior_params.M,2);
                
                theta = struct('invSigma',zeros(dimu,dimu,Kz,Ks),'A',zeros(dimu,dimX,Kz,Ks),'mu',zeros(dimu,Kz,Ks),'ARDhypers',zeros(dimX,Kz,Ks));
                              
            case {'Afixed-IW-N'}
                
                dimu = size(prior_params.A,1);
                dimX = size(prior_params.A,2);
                
                theta = struct('invSigma',zeros(dimu,dimu,Kz,Ks),'A',repmat(prior_params.A,[1 1 Kz Ks]),'mu',zeros(dimu,Kz,Ks));
                
            otherwise
                error('no known prior type')
        end
        
        Ustats = struct('card',zeros(Kz,Ks),'XX',zeros(dimX,dimX,Kz,Ks),'YX',zeros(dimu,dimX,Kz,Ks),'YY',zeros(dimu,dimu,Kz,Ks),'sumY',zeros(dimu,Kz,Ks),'sumX',zeros(dimX,Kz,Ks));
        
        if strcmp(model.obsModel.type,'SLDS')
            
            model.obsModel.r = 1;
            
            if ~isfield(settings,'Kr')
                Kr = 1;
                model.HMMmodel.params.a_eta = 1;
                model.HMMmodel.params.b_eta = 1;
                display('Using single Gaussian measurement noise model')
            else
                Kr = settings.Kr;
                display('Using mixture of Gaussian measurement noise model')
            end
            
            dimy = size(prior_params.C,1);
            
            switch model.obsModel.y_priorType
                
                case 'IW'
                    theta.theta_r = struct('invSigma',zeros(dimy,dimy,Kr));
                case {'NIW','IW-N'}
                    theta.theta_r = struct('invSigma',zeros(dimy,dimy,Kr),'mu',zeros(dimy,Kr));
                otherwise
                    error('no known prior type for measurement noise')
            end
            
            Ustats.Ustats_r = struct('card',zeros(1,Kr),'YY',zeros(dimy,dimy,Kr),'sumY',zeros(dimy,Kr));
            
            hyperparams.eta0 = 0;
            
            stateCounts.Nr = zeros(1,Kr);
                        
        end
        
        for ii=1:length(data_struct)
            if ~isfield(data_struct(ii),'X') || isempty(data_struct(ii).X)
                
                [X,valid] = makeDesignMatrix(data_struct(ii).obs,model.obsModel.r);
                
                data_struct(ii).obs = data_struct(ii).obs(:,find(valid));
                data_struct(ii).X = X(:,find(valid));
                if isempty(data_struct(ii).blockSize)
                    data_struct(ii).blockSize = ones(1,size(data_struct(ii).obs,2));
                end
                data_struct(ii).blockEnd = cumsum(data_struct(ii).blockSize);
                if isfield(data_struct(ii),'true_labels')
                    data_struct(ii).true_labels = data_struct(ii).true_labels(find(valid));
                end
            end
        end
end


stateCounts.N = zeros(Kz+1,Kz);
stateCounts.Ns = zeros(Kz,Ks);
stateCounts.uniqueS = zeros(Kz,1);
stateCounts.M = zeros(Kz+1,Kz);
stateCounts.barM = zeros(Kz+1,Kz);
stateCounts.sum_w = zeros(1,Kz);

hyperparams.alpha0_p_kappa0 = 0;
hyperparams.rho0 = 0;

switch model.HMMmodel.type
    case 'HDP'
        hyperparams.gamma0 = 0;
    case 'finite'
        if isfield(model.obsModel.params,'alpha0')
            hyperparams.alpha0_p_kappa0 = model.obsModel.params.alpha0;
        end
end


switch model.obsModel.mixtureType
    
    case 'infinite'
          
    hyperparams.sigma0 = 0;
    
    case 'finite'
        
    hyperparams.sigma0 = zeros(1,Ks);
    
end

numSaves = settings.saveEvery/settings.storeEvery;
numStateSeqSaves = settings.saveEvery/settings.storeStateSeqEvery;
T = size(data_struct(1).obs,2);
if strcmp(model.obsModel.type,'SLDS')
    S.stateSeq(1:numStateSeqSaves,length(data_struct)) = struct('z',zeros(1,T),'s',zeros(1,T),'r',zeros(1,T),'state',zeros(dimX,T));
    S.dist_struct(1:numSaves) = struct('pi_r',zeros(1,Kr),'pi_z',zeros(Kz,Kz),'pi_init',zeros(1,Kz),'pi_s',zeros(Kz,Ks),'beta_vec',zeros(1,Kz));
else
    S.stateSeq(1:numStateSeqSaves,length(data_struct)) = struct('z',zeros(1,T),'s',zeros(1,T));
    S.dist_struct(1:numSaves) = struct('pi_z',zeros(Kz,Kz),'pi_init',zeros(1,Kz),'pi_s',zeros(Kz,Ks),'beta_vec',zeros(1,Kz));
end
S.theta(1:numSaves) = theta;
S.hyperparams(1:numSaves) = hyperparams;
S.m = 1;
S.n = 1;

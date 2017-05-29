%%%%%%%%%%%%%%%%%%%%%%%%%%% HDPHMMDPinference.m %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%           ****SEE 'utilities/runstuff.m' FOR EXAMPLE INPUTS *****
%%
%% Inputs:
%%%% data_struct - structure of observations, initial segmentation of data, etc.
%%%% model - structure containing hyperparameters for transition distributions and dynamic parameters 
%%%% settings - structure of settings including number of Gibbs iterations, directory to save statistics to, how often to save, etc.
%%
%% Outputs
%%%% various statistics saved at preset frequency to
%%%% settings.saveDir/HDPHMMDPstatsiter[insert iter #]trial[insert trial #].mat
%%%% in a structure of the form S(store_count).field(time_series).subfield
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ChainStats] = HDPHMMDPinference(data_struct,model,settings,restart)

trial = settings.trial;
if ~isfield(settings,'saveMin')
    settings.saveMin = 1;
end
resample_kappa = settings.resample_kappa;
Kz = settings.Kz;
Niter = settings.Niter;

display(strcat('Trial:',num2str(trial)))

%%%%%%%%%% Generate observations (if not provided) %%%%%%%%%%
%%%                       and                             %%%
%%%%%%%%%%        Initialize variables             %%%%%%%%%%

if exist('restart','var')
    % The optional 'restart' argument allows you to restart the inference
    % algorithm using the statistics that were stored at settings.lastSave
    if restart==1
        n = settings.lastSave;
        
        % Build necessary structures and clear structures that exist as
        % part of the saved statistics:
        [theta Ustats stateCounts hyperparams data_struct model S] = initializeStructs(model,data_struct,settings);
        clear theta Ustats stateCounts hyperparams S
        
        % Load the last saved statistics structure S:
        if isfield(settings,'filename')
            filename = strcat(settings.saveDir,'/',settings.filename,'iter',num2str(n),'trial',num2str(settings.trial));    % create filename for current iteration
        else
            filename = strcat(settings.saveDir,'/HDPHMMDPstats','iter',num2str(n),'trial',num2str(settings.trial));    % create filename for current iteration
        end
        
        load(filename)
        
        obsModel = model.obsModel;  % structure containing the observation model parameters
        obsModelType = obsModel.type;   % type of emissions including Gaussian, multinomial, AR, and SLDS.
        HMMhyperparams = model.HMMmodel.params; % hyperparameter structure for the HMM parameters
        HMMmodelType = model.HMMmodel.type; % type of HMM including finite and HDP
        
        % Set new save counter variables to 1:
        S.m = 1;
        S.n = 1;
        
        % Grab out the last saved statistics from the S structure:
        numSaves = settings.saveEvery/settings.storeEvery;
        numStateSeqSaves = settings.saveEvery/settings.storeStateSeqEvery;
        theta = S.theta(numSaves);
        dist_struct = S.dist_struct(numSaves);
        hyperparams = S.hyperparams(numSaves);
        stateSeq = S.stateSeq(numStateSeqSaves);
        
        % Set the new starting iteration to be lastSave + 1:
        n_start = n + 1;
        
    end
else
    % Set the starting iteration:
    n_start = 1;
    
    % Build initial structures for parameters and sufficient statistics:
    [theta Ustats stateCounts hyperparams data_struct model S] = initializeStructs(model,data_struct,settings);
    
    obsModel = model.obsModel;  % structure containing the observation model parameters
    obsModelType = obsModel.type;   % type of emissions including Gaussian, multinomial, AR, and SLDS.
    HMMhyperparams = model.HMMmodel.params; % hyperparameter structure for the HMM parameters
    HMMmodelType = model.HMMmodel.type; % type of HMM including finite and HDP
        
    % Resample concentration parameters:
    hyperparams = sample_hyperparams_init(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
    
    % Sample the transition distributions pi_z, initial distribution
    % pi_init, emission weights pi_s, and global transition distribution beta
    % (only if HDP-HMM) from the priors on these distributions:
    dist_struct = sample_dist(stateCounts,hyperparams,model);
    
    % If the optional 'formZInit' option has been added to the settings
    % structure, then form an initial mode sequence in one of two ways.  If
    % 'z_init' is a field of data_struct, then the specified initial
    % sequence will be used. Otherwise, the sequence will be sampled from
    % the prior.
    if isfield(settings,'formZInit')
        if settings.formZInit == 1
            [stateSeq INDS stateCounts] = sample_zs_init(data_struct,dist_struct,obsModelType);
            if strcmp(obsModelType,'SLDS')
                stateSeq.r = ones(size(stateSeq.z));
                for Ninit = 1:50
                    theta = sample_theta(theta,Ustats,obsModel);
                    [Lambda_t_t_bwd theta_t_t_bwd] = backward_filter(stateSeq,data_struct,theta,C);
                    internal_data_struct = sample_x(data_struct,stateSeq,theta,Lambda_t_t_bwd,theta_t_t_bwd,C,invP0);
                    [stateSeq INDS stateCounts.Nr] = sample_r(internal_data_struct,dist_struct,theta,stateSeq,INDS);
                    Ustats = update_Ustats(internal_data_struct,INDS,stateCounts,obsModelType);
                end
            else
                Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType);
            end
            display('Forming initial z using specified z_init or sampling from the prior using whatever fixed data is available')
        end
    elseif length(data_struct)>length(data_struct(1).test_cases)
        display('Do you want z_init set to truth for extra datasets?  If so, make setttings.formZinit =1 ')
    end 
        
    % Sample emission params theta_{z,s}'s. If the above 'formZInit' option
    % was not utilized, the initial parameters will just be drawn from the
    % prior.
    theta = sample_theta(theta,Ustats,obsModel);
    
    % Get SLDS fixed params:
    if strcmp(obsModelType,'SLDS')
        C = obsModel.params.C;
        invP0 = inv(obsModel.params.P0);
    end
    
    % Create directory in which to save files if it does not currently
    % exist:
    if ~exist(settings.saveDir,'file')
        mkdir(settings.saveDir);
    end
    
    % Save initial statistics and settings for this trial:
    if isfield(settings,'filename')
        settings_filename = strcat(settings.saveDir,'/',settings.filename,'_info4trial',num2str(trial));    % create filename for current iteration
        init_stats_filename = strcat(settings.saveDir,'/',settings.filename,'initialStats_trial',num2str(trial));    % create filename for current iteration
    else
        settings_filename = strcat(settings.saveDir,'/info4trial',num2str(trial));    % create filename for current iteration
        init_stats_filename = strcat(settings.saveDir,'/initialStats_trial',num2str(trial));    % create filename for current iteration
    end
    save(settings_filename,'data_struct','settings','model') % save current statistics
    save(init_stats_filename,'dist_struct','theta','hyperparams') % save current statistics
    
end

if isfield(data_struct(1),'true_labels')
    total_length = 0;
    length_ii = zeros(1,length(data_struct));
    for ii=1:length(data_struct)
        length_ii(ii) = length(data_struct(ii).true_labels);
        total_length = total_length + length_ii(ii);
    end
    cummlength = cumsum(length_ii);
    z_tot = zeros(1,cummlength(end));
    true_labels_tot = zeros(1,cummlength(end));
    true_labels_tot(1:length_ii(1)) = data_struct(1).true_labels;
    for ii=2:length(data_struct)
        true_labels_tot(cummlength(ii-1)+1:cummlength(ii)) = data_struct(ii).true_labels;
    end
end

if isfield(settings,'ploton') & isfield(data_struct,'true_labels')
    if settings.ploton == 1
        H = figure;
    end
end

clear ChainStats
stats_iter = 1;
logliks = zeros(length(data_struct),Niter);
%%%%%%%%%% Run Sampler %%%%%%%%%%
for n=n_start:Niter

    % Sample z and s sequences given data, transition distributions,
    % HMM-state-specific mixture weights, and emission parameters:
    if strcmp(obsModelType,'SLDS')
        % Sequentially sample z_t|z_\t,y_{1:T}, integrating over x_{1:T}:
        if n==1 && ~isfield(settings,'formZInit')
            % Uses data_struct.z_init if it exists, otherwise samples
            % sequentially treating each time step as if it were the last:
            [stateSeq Lambda_t_t_bwd theta_t_t_bwd] = sample_z_seq_init(data_struct,dist_struct,theta,C,invP0);
        elseif ~rem(n,settings.seqSampleEvery)
            [stateSeq Lambda_t_t_bwd theta_t_t_bwd] = sample_z_seq(stateSeq,data_struct,dist_struct,theta,C,invP0);
        else
            [Lambda_t_t_bwd theta_t_t_bwd] = backward_filter(stateSeq,data_struct,theta,C);
        end
        % Block sample x_{1:T}|z_{1:T},y_{1:T}:
        internal_data_struct = sample_x(data_struct,stateSeq,theta,Lambda_t_t_bwd,theta_t_t_bwd,C,invP0);
        % Block sample z_{1:T}|x_{1:T}
        [stateSeq INDS stateCounts] = sample_zs(internal_data_struct,dist_struct,theta,obsModelType);
        % Block sample r_{1:T}|tildeY_{1:T}, where tildeY_t = y_t - C*x_t.
        % If a single Gaussian measurement noise model is used, r_{1:T}
        % will always be 1's.
        [stateSeq INDS stateCounts.Nr] = sample_r(internal_data_struct,dist_struct,theta,stateSeq,INDS);
        % Create sufficient statistics:
        Ustats = update_Ustats(internal_data_struct,INDS,stateCounts,obsModelType);
        % Note that for the SLDS, the s_{1:T} sequence will always be 1's
        % since we are assuming a single Gaussian process noise model.
    else
        % Block sample (z_{1:T},s_{1:T})|y_{1:T}
        [stateSeq INDS stateCounts] = sample_zs(data_struct,dist_struct,theta,obsModelType);
        % Create sufficient statistics:
        Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType);
    end
 
    % Based on mode sequence assignment, sample how many tables in each
    % restaurant are serving each of the selected dishes. Also sample the
    % dish override variables:
    stateCounts = sample_tables(stateCounts,hyperparams,dist_struct.beta_vec,Kz);
    
    % Sample the transition distributions pi_z, initial distribution
    % pi_init, emission weights pi_s, and avg transition distribution beta:
    dist_struct = sample_dist(stateCounts,hyperparams,model);
    
    % Sample theta_{z,s}'s conditioned on the z and s sequences and the
    % sufficient statistics structure Ustats:
    theta = sample_theta(theta,Ustats,obsModel);

    % Resample concentration parameters:
    hyperparams = sample_hyperparams(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);

    % Build and save stats structure:
    if strcmp(obsModelType,'SLDS')
        S = store_SLDS_stats(S,n,settings,stateSeq,internal_data_struct,dist_struct,theta,hyperparams);
    else
        S = store_stats(S,n,settings,stateSeq,dist_struct,theta,hyperparams);
    end
    
    if (mod(n,settings.saveEvery)==0)
        ChainStats(stats_iter) = S;        
        stats_iter = stats_iter + 1;
    end
   
    
    % Plot stats:
    if isfield(data_struct,'true_labels') & settings.ploton
        % If the 'ploton' option is included in the settings structure (and if it
        % is set to 1), then create a figure for the plots:

        
        if rem(n,settings.plotEvery)==0
            
            Nsets = length(data_struct);

            
            sub_x = Nsets;
            sub_y = 1;
            
            z_tot(1:length_ii(1)) = stateSeq(1).z;            
            for ii=2:Nsets
                z_tot(cummlength(ii-1)+1:cummlength(ii)) = stateSeq(ii).z;
            end
            
            [relabeled_z Hamm assignment relabeled_true_labels] = mapSequence2Truth(true_labels_tot,z_tot);
            
            A1 = subplot(sub_x,sub_y,1,'Parent',H);
            imagesc([relabeled_z(1:cummlength(1)); relabeled_true_labels(1:cummlength(1))],'Parent',A1,[1 settings.Kz]); title(A1,['Iter: ' num2str(n)]);
            for ii=2:Nsets
                F_used(ii,unique(stateSeq(ii).z)) = 1;
                A1 = subplot(sub_x,sub_y,ii,'Parent',H);
                imagesc([relabeled_z(cummlength(ii-1)+1:cummlength(ii)); relabeled_true_labels(cummlength(ii-1)+1:cummlength(ii))],'Parent',A1,[1 settings.Kz]); title(A1,['Iter: ' num2str(n)]);
            end
            drawnow;
            
            if isfield(settings,'plotpause') && settings.plotpause
                if isnan(settings.plotpause), waitforbuttonpress; else pause(settings.plotpause); end
            end
            
            
        end
    end
    
end

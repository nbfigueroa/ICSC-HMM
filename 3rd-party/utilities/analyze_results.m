function [hamming_dist alpha_hist gamma_hist rho_hist sigma_hist unique_z ARDhypers] = analyze_results(trial_vec,settings,object_ind)

Ntrial = length(trial_vec);
saveDir = settings.saveDir;
saveEvery = settings.saveEvery;
storeStateSeqEvery = settings.storeStateSeqEvery;
Niter = settings.Niter;
hamming_dist = zeros(Ntrial,Niter/storeStateSeqEvery);

load([saveDir 'info4trial' num2str(trial_vec(1))])

% beeData = readBeeData(1);
% data_struct(1).true_labels = beeData.labels(2:end);

if strcmp(model.obsModel.priorType,'ARD')
    r = model.obsModel.r;
    [numRow numCol] = size(model.obsModel.params.M);
    if r == 1
        % One hyperparameter per column of SLDS matrix:
        numHypers = numCol;
    else
        % One hyperparameter per lag matrix in VAR matrix:
        numHypers = r;
    end
    ARDhypers = zeros(Ntrial,Niter/storeStateSeqEvery,numHypers,settings.Kz);
% elseif settings.Kr>1
%     eta_hist = zeros(Ntrial,Niter);
end

unique_z = zeros(Ntrial,Niter);

alpha_p_kappa_hist = zeros(size(hamming_dist));
gamma_hist = zeros(size(hamming_dist));
rho_hist = zeros(size(hamming_dist));
sigma_hist = zeros(size(hamming_dist));

numObj = length(data_struct(1).test_cases);
total_length = 0;
length_ii = zeros(1,numObj);
for ii=1:numObj
    length_ii(ii) = length(data_struct(data_struct(1).test_cases(ii)).true_labels);
    total_length = total_length + length_ii(ii);
end
cummlength = cumsum(length_ii);
z = zeros(1,cummlength(end));
true_labels = zeros(1,cummlength(end));
            
trial_count = 1;
for trial = trial_vec
    iter_count = 1;
    for iter=1:storeStateSeqEvery:Niter
        n = iter+saveEvery-1;
        if rem(n,saveEvery)==0 & n<=Niter
            filename = [saveDir 'HDPHMMDPstatsiter' num2str(n) 'trial' num2str(trial) '.mat'];
            load(filename)
            store_count = 1;
        end
        
        z(1:length_ii(1)) = S.stateSeq(store_count,data_struct(1).test_cases(1)).z;
        true_labels(1:length_ii(1)) = data_struct(data_struct(1).test_cases(1)).true_labels;
        for ii=2:numObj
            z(cummlength(ii-1)+1:cummlength(ii)) = S.stateSeq(store_count,data_struct(1).test_cases(ii)).z;
            true_labels(cummlength(ii-1)+1:cummlength(ii)) = data_struct(data_struct(1).test_cases(ii)).true_labels;
        end
        [relabeled_z hamming_dist(trial_count,iter_count) assignment relabeled_true_labels] = mapSequence2Truth(true_labels,z);
        
        if exist('object_ind','var')
            hamm_inds = cummlength(object_ind-1):1:cummlength(object_ind);
            hamming_dist(trial_count,iter_count) = sum(relabeled_z(hamm_inds)~=relabeled_true_labels(hamm_inds))/length(hamm_inds);         
        end
        
        %         z = S.stateSeq(store_count,ii).z;
        %         true_labels = data_struct(ii).true_labels;
        %         [relabeled_z hamming_dist(trial_count,iter_count) assignment relabeled_true_labels] = mapSequence2Truth(true_labels,z);
        
        % %         blah_temp = sub2ind([3 200],trial_count,iter_count);
        % %         if ismember(blah_temp,blah)
        %             subplot(1,2,1),
        %             plot(relabeled_true_labels); ylim([0 5]);
        %             title(['Hamming Dist: ' num2str(hamming_dist(trial_count,iter_count)) ' Trial: ' num2str(trial) ' Iter: ' num2str(iter)])
        %             hold on; plot(relabeled_z,'r'); hold off; %pause(1);
        %
        %             subplot(1,2,2)
        %             ARDtmp = zeros(4,10);
        %             unique_z = unique(z);
        %             for zz=unique_z
        %                 ARDtmp(:,assignment(find(unique_z==zz))) = S.theta(store_count).ARDhypers(:,zz);
        %             end
        %             bar(ARDtmp'); xlim([0 length(unique_z)+1])
        %
        %             waitforbuttonpress;
        % %         end
        
        if n==Niter
            %figure;
            sub_x = floor(sqrt(numObj));
            sub_y = ceil(numObj/sub_x);
            A1 = subplot(sub_x,sub_y,1);
            Kz = max(union(relabeled_true_labels,relabeled_z));
            imagesc([relabeled_z(1:cummlength(1)); relabeled_true_labels(1:cummlength(1))],'Parent',A1,[1 Kz]);
            for ii=2:numObj
                A1 = subplot(sub_x,sub_y,ii);
                imagesc([relabeled_z(cummlength(ii-1)+1:cummlength(ii)); relabeled_true_labels(cummlength(ii-1)+1:cummlength(ii))],'Parent',A1,[1 Kz]);
            end
            drawnow;
        end
        
        counts = histc(z,[1:settings.Kz]);
        counts = counts/sum(counts);
        uniques = find(counts>0.01);
        
        unique_z(trial_count,iter_count) = length(uniques);
        
        [sorted_counts sort_ind] = sort(counts,'descend');
        
        %unique_z(trial_count,iter_count) = length(unique(S.stateSeq(store_count).z));
        
        if isfield(S.theta(store_count),'ARDhypers')
            ARDhypers(trial_count,iter_count,:,1:length(uniques)) = S.theta(store_count).ARDhypers(:,sort_ind(1:length(uniques)));
%         elseif isfield(S.hyperparams(store_count),'eta0')
%             eta_hist(trial_count,iter_count) =
%             S.hyperparams(store_count).eta0

        end
        
        alpha_p_kappa_hist(trial_count,iter_count) = S.hyperparams(store_count).alpha0_p_kappa0;
        gamma_hist(trial_count,iter_count) = S.hyperparams(store_count).gamma0;
        rho_hist(trial_count,iter_count) = S.hyperparams(store_count).rho0;
        sigma_hist(trial_count,iter_count) = S.hyperparams(store_count).sigma0;

        store_count = store_count + 1;
        iter_count = iter_count + 1;
    end
    trial_count = trial_count + 1;
end

alpha_hist = (ones(size(rho_hist))-rho_hist).*alpha_p_kappa_hist;


% 
% [hamming_dist ARDhypers] = analyze_results([1:3],settings);
% 
% for ii=1:3
%     ARD(ii).hypers = ARDhypers(:,:,:,1);
%     ARD(ii).hypers = reshape(ARD(ii).hypers,[600 3]);
%     figure; hist(ARD(ii).hypers)
% end
% 
% 2modeinds = find(ARDhypers(:,:,:,3)==0);
% 3modeinds = find(ARDhypers(:,:,:,4)==0);
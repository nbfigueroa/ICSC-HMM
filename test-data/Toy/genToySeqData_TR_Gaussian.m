function  [data, PsiTrue, Data , True_states, True_theta ] = genToySeqData_TR_Gaussian( nStates, nDim, N, T, pIncludeFeature)
% INPUTS ----------------------------------------------------------
%    nStates = # of available Markov states
%    nDim = number of observations at each time instant
%    N = number of time series objects
%    T = length of each time series
% OUTPUT ----------------------------------------------------------
%    data  :  SeqData object

% ------------------------------- Remember old state to use again afterward
curStream = RandStream.getGlobalStream();
entryState = curStream.State;

% Reset PRNG state to default value with SEED 0
%       so that we always get same synth data regardless of when called
% reset( RandStream.getGlobalStream(), 0);
reset( RandStream.getGlobalStream());

if T < 0
    doVaryLength = 1;
    T = abs(T);
else 
    doVaryLength = 0;
end

doVaryLength = 1;

if ~exist( 'pIncludeFeature', 'var' )
    pIncludeFeature = 0.5;
end
pSelfTrans = 1-(5*nStates)/T;

% Create initial state distribution (uniform)
Pz_init = ones(1, nStates);

% Create state-to-state transition matrix Pz
Pz = zeros(  nStates, nStates );
for k = 1:nStates
    Pz(k,k) = pSelfTrans;
    Pz(k, [1:k-1 k+1:end] ) = (1-pSelfTrans)/(nStates-1);
end
doRowsSumToOne = ( sum(Pz,2) - ones(size(Pz,1),1) ) <= 1e-10;
assert( all(doRowsSumToOne), 'ERROR: Not a valid transition distr.' );

% Create state-specific emission params Px
%   Means are evenly spaced around the unit circle
%   Covariances are aligned so major diagonal of ellipsoid
%        points toward the origin
Px.Mu = zeros( nStates, nDim );
Px.Sigma = zeros( nDim, nDim, nStates );
V = 0.05;
ts = linspace( -pi, pi, nStates+1 );
xs = cos(ts);
ys = sin(ts);

S(:,:,3) = [V 0; 0 V];
S(:,:,1) = 4*[0.5*V .4*V; .4*V .5*V];
S(:,:,2) = [4*V 0 ;0 4*V];
S(:,:,4) = 4*[0.5*V -.4*V; -.4*V .5*V];

if nDim == 1
    Px.Mu = linspace( -1*V, 1*V, nStates )';
    for kk = 1:nStates
        Px.Sigma(:,:,kk) = V;
    end
elseif nDim == 2
%     Px.Mu = [xs(1:end-1)' ys(1:end-1)']
    Px.Mu = [ 0 0; 2 0; 0 2;  2 2];
    
    if nStates == 8
        Px.Sigma(:,:,1:4) = S(:,:,1:4);
        Px.Sigma(:,:,5:8) = S(:,:,1:4);
    elseif nStates == 4
%         Px.Sigma(:,:,1:4) = S(:,:,[1 3 1 3] );
        Px.Sigma(:,:,1:4) = S(:,:,[1 2 3 4] );
    else
        for kk = 1:nStates
            Px.Sigma(:,:, mod(kk,4)+1 ) = S(:,:,mod(kk,4)+1);
        end
    end
else
    Px.Mu = [xs(1:end-1)' ys(1:end-1)'  zeros( nStates, nDim-2 ) ];
    
    if nDim > 10
       nExtras = floor( (nDim-1) / 10 ); 
       Px.Mu(:, 10*(1:nExtras) ) = repmat( 10*sqrt(V)*xs(1:end-1)', 1, nExtras );
       Px.Mu(:, 10*(1:nExtras)+1 ) = repmat( 10*sqrt(V)*ys(1:end-1)', 1, nExtras );
    end
    
    for kk = 1:nStates
       Px.Sigma(1:2, 1:2, kk) = S(:, :, mod(kk-1, 4)+1  ); 
       Px.Sigma(3:end, 3:end,kk) = V*eye( nDim-2 );
    end
end


% Build time series
data = SeqData();
F = zeros( N, nStates );

F_forced = [1 1 0 0; 1 0 1 0; 0 0 1 1; 0 1 0 1];
sTrueAll = [];
for i = 1:N   
    if doVaryLength
        Ti = poissrnd(T);
    else
        Ti = T;
    end
      
%     % Draw subset of states that this time-series exhibits
%     mask = rand( 1, nStates ) < pIncludeFeature;
%     % Ensure mask isn't all zeros
%     if sum( mask ) < 1
%         kk = randsample( nStates, 1);
%         mask(  kk  ) = 1;
%     end
%     if i == 1
%         mask=true(1,nStates);
%     end
    
    mask = F_forced(i+4-N,:);
    F(i,:) = mask;
    
    zTrue = zeros(1,Ti);
    X = zeros( nDim, Ti );
    for t = 1:Ti
        % ---------------------------------------------- Assign true label
        if t == 1
            zcur = multinomial_single_draw( mask.*Pz_init );
        else
            zprev = zTrue(t-1);
            zcur = multinomial_single_draw( mask.*Pz(zprev,:) );
        end
        zTrue(t) = zcur;

        % ---------------------------------------- Assign emissions
       X(:,t) = mvnrnd( Px.Mu(zcur,:),  Px.Sigma(:,:,zcur) );
           
    end
    data = data.addSeq( X, num2str(i), zTrue );
    sTrue = zeros(size(zTrue));
    sTrue(zTrue == 1) = 1;
    sTrue(zTrue == 4) = 1;
    sTrue(zTrue == 2) = 2;
    sTrue(zTrue == 3) = 2;
    PsiTrue.z{i} = zTrue;
    PsiTrue.s{i} = sTrue;
    
    sTrueAll = [sTrueAll sTrue];
end

% ---------------------------------------------------------  Reset stream
curStream = RandStream.getGlobalStream();
curStream.State = entryState;


PsiTrue.F = zeros(N, nStates);
for ii = 1:N
    PsiTrue.F(ii, unique( data.zTrue(ii) ) ) = 1;
end


% Extract true parameters
Mu = zeros(2,nStates);
Sigma = zeros(2,2,nStates);
for kk = 1:nStates
    PsiTrue.theta(kk).mu = Px.Mu(kk,:);
    PsiTrue.theta(kk).invSigma = inv( Px.Sigma(:,:,kk) );
    
    Mu(:,kk) = Px.Mu(kk,:);
    Sigma(:,:,kk) = Px.Sigma(:,:,kk);
end

PsiTrue.Pz = Pz;
PsiTrue.zTrueAll = data.zTrueAll;
PsiTrue.sTrueAll = sTrueAll;

True_theta.K = nStates;
True_theta.Mu = Mu;
True_theta.Sigma = Sigma;


Data = []; True_states = [];
% Extract data for HMM
for i=1:data.N
    Data{i} = data.seq(i)';
    True_states{i} = data.zTrue(i)';
end

label_range_s = unique(data.zTrueAll);
label_range_z = unique(PsiTrue.sTrueAll);

ts = [1:length(Data)];
figure('Color',[1 1 1])
for i=1:length(ts)
    X = Data{ts(i)};
    true_states       = data.zTrue(i)';
    true_super_states = PsiTrue.s{i}';
    % Plot time-series with true labels and true super labels
    subplot(length(ts),1,i);
    data_labeled = [X true_super_states true_states]';
    plotDoubleLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), {'x_1','x_2'}, label_range_z, label_range_s);    
end

title_name  = 'True Emission Parameters';
plot_labels = {'$y_1$','$y_2$'};
labels = [1 2 3 4 1 1] ;
plotGaussianEmissions2D(True_theta, plot_labels, title_name, labels);

% Similarity matrix
sigma = [];
for i=1:True_theta.K
    sigmas{i} = True_theta.Sigma(:,:,i);
end

gamma = 2; 
dis_type = 2;
spcm = ComputeSPCMfunctionMatrix(sigmas, gamma, dis_type);  
S = spcm(:,:,2);

PsiTrue.S = S;


end % main function




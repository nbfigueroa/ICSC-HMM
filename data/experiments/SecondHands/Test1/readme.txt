% Data
- original sampling rate
- smoothed out
- enforce zero-mean

% Setting model hyper-parameters
modelP = {'bpM.gamma', 2,'obsM.Scoef',1,'bpM.c', 1};

% Sampling Algorithm Settings
algP = {'Niter', 100,'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD

% Initial State of Markov Chain
initP  = {'F.nTotal', 1}; 

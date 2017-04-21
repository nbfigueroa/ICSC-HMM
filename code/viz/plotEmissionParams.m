% function [hh] = plotEmissionParams( varargin )
function [hh] = plotEmissionParams( Psi , cmap)

MAX_K   = 12;
MIN_THR = 0.01;
% 
% % ================================================ Process User Input
% if isstruct( varargin{1} )
%     if isfield( varargin{1}, 'ThetaM' )
%         
%         theta = varargin{1}.ThetaM.theta;
%     elseif isfield( varargin{1}, 'theta' ) || isprop(varargin{1},'theta')
%         
%         theta = varargin{1}.theta;
%     else
%         theta = varargin{1};
%     end
%     
%     if isfield( varargin{1}, 'stateSeq' )
%         stateSeq = varargin{1}.stateSeq;
%     end
%     
%     for ll = 2:length( varargin )
%        curArg = varargin{ll};
%        if ishandle(curArg)
%            figH = curArg;
%        elseif isobject(curArg);
%            data = curArg;
%        end
%     end
%     
% elseif isobject( varargin{1} )
%     if isprop( varargin{1}, 'theta' )
%         theta = varargin{1}.theta;
%     end
% else
%     jobID = varargin{1};
%     taskID = varargin{2};
%     DATA = loadSamplerOutput( jobID, taskID, {'iters', 'Psi'} );
%     data = loadSamplerInfo( jobID, taskID, {'data'} );
% 
%     if length( varargin ) >= 3
%         queryIter = varargin{3};
%         [~, idx] = min( abs( queryIter - DATA.iters.Psi ) );
%     else
%         idx = length( DATA.Psi );
%     end
%     Psi = DATA.Psi( idx );
%     theta = Psi.theta;
%     stateSeq = Psi.stateSeq;
%     
%     if length( varargin ) >= 4
%         objIDs = varargin{4};
%     end
% end

theta = Psi.theta;
stateSeq = Psi.stateSeq;

K = length( theta );

if ~exist('figH','var')
    figH = gca;
end

if exist( 'stateSeq', 'var' )
    
    Zall = horzcat( stateSeq(:).z );
    N = length( Zall );
    [Zcounts, sortIDs] = sort( histc( Zall, 1:K ), 'descend' );
    sortIDs = sortIDs( Zcounts > 0 );
    
    K = length( sortIDs );
    K = min( K, MAX_K );
       
    theta = theta( sort(sortIDs(1:K)) ) ;
    
else
    
    K = min( K, MAX_K );
    theta = theta( 1:K );
    
end



% K = length( theta );

fNames = fieldnames(theta);
if fNames{1} == 'mu'
    obsType = 'Gaussian';
elseif fNames{1} == 'A'
    obsType = 'AR-Gaussian';
elseif fNames{1} == 'logp'
    obsType = 'Multinomial';
end


switch obsType
    case 'Multinomial'
       
        Px = exp( vertcat( theta(:).logp ) );
        set( gcf, 'Units', 'normalized', 'Position', [0 0.5 0.25 0.5] );
        imagesc( figH, Px, [0 0.01] );
        set( gca, 'Position', [0.1 0.1 0.8 0.06*K] );
    case 'Gaussian'

        if exist( 'data', 'var' )
            X = data.Xdata;
            plot(figH, X(1,:), X(2,:), 'k.' );
        end

        for kk = 1:K            
            Mu(kk,:) = theta(kk).mu(1:2);
            invSigma(:,:,kk) = theta(kk).invSigma(1:2,1:2);
        end         
        
%         plotGauss( Mu, invSigma, 1, jet(K), figH );
        plotGauss( Mu, invSigma, 1, cmap, figH );
%         B = 1.6;
%         axis( [-B B -B B] );
        
    case 'AR-Gaussian'
        D = size( theta(1).A, 1);
        for kk = 1:K
            Mu(kk,:) = [ mean( diag( theta(kk).A ) )  zeros( 1, D-1) ];
            invSigma(:,:,kk) = theta(kk).invSigma;
        end 
        plotGauss( Mu, 1000*invSigma, 1, jet(MAX_K), figH );
      
    case 'Bernoulli'
        error( 'TO DO' );
    case 'Multinomial'
        error( 'TO DO' );
end

hold off;
drawnow;
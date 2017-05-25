%% Generate random behavior sequences
% INPUTS ----------------------------------------------------------
%    nStates = # of available Markov states
%    nDim = number of observations at each time instant
%    N = number of time series objects
%    T = length of each time series
clear all
% nStates = 5; %Full
nStates = 4; %No screw
% nStates = 3; %Min
Tmax = 3000;
Tmin = 1000;
Timin = 500;
N = 5;
nDim = 7;
% OUTPUT ----------------------------------------------------------
%    data  :  ARSeqData object
%               note that each sequence will *actually* have T-R
%               observations, since we need "R" to properly
%                define the likelihood of the first "kept" observation

pIncludeFeature = 0.80;
pSelfTrans = 1 - 1/randsample( 10, 1);

% Build randoms behavior transition matrix
F = zeros( N, nStates );
for i = 1:N
    % Draw subset of states that this time-series exhibits
    mask = rand( 1, nStates ) < pIncludeFeature;
    % Ensure mask isn't all zeros
    if sum( mask ) < 1
        kk = randsample( nStates, 1);
        mask(  kk  ) = 1;
    end
    F(i,:) = mask;    
end


% n = 2;
% F = zeros( N, nStates+n);
% for i = 1:N
%     % Draw subset of states that this time-series exhibits
%     mask = rand( 1, nStates+n ) < pIncludeFeature;
%     % Ensure mask isn't all zeros
%     if sum( mask ) < 1
%         kk = randsample( nStates+n, 1);
%         mask(  kk  ) = 1;
%     end
%     F(i,:) = mask;    
% end



%% Build N-timeseries
Nstd = 0;
for i=1:N
    fprintf('Series %d \n',i)
    behaviors = F(i,:);
    T = Tmin + randsample(Tmax,1);
    rs = randsample(T,sum(behaviors)+randsample(2,1)-1) + Timin;
    Tint = floor(T*(rs/sum(rs)));
    T = sum(Tint);
    k=0;
    %Generate random behavior times
%     for j=1:size(behaviors,2)
%         if (behaviors(1,j)==1)
%             k=k+1;
%             behaviorsT(1,j) = Tint(k);
%         else
%             behaviorsT(1,j) = 0;
%         end
%     end
%     for j=1:size(behaviors,2)
%         if (behaviors(1,j)==1)
%             behaviorsT(1,j) = Tint(j);
%         end
%     end
    behaviorsID=find(behaviors==1);
    behaviorsIDrand = behaviorsID(randperm(length(behaviorsID)));
    behaviorsID = behaviorsIDrand;
    if length(Tint)>length(behaviorsID)
%         fprintf('inside if');
        behLeft = length(Tint) - length(behaviorsID);
        for ii=1:behLeft
            temp = abs(behaviorsID(end)-randsample(length(behaviorsID-1),1)-1);
            if temp==0
                temp=1;
            end
            behaviorsID = [behaviorsID  behaviorsID(temp)];    
        end
    end
    
    %Randomize behaviors
    behaviorsIDrand = behaviorsID(randperm(length(behaviorsID)-1));
    behaviorsID = behaviorsIDrand;
    %Generate timeseries
%     Ztrue{i,1}=behaviorsID;
    cc=hsv(size(behaviorsID,2));
    figure;
    for k=1:size(behaviorsID,2)
        fprintf('Behavior %d type %d: ',k,behaviorsID(k));
        Xik = generateBehaviorsR(behaviorsID(k),Tint(k));
%         Xik = generateMinBehaviors(behaviorsID(k),Tint(k));

        if k~=1
        %With translations        
            lastXik = X{i,k-1}(1:3,size(X{i,k-1},2));
            qlast = [X{i,k-1}(4,size(X{i,k-1},2));X{i,k-1}(5,size(X{i,k-1},2));X{i,k-1}(6,size(X{i,k-1},2));X{i,k-1}(7,size(X{i,k-1},2))];
            Rlast = quaternion(qlast);
            Tlast=eye(4,4);Tlast(1:3,1:3)=Rlast;Tlast(1:3,4)=lastXik;
            firstXik = Xik(1:3,1);
            qfirst = [Xik(4,1);Xik(5,1);Xik(6,1);Xik(7,1)];
            Rfirst = quaternion(qfirst);
        % Independent Rotations
        back = repmat(firstXik,[1 size(Xik,2)]);
        forward = repmat(lastXik,[1 size(Xik,2)]);
        Xcurr = Xik(1:3,:) - back + forward;
        Xcurr = [Xcurr;Xik(4:7,:)];
        
        Ttemp = length(Xcurr);
        Tintervals = [Tintervals Ttemp];
        
        Xtemp = cat(2,Xall,Xcurr);
        Xall = [];
        Xall = Xtemp;
        
        %Without Translations
        XcurrO = Xik;
        
        TtempO = length(XcurrO);
        TintervalsO = [TintervalsO TtempO];
        
        XtempO = cat(2,XallO,XcurrO);
        XallO = [];
        XallO = XtempO;

        else
            Xcurr = Xik;
            Xall = Xcurr;
            Tintervals = length(Xcurr);
            
            %Without translations
            XcurrO = Xik;
            XallO = XcurrO;
            TintervalsO = length(XcurrO);
        end     
        X{i,k} = Xcurr;
        XtO{i,k} = XcurrO;
        subplot(2,1,1)
        plot3(Xcurr(1,:),Xcurr(2,:),Xcurr(3,:), 'color',cc(k,:));hold on
        subplot(2,1,2)
        plot3(XcurrO(1,:),XcurrO(2,:),XcurrO(3,:), 'color',cc(k,:));hold on
    end
    

    Xn{i,1} = Xall; 
    XnO{i,1} = XallO;
    Tn{i,1} = Tintervals;
    TnO{i,1} = TintervalsO;
end

%% Add Noise
Nstd = 0.1;
for i=1:N
    %Add noise
    Xalln = Xn{i,1};
    XallnO = XnO{i,1};
    for ii=1:size(Xalln,1)
            for jj=1:size(Xalln,2)
                Xalln(ii,jj) = Xalln(ii,jj) + Xalln(ii,jj)*(Nstd*randn(1,1)); 
                XallnO(ii,jj) = XallnO(ii,jj) + XallnO(ii,jj)*(Nstd*randn(1,1)); 
            end
    end
    XnNoise{i,1}=Xalln;
    XnNoiseO{i,1}=XallnO;
end


%% Preprocess for segmentation (From MoCap data pre-processing experiment)

RobotChannelNames = {'EndEffector.x', 'EndEffector.y', 'EndEffector.z', ... 
    'EndEffector.qw', 'EndEffector.qx', 'EndEffector.qy','EndEffector.qz'};

% nDim = 8;
nDim = 7;
Preproc.nObj = N;
Preproc.obsDim = nDim;
Preproc.R = 1;
Preproc.windowSize = nDim;
Preproc.channelNames = RobotChannelNames;

% Create data structure
Robotdata = SeqData();
RobotdataN = SeqData();

% Create data structure for PR2 Code
% data_struct = struct;

% Read in every sequence, one at a time
for ii = 1:length( Xn )
%     
%     D = Xn{ii};   
%     Dn = XnNoise{ii};
    
    D = Xn{ii}(1:7,:);   
    Dn = XnNoise{ii}(1:7,:);

%     % Enforce zero-mean
    D = D';
    Dn = Dn';
    D = bsxfun( @minus, D, mean(D,2) );
    Dn = bsxfun( @minus, Dn, mean(Dn,2) );
    D = D';
    Dn = Dn';

%     D = preprocessMocapData(D, Preproc.windowSize,2);
    %Scale each component of the observation vector so empirical variance
    %Renormalize so for each feature, the variance of the first diff is 1.0
    %of 1st difference measurements is = 1
%     varFirst =  var(diff(D'));
%     for jj=1:length(varFirst)
%         D(jj,:) = D(jj,:)./sqrt(varFirst(jj));
%     end   
%     
%     varFirstN =  var(diff(Dn'));
%     for jj=1:length(varFirstN)
%         Dn(jj,:) = Dn(jj,:)./sqrt(varFirstN(jj));
%     end   
%     

%     
    Tints = Tn{ii,1};
    labels = Ztrue{ii,1};
    for jj=1:length(Tints)
        if jj==1
            Truelabels = repmat(labels(jj),[1 Tints(jj)]); 
        else
            temp = repmat(labels(jj),[1 Tints(jj)]);
            Truelabels = [Truelabels temp];
        end
    end
    TL{ii,1} = Truelabels; 
    % Add the sequence to our data structure
    Robotdata = Robotdata.addSeq( D, num2str(ii), Truelabels);
%     RobotdataN = RobotdataN.addSeq(Dn, num2str(ii), Truelabels);
%     data_struct(ii).obs = D;
%     data_struct(ii).true_labels = Truelabels;
end

% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions

Robotdata = Robotdata; % keeping odata around lets debug with before/after
RobotdataN = RobotdataN;
RobotdataAR = ARSeqData( Preproc.R, Robotdata);
RobotdataARN = ARSeqData( Preproc.R, RobotdataN);
%%
    Dall = Robotdata.Xdata; 

    meanSigma = 2.0*cov(diff(Dall'));  %If bad segmentation, try values between 0.75 and 5.0
    for i=1:size(meanSigma,1)
        for j=1:size(meanSigma,2)
            if(i~=j) 
                meanSigma(i,j) = 0;
            end
        end
    end
    sig0 = meanSigma;  %Only needed for MNIW-N prior
    
    
%%
    
    meanSigma = 2.0*cov(diff(Ybig3'))  %If bad segmentation, try values between 0.75 and 5.0
    for i=1:size(meanSigma,1)
        for j=1:size(meanSigma,2)
            if(i~=j) 
                meanSigma(i,j) = 0;
            end
        end
    end




%% Plot generated timeseries
for j=1:1
scrsz = get(0,'ScreenSize');
%ScreenSize is a four-element vector: [left, bottom, width, height]:

fig=figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
title_handle = title('Dummy Action Sequence');

itv=20;
rotation_spd=0.5;
delay=0.000001;

length = 0.4;

az=15;
el=64;
view(az,el);
grid on;
xlabel('x', 'fontsize',16);
ylabel('y', 'fontsize',16);
zlabel('z', 'fontsize',16);
h_legend=legend('X','Y','Z');


% Xtest = generateBehaviors(5,500);
% data = Xtest;
% Action Sequence Data
data = Xn{j};
    count = j;
    for i=1:4:size(data,2)
        el=64;
        
        roll = 0; pitch = 0; yaw = 0;
        % From peter's toolbox
%         R = rpy2r(roll,pitch,yaw);
        R = quaternion([data(4,i);data(5,i);data(6,i);data(7,i)], true);
%         roll = data(4,i);pitch=data(5,i);yaw=data(6,i);
%         R = rpy2r(roll,pitch,yaw);

        % generate axis vectors
        tx = [length,0.0,0.0];
        ty = [0.0,length,0.0];
        tz = [0.0,0.0,length];
        % Rotate it by R
        t_x_new = R*tx';
        t_y_new = R*ty';
        t_z_new = R*tz';
        
        % translate vectors to camera position. Make the vectors for plotting
        origin=[data(1,i),data(2,i),data(3,i)];
        tx_vec(1,1:3) = origin;
        tx_vec(2,:) = t_x_new + origin';
        ty_vec(1,1:3) = origin;
        ty_vec(2,:) = t_y_new + origin';
        tz_vec(1,1:3) = origin;
        tz_vec(2,:) = t_z_new + origin';
        hold on;
        
        
        
        % Plot the direction vectors at the point
        p1=plot3(tx_vec(:,1), tx_vec(:,2), tx_vec(:,3));
        set(p1,'Color','Green','LineWidth',1);
        p1=plot3(ty_vec(:,1), ty_vec(:,2), ty_vec(:,3));
        set(p1,'Color','Blue','LineWidth',1);
        p1=plot3(tz_vec(:,1), tz_vec(:,2), tz_vec(:,3));
        set(p1,'Color','Red','LineWidth',1);
        
        perc = count*itv/numel(roll)*100;
        %fprintf('Process = %f\n',perc);
        %text(1,-3,0,['Process = ',num2str(perc),'%']);
        set(title_handle,'String',['Process = ',num2str(perc),'%'],'fontsize',16);
        count=count+1;
        
        az=az+rotation_spd;
        view(az,el);
        drawnow;
        pause(delay);  % in second
%         f = getframe(fig);
%         aviobj=addframe(aviobj,f);
    end;
end


        % Unified Rotations
%         Xcurr = zeros(size(Xik));
%         for p=1:size(Xik,2)
%                 Poscurr = Xik(1:3,p);
%                 Rcurr = quaternion([Xik(4,p);Xik(5,p);Xik(6,p);Xik(7,p)]);
%                 Tfirst = eye(4,4);Tfirst(1:3,1:3)=Rfirst;Tfirst(1:3,4)=firstXik;
%                 Tcurr=eye(4,4);Tcurr(1:3,1:3)=Rcurr;Tcurr(1:3,4)=Poscurr;
%                 Tnew=inv(Tcurr)*inv(Tfirst)*Tlast;
%                 Xcurr(1:3,p)=Tnew(1:3,4);Xcurr(4:7,p)=quaternion(Tnew(1:3,1:3));
%         end


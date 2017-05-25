%% Convert 7d (x,y,z,qw,qi,qj,qk) Trajectories to Homogeneous Matrices (Ridid Body Motion Trajectories)
%rescale quaternion
Xnr = Xn;
for j=1:length(Xn)
Xnr{j}(4:7,:) = Xn{j}(4:7,:)/50; 
data = Xnr{j};
R = quaternion([data(4,:);data(5,:);data(6,:);data(7,:)], true);
t = reshape(data(1:3,:),3,1,size(data,2));
norm = [0 0 0 1];
n = repmat(norm',1,size(data,2));
N = reshape(n,1,4,size(data,2));
H_data = cat(2,R,t);
H = cat(1,H_data,N);
Hn{j,1} = H;
end

%% Visualize Rigid Body Trajectories
named_figure('Trajectory',1);
clf;
drawframetraj(Hn{3},2,1);
% animframetraj(Hn{1});
%  nice3d()
%  hold on;

%% Convert Homogenous Rigid Motion Representation (H=4x4) to Time-based Invariants per motion (w1,v1,w2,v2,w3,v3)
% for j=1:length(Hn)   
for j=1:1
H = Hn{j,1};
truelabels = TL{j,1};
Tints = Tn{j,1};
% tb_inv = zeros(6,size(H,3));
tb_inv = [];
fprintf('----------------------------\n');
fprintf('Sequence %d Length: %d\n',j, length(H));
for ii=1:length(Tints)
    fprintf('Motion %d\n', ii);
    if ii == 1
        t_init = 1; 
        t_end = Tints(ii);
    else
        t_init = t_end + 1;
        t_end = t_init - 1 + Tints(ii);
    end
    fprintf ('Init: %d End:%d \n',t_init,t_end);
    Xm = Xnr{j,1}(:,t_init:t_end);
    Mtype = unique(truelabels(t_init:t_end));
    x = 1:1:length(Xm);
    figure;
    subplot(4,1,1);
    tit = strcat(num2str(Mtype),' xyzq');
    plot(x,Xm); title(tit)
    
    %Compute twist for Motion ii of j
    Hm = H(:,:,t_init:t_end);
    tw = zeros(6,length(Hm));
    tb_inv_m = zeros(6,length(Hm));
    for i=1:length(Hm)
        if ii==1 && i==1
            T12 = Hm(:,:,i);
        else
            %Relative twist
            if i==1
                T1 = H(:,:,t_init-1);
            else
                T1 = H(:,:,i-1);
            end
            T2 = H(:,:,i);
            T12 = inv(T1)*T2;
        end
        Hrel(:,:,i) = T12;
        [xi theta] = homtotwist(T12);
        tw(:,i)=xi*theta;
    end
    TwistTh{ii,1} = tw;
    subplot(4,1,2);
    tit2 = strcat(num2str(Mtype),' twist');
    plot(x,tw); title(tit2)
%     
    % twist derivatives
    if ii==1
        tw_dot = [zeros(6,1) diff(tw')'];
    else
        lastTwist = TwistTh{ii-1,1}(:,end);
        tw_m = [lastTwist tw];
        tw_dot = diff(tw_m')';
    end
        tw_dotdot = [zeros(6,1) diff(tw_dot')'];
        Xm3d_dot = [zeros(3,1) diff(Xm(1:3,:)')'];
        Xm3d_dotdot = [zeros(3,1) diff(Xm3d_dot')'];
        Xm3d_dotdotdot = [zeros(3,1) diff(Xm3d_dotdot')'];

    clear norm;
    %Computation of time-based invariants
    for i=1:length(Hm)
            if i==1
               w1=0;v1=0;w2=0;v2=0;w3=0;v3=0;
            else
            %preliminary variables
            omega = tw(4:6,i);
            omega_dot = tw_dot(4:6,i);
            omega_dotdot = tw_dotdot(4:6,i);
            v = tw(1:3,i);
            v_dot = tw_dot(1:3,i);
            v_dotdot = tw_dotdot(1:3,i);
            sign = +1;
           
            if norm(omega)==0 % Pure translation
%                 display('pure translation')
                w1 = 0;
                v1 = sign*(norm(v));
%                 w2 = sign*(norm(cross(v,v_dot))/(norm(v)^2));
                %using formulas for curvature and torsin of a space curve
                k = norm(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)))/(norm(Xm3d_dot(:,i))^3);
                tau = dot(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)),(Xm3d_dotdotdot(:,i))/(norm(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)))^2));
                w2 = sign*k*norm(v);
                w3 = sign*tau*norm(v);
                v2 = 0;%undefined
                v3 = 0;%undefined                
            elseif norm(cross(omega,omega_dot))==0 %ISA w/constant orientation
                display('constant orientation')
                ex = sign*(omega/norm(omega));
                w1 = dot(omega,ex);
                v1 = dot(v,ex);    
                w2 = 0;                
                pfirstdot = (cross(omega_dot,v) + cross(omega,v_dot))*norm(omega)^2;
                psecdot = cross(omega,v)*dot(omega,omega_dot);
                p_perp_dot = (pfirstdot-2*psecdot)/(norm(omega)^4);
                v2 = sign*norm(p_perp_dot);
%                 w3 = norm(cross(p_perp_dot,p_perp_dotdot))/(norm(p_perp_dot)^2);
                w3 = 0; %undefined
                v3 = 0; %undefined
            else
%                 display('Normal motion')
                ex = sign*(omega/norm(omega));
                ey = sign*(cross(omega,omega_dot)/norm(cross(omega,omega_dot)));
                p_perp = cross(omega,v)/(norm(omega)^2); 
                pfirstdot = (cross(omega_dot,v) + cross(omega,v_dot))*norm(omega)^2;
                psecdot = cross(omega,v)*dot(omega,omega_dot);
                p_perp_dot = (pfirstdot-2*psecdot)/(norm(omega)^4);
                
                %time-based invariants
%                 w1 = dot(omega,ex);
%                 v1 = dot(v,ex);
%                 w2 = dot(cross(omega,omega_dot)/(norm(omega)^2),ey);
                v2 = dot(ey,p_perp_dot);                              
                w1 = sign*norm(omega);
                v1 = sign*dot(v,omega)/norm(omega);
                w2 = sign*norm(cross(omega,omega_dot))/(norm(omega)^2);
               
                w3 = dot((cross(cross(omega,omega_dot),cross(omega,omega_dotdot)))/(norm(cross(omega,omega_dot))^2),ex);
                num1_1 = cross(omega_dot,cross(omega,omega_dot)) + cross(omega,cross(omega,omega_dotdot));
                num1_2 = (norm(omega)^2 * (cross(omega_dot,v) + cross(omega,v_dot))) - 2*dot(omega,omega_dot)*cross(omega,v);
                den12 = norm(omega)^3 * norm(cross(omega,omega_dot))^2;
                v3first = sign*dot(num1_1,num1_2)/den12;
                num2_1 = cross(omega,cross(omega,omega_dot));
                num2_2 = norm(omega)^2 * (cross(omega_dotdot,v) + cross(2*omega_dot,v) + cross(omega,v_dotdot)) - 2*(norm(omega_dot)^2 + dot(omega,omega_dot))*cross(omega,v);
                v3second = sign*dot(num2_1,num2_2)/den12;
                v3third1 = (3/2*(dot(omega,omega_dot))/(norm(omega)^2)) + (dot(cross(omega,omega_dot),cross(omega,omega_dotdot)))/(norm(cross(omega,omega_dot))^2); 
                v3third2num_1 = cross(omega,cross(omega,omega_dot));
                v3third2num_2 = norm(omega)^2 * (cross(omega_dot,v) - cross(omega,v_dot)) - 2*dot(omega,omega_dot)*cross(omega,v); 
                v3third2 = dot(v3third2num_1,v3third2num_2)/den12;
                v3 = v3first + v3second + sign*v3third1*v3third2;
            end
            end
            if isnan(w1),w1=0;end
            if isnan(v1),v1=0;end
            if isnan(w2),w2=0;end
            if isnan(v2),v2=0;end
            if isnan(w3),w3=0;end
            if isnan(v3),v3=0;end
            tb_inv_m(:,i) = [w1;v1;w2;v2;w3;v3];
%             tb_inv_m(:,i) = [w1;v1;w2;v2];
    end
    subplot(4,1,3);
    tit2 = strcat(num2str(Mtype),'tb_inv');
    plot(x,tb_inv_m); title(tit2)
    
    
    %Check if time-based invariants are correct (reconstruct original
    %trajectory)
    Hm_rec = zeros(4,4,size(tb_inv_m,2));
    Xm_rec = zeros(7,size(tb_inv_m,2));
    if ii==1
        lastH = H(:,:,t_init);
    end
    for jj=1:size(tb_inv_m,2)
               delta_T = eye(4);
               twist_ISA = [tb_inv_m(5,jj);tb_inv_m(3,jj);0;tb_inv_m(6,jj);tb_inv_m(4,jj);0];
               delta_p = twist_ISA(4:6)*jj;
               if norm(twist_ISA(1:3))==0
                   e_eq = [0;0;0];
               else
                   e_eq = twist_ISA(1:3)/norm(twist_ISA(1:3));
               end
               delta_theta = norm(twist_ISA(1:3))*jj;
               delta_T(1:3,1:3) = skewexp(e_eq,delta_theta);
               delta_T(1:3,4) = delta_p;
               twist_ref = [tb_inv_m(2,jj);0;0;tb_inv_m(1,jj);0;0];
               if jj==1
                    Hi = lastH*delta_T*twistexp(twist_ref);
               else
                    T = Hm_rec(:,:,jj-1)*delta_T;               
                    Hi = T*twistexp(twist_ref);
               end
               Hm_rec(:,:,jj) = Hi; 
               Xm_rec(1:3,jj) = Hi(1:3,4);
               Xm_rec(4:7,jj) = quaternion(Hi(1:3,1:3),true);
    end
    lastH = Hm_rec(:,:,end);
    
    subplot(4,1,4);
    tit2 = strcat(num2str(Mtype),'reconstructed');
    plot(x,Xm_rec); title(tit2)
    
%     named_figure('Trajectory',ii);
%     clf;
%     drawframetraj(Hm_rec,2,1);
%     
    %Scaled time-based invariants
%     for jj=1:size(tb_inv_m,1)
%         tb_inv_m(jj,:)=(tb_inv_m(jj,:)-min(tb_inv_m(jj,:)))/range(tb_inv_m(jj,:));
%     end
    
    tb_inv = [tb_inv tb_inv_m];
    
end
TB_INV{j,1} = tb_inv;
end
%% Convert Homogenous Rigid Motion Representation (H=4x4) to Time-based Invariants full trajectory (w1,v1,w2,v2,w3,v3)
for j=1:length(Hn)   
% for j=1:3
H = Hn{j,1};
truelabels = TL{j,1};
Tints = Tn{j,1};
fprintf('----------------------------\n');
fprintf('Sequence %d Length: %d\n',j, length(H));

    Xm = Xnr{j,1};
    Mtype = unique(truelabels);
    x = 1:1:length(Xm);
    figure;
    subplot(3,1,1);
    plot(x,Xm); title('xyzq')
    
    %Compute twist for trajectory j
    Hm = H;
    tw = zeros(6,length(Hm));
    tb_inv = zeros(6,length(Hm));
    for i=1:length(Hm)
        T = Hm(:,:,i);
        [xi theta] = homtotwist(T);
        tw(:,i)=xi*theta;
    end
%     subplot(2,1,2);
%     tit2 = strcat(num2str(Mtype),'twist');
%     plot(x,tw); title(tit2)
%     
    % twist derivatives
    tw_dot = [zeros(6,1) diff(tw')'];
    tw_dotdot = [zeros(6,1) diff(tw_dot')'];
    Xm3d_dot = [zeros(3,1) diff(Xm(1:3,:)')'];
    Xm3d_dotdot = [zeros(3,1) diff(Xm3d_dot')'];
    Xm3d_dotdotdot = [zeros(3,1) diff(Xm3d_dotdot')'];
    
    clear norm;
    
    for i=1:length(Hm)
            if i==1
               w1=0;v1=0;w2=0;v2=0;w3=0;v3=0;
            else
            %preliminary variables
            omega = tw(4:6,i);
            omega_dot = tw_dot(4:6,i);
            omega_dotdot = tw_dotdot(4:6,i);
            v = tw(1:3,i);
            v_dot = tw_dot(1:3,i);
            v_dotdot = tw_dotdot(1:3,i);
            sign = +1;
           
            if norm(omega)==0 % Pure translation
%                 display('pure translation')
                w1 = 0;
                v1 = sign*(norm(v));
%                 w2 = sign*(norm(cross(v,v_dot))/(norm(v)^2));
                %using formulas for curvature and torsin of a space curve
                k = norm(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)))/(norm(Xm3d_dot(:,i))^3);
                tau = dot(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)),(Xm3d_dotdotdot(:,i))/(norm(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)))^2));
                w2 = sign*k*norm(v);
                w3 = sign*tau*norm(v);
                v2 = 0;%undefined
                v3 = 0;%undefined                
            elseif norm(cross(omega,omega_dot))==0 %ISA w/constant orientation
                display('constant orientation')
                ex = sign*(omega/norm(omega));
                w1 = dot(omega,ex);
                v1 = dot(v,ex);    
                w2 = 0;                
                pfirstdot = (cross(omega_dot,v) + cross(omega,v_dot))*norm(omega)^2;
                psecdot = cross(omega,v)*dot(omega,omega_dot);
                p_perp_dot = (pfirstdot-2*psecdot)/(norm(omega)^4);
                v2 = sign*norm(p_perp_dot);
%                 w3 = norm(cross(p_perp_dot,p_perp_dotdot))/(norm(p_perp_dot)^2);
                w3 = 0; %undefined
                v3 = 0; %undefined
            else
%                 display('Normal motion')
                ex = sign*(omega/norm(omega));
                ey = sign*(cross(omega,omega_dot)/norm(cross(omega,omega_dot)));
                p_perp = cross(omega,v)/(norm(omega)^2); 
                pfirstdot = (cross(omega_dot,v) + cross(omega,v_dot))*norm(omega)^2;
                psecdot = cross(omega,v)*dot(omega,omega_dot);
                p_perp_dot = (pfirstdot-2*psecdot)/(norm(omega)^4);
                
                %time-based invariants
                w1 = dot(omega,ex);
                v1 = dot(v,ex);
                w2 = dot(cross(omega,omega_dot)/(norm(omega)^2),ey);
                v2 = dot(ey,p_perp_dot);                                
                w3 = dot((cross(cross(omega,omega_dot),cross(omega,omega_dotdot)))/(norm(cross(omega,omega_dot))^2),ex);
                num1_1 = cross(omega_dot,cross(omega,omega_dot)) + cross(omega,cross(omega,omega_dotdot));
                num1_2 = (norm(omega)^2 * (cross(omega_dot,v) + cross(omega,v_dot))) - 2*dot(omega,omega_dot)*cross(omega,v);
                den12 = norm(omega)^3 * norm(cross(omega,omega_dot))^2;
                v3first = sign*dot(num1_1,num1_2)/den12;
                num2_1 = cross(omega,cross(omega,omega_dot));
                num2_2 = norm(omega)^2 * (cross(omega_dotdot,v) + cross(2*omega_dot,v) + cross(omega,v_dotdot)) - 2*(norm(omega_dot)^2 + dot(omega,omega_dot))*cross(omega,v);
                v3second = sign*dot(num2_1,num2_2)/den12;
                v3third1 = (3/2*(dot(omega,omega_dot))/(norm(omega)^2)) + (dot(cross(omega,omega_dot),cross(omega,omega_dotdot)))/(norm(cross(omega,omega_dot))^2); 
                v3third2num_1 = cross(omega,cross(omega,omega_dot));
                v3third2num_2 = norm(omega)^2 * (cross(omega_dot,v) - cross(omega,v_dot)) - 2*dot(omega,omega_dot)*cross(omega,v); 
                v3third2 = dot(v3third2num_1,v3third2num_2)/den12;
                v3 = v3first + v3second + sign*v3third1*v3third2;
            end
            end
            
            if isnan(w1),w1=0;end
            if isnan(v1),v1=0;end
            if isnan(w2),w2=0;end
            if isnan(v2),v2=0;end
            if isnan(w3),w3=0;end
            if isnan(v3),v3=0;end
            tb_inv(:,i) = [w1;v1;w2;v2;w3;v3];
    end
    subplot(3,1,2);
    plot(x,tb_inv); title('tb_inv')
    
    
    
    %Check if time-based invariants are correct (reconstruct original trajectory)
    Hm_rec = zeros(4,4,size(tb_inv,2));
    Hm_rec(:,:,1) = eye(4);
    Xm_rec = zeros(7,size(tb_inv,2));
    for jj=2:size(tb_inv_m,2)
           delta_T = eye(4);
           twist_ISA = [tb_inv_m(5,jj);tb_inv_m(3,jj);0;tb_inv_m(6,jj);tb_inv_m(4,jj);0];
           delta_p = twist_ISA(4:6)*jj;
           if norm(twist_ISA(1:3))==0
               e_eq = [0;0;0];
           else
               e_eq = twist_ISA(1:3)/norm(twist_ISA(1:3));
           end
           delta_theta = norm(twist_ISA(1:3))*jj;
           delta_T(1:3,1:3) = skewexp(e_eq,delta_theta);
           delta_T(1:3,4) = delta_p;
           T = Hm_rec(:,:,jj-1)*delta_T;
           S = ad(T);
           twist_ref = [tb_inv_m(2,jj);0;0;tb_inv_m(1,jj);0;0];
           twist = S*twist_ref;
           Hm_rec(:,:,jj) = twistexp(twist); 
           Xm_rec(1:3,jj) = Hm_rec(1:3,4,jj);
           Xm_rec(4:7,jj) = quaternion(Hm_rec(1:3,1:3,jj),true);
    end
    
    subplot(3,1,3);
    tit2 = strcat(num2str(Mtype),'reconstructed');
    plot(x,Xm_rec); title(tit2)
    
    
    TB_INV{j,1} = tb_inv;
end
%% Convert Homogenous Rigid Motion Representation (H=4x4) to Twist (xi*theta=6)
for j=1:length(Hn)
H = Hn{j,1};
twTh = zeros(6,size(H,3));
for i=1:length(H)
    T = H(:,:,i);
    [xi theta] = homtotwist(T);
    twTh(:,i) = [xi*theta];
end
TwistTh{j,1} = twTh;
end

%% Test conversion to twist and back
twth = TwistTh{3,1};
Hback = zeros(4,4,length(twth));
Hback(:,:,1) = eye(4);
for i=2:length(Hback)
%     Hback(:,:,i) = twistexp(twth(1:6,i),twth(7,i));
    Hback(:,:,i) = twistexp(twth(1:6,i));
end

named_figure('Trajectory',1);
clf;
drawframetraj(Hback,2,1);


%% Convert Homogenous Rigid Motion Representation (H=4x4) to Relative Twist (xi,theta=7)
for j=1:length(Hn)
H = Hn{j,1};
twTh = zeros(6,size(H,3));
Hrel = zeros(4,4,size(H,3));
for i=1:length(H)
    if i==1
        T = H(:,:,i);
        Hrel(:,:,i) = T;
        [xi theta] = homtotwist(T);
        twTh(:,i) = [xi*theta];
    else
        T1 = H(:,:,i-1);
        T2 = H(:,:,i);
        T12 = inv(T1)*T2;
        Hrel(:,:,i) = T12;
        [xi theta] = homtotwist(T12);
        twTh(:,i) = [xi*theta];
    end
end
RelTwistTh{j,1} = twTh;
RelH{j,1} = Hrel;
end
%% Test relative transformations
Hrel = RelH{1,1};
Hback_rel = zeros(4,4,size(Hrel,3));
for j=1:length(Hrel)
    if j==1
        Hback_rel(:,:,j) = Hrel(:,:,1);
    else
        Hrel1 = Hback_rel(:,:,j-1);
        Hrel12 = Hrel(:,:,j);
        Hback_rel(:,:,j) = Hrel1*Hrel12;
    end
end
named_figure('Trajectory',2);
clf;
drawframetraj(Hback_rel,2,1);
%% Test conversion to relative twist and back
rel_tw = RelTwistTh{1,1};
Hback = zeros(4,4,length(rel_tw));
Hback(:,:,1) = twistexp(rel_tw(:,1));
for i=2:length(Hback)
%     Hback(:,:,i) = twistexp(twth(1:6,i),twth(7,i));
    Hrel1 = Hback(:,:,i-1);
    Hrel12 = twistexp(rel_tw(:,i));
    Hback(:,:,i) = Hrel1*Hrel12;
end

named_figure('Trajectory',1);
clf;
drawframetraj(Hback,2,1);

%% Preprocess for segmentation (From MoCap data pre-processing experiment)

% RobotChannelNames = {'EndEffector.x', 'EndEffector.y', 'EndEffector.z', ... 
%     'EndEffector.qw', 'EndEffector.qx', 'EndEffector.qy','EndEffector.qz'};
RobotChannelNames = {'EndEffector.v1','EndEffector.v2','EndEffector.v3', ...
    'EndEffector.w1', 'EndEffector.w3', 'EndEffector.w3'};


% nDim = 8;
nDim = 6;
Preproc.nObj = N;
Preproc.obsDim = nDim;
Preproc.R = 1;
Preproc.windowSize = nDim;
Preproc.channelNames = RobotChannelNames;

% Create data structure
Robotdata = SeqData();
% RobotdataN = SeqData();

% Create data structure for PR2 Code
% data_struct = struct;

% Read in every sequence, one at a time
% for ii = 1:length( Xn )
for ii = 1:length(TB_INV)
%     
%     D = Xn{ii};   
%     Dn = XnNoise{ii};
    
    D = TB_INV{ii}(:,:);   
    D = D';    
    D = bsxfun( @minus, D, mean(D,2) );
    D = D';
%     
%     Dn = XnNoise{ii}(1:7,:);
%     % Enforce zero-mean
%     Dn = Dn';
%     Dn = bsxfun( @minus, Dn, mean(Dn,2) );
%     Dn = Dn';
    
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

RobotdataAR = Robotdata; % keeping odata around lets debug with before/after
% RobotdataARN = RobotdataN;
RobotdataAR = ARSeqData( Preproc.R, Robotdata);
% RobotdataARN = ARSeqData( Preproc.R, RobotdataN);

%% Visualizations

named_figure('skew');
for i=1:10, 
  clf;
  drawskewtraj(randskew(),0:pi/20:1/2*pi);
  nice3d; 
end

named_figure('twist');
for i=1:10,
  clf;
  drawtwisttraj(randtwist(),0:pi/20:1/2*pi);
%   nice3d; 
end

named_figure('3 DOF robot');
clf;
r = robot({randtwist('r');randtwist('t');randtwist('s')},randhom());
fk = fkine(r, [0:pi/50:pi ; 0:pi/25:2*pi ; 0:pi/25:2*pi]);
animframetraj(fk);

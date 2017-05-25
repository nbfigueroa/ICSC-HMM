function [X] = generateBehaviors( type, time)
% INPUTS ----------------------------------------------------------
%    type = # of behavior type
%    T = length of each time series
% OUTPUT ----------------------------------------------------------
%    X  :  behavior data

nDim = 7; % X,Y,Z,qw,qi,qj,qk,O
% X = zeros( nDim, time );
time = time - 1;
dt = randsample(10,1);
% length = poissrnd(50);
% length = randsample(50,1);
length = 50;
q = [1 0 0 0]';
OpenPerc=0.05;

%Random Coordinate frame Rotation
%Choose axis
CoordRot=eye(3);
axis = [0,1,2,3];% 0-no,1-x,2-y,z-3
randAxis = axis(randsample(4,1));
angle = pi/2;
if randAxis>0
    if randAxis==1
        CoordRot = CoordRot*rotx(angle);
    end
    if randAxis==2
        CoordRot = CoordRot*roty(angle);
    end
    if randAxis==3
        CoordRot = CoordRot*rotz(angle);
    end
end


switch type
    case 1
        disp('Hyperbolic Parabloid');
        x = linspace(0,length, time);
        y = linspace(0,length, time);
        [Xm,Ym] = meshgrid(x,y);
%         if round(rand(1,1)) == 1
%             i=1;j=-1;
%         else
%             i=-1;j=1;
%         end
        i=1;j=-1;
        F =  i*(Ym.^2) + j*(Xm.^2);
        f = F(:,randsample(10,1));
%         X(1,:) = x/100; X(2,:) = y/100; X(3,:) = (f/length)/100; 
%         X(1,:) = x; X(2,:) = y; X(3,:) = (f/length); 
        X(1,:) = x; X(2,:) = y; X(3,:) = f/length; 
        
        %Coordinate frame rotation
        for r=1:size(X,2)
            X(1:3,r) = CoordRot*X(1:3,r);
        end
        
        % Rotations
        for i=1:size(X,2)
%             if i==1
%                 R = eye(3,3);
%             else
            if i==(size(X,2))
%                 R = rpy2r([X(4,i-1),X(5,i-1),X(6,i-1)]);
                R = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
            else
                x = X(1,i+1)-X(1,i); y = X(2,i+1)-X(2,i); z = X(3,i+1)-X(3,i);
                xzlen = sqrt(x*x + z*z);
                beta = acosh(y/xzlen);
                veclen = sqrt(x*x + y*y + z*z);
                alpha = acosh(xzlen/veclen);
                R = rotationMatrx('x',real(alpha))*rotationMatrx('y',real(beta));
            end
%             T = r2t(R);
%             rpyaw = (tr2rpy(T)*length)/(2*pi);
%             X(4,i) = rpyaw(1); X(5,i)=rpyaw(2); X(6,i)=rpyaw(3);
            q = quaternion(R, true);    
            X(4,i) = q(1)*length; X(5,i) = q(2)*length; X(6,i) = q(3)*length; X(7,i) = q(4)*length;
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
            
        end
        % Gripper Opening
        Op = ones(1,size(X,2));
%         Opinit = 0:1/floor(size(X,2)*OpenPerc):1;
%         Opend = 1:-1/floor(size(X,2)*OpenPerc):0;
%         Op(1,1:size(Opinit,2)) = Opinit;
%         Op(1,end-(size(Opend,2)-1):end) = Opend;
%         X(8,:)=Op*length;
%         X(8,:)=Op;
    case 2
%         disp('Hyperbolic Parabloid');
        disp('Line');
        x = linspace(0,length, time);
        y = linspace(0,length, time);
        [Xm,Ym] = meshgrid(x,y);
%         if round(rand(1,1)) == 1
%             i=1;j=-1;
%         else
%             i=-1;j=1;
%         end
        i=-1;j=1;
        F =  i*(Ym.^2) + j*(Xm.^2);
        f = F(:,randsample(10,1));
%         X(1,:) = x/100; X(2,:) = y/100; X(3,:) = (f/length)/100; 
        z = zeros(1,size(x,2));
        X(1,:) = x; X(2,:) = z; X(3,:) = z; 
        
        %Coordinate frame rotation
        for r=1:size(X,2)
            X(1:3,r) = CoordRot*X(1:3,r);
        end
        
        % Rotations
        for i=1:size(X,2)
%             if i==1
%                 R = eye(3,3);
            if i==(size(X,2))
%                 R = rpy2r([X(4,i-1),X(5,i-1),X(6,i-1)]);
                R = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
            else
                x = X(1,i+1)-X(1,i); y = X(2,i+1)-X(2,i); z = X(3,i+1)-X(3,i);
                xzlen = sqrt(x*x + z*z);
                beta = acosh(y/xzlen);
                veclen = sqrt(x*x + y*y + z*z);
                alpha = acosh(xzlen/veclen);
                R = rotationMatrx('x',real(alpha))*rotationMatrx('y',real(beta));
            end
%             T = r2t(R);
%             rpyaw = (tr2rpy(T)*length)/(2*pi);
%             X(4,i) = rpyaw(1); X(5,i)=rpyaw(2); X(6,i)=rpyaw(3);
            q = quaternion(R, true);    
            X(4,i) = q(1)*length; X(5,i) = q(2)*length; X(6,i) = q(3)*length; X(7,i) = q(4)*length;
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
        end
        % Gripper Opening
        Op = ones(1,size(X,2));
%         Opinit = 0:1/floor(size(X,2)*OpenPerc):1;
%         Opend = 1:-1/floor(size(X,2)*OpenPerc):0;
%         Op(1,1:size(Opinit,2)) = Opinit;
%         Op(1,end-(size(Opend,2)-1):end) = Opend;
%         X(8,:)=Op*length;
    case 3
        disp('Helixoid');     
        length = length/2;
        ts = length/time;
        T = length;
        t = 0:ts:T;        
%         x=sin(t)*length/randsample(3,1); y=cos(t)*length/randsample(3,1); z=t;
        x=sin(t)*length/3; y=cos(t)*length/3; z=t;
%         X(1,:) = x/100; X(2,:) = y/100; X(3,:) = z/100; 
        X(1,:) = x; X(2,:) = y; X(3,:) = z; 
        %Coordinate frame rotation
        for r=1:size(X,2)
            X(1:3,r) = CoordRot*X(1:3,r);
        end
        
        % Rotations
        for i=1:size(X,2)
%             if i==1
%                 R = eye(3,3);
            if i==(size(X,2))
%                 R = rpy2r([X(4,i-1),X(5,i-1),X(6,i-1)]);
                R = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
            else
                x = X(1,i+1)-X(1,i); y = X(2,i+1)-X(2,i); z = X(3,i+1)-X(3,i);
                xzlen = sqrt(x*x + z*z);
                beta = acosh(z/xzlen);
                veclen = sqrt(x*x + y*y + z*z);
                alpha = acosh(xzlen/veclen);
                R = rotationMatrx('x',real(alpha))*rotationMatrx('y',real(beta));
            end
%             T = r2t(R);                
%             rpyaw = (tr2rpy(T)*length)/(2*pi);
%             X(4,i) = rpyaw(1); X(5,i)=rpyaw(2); X(6,i)=rpyaw(3);
%             end
            q = quaternion(R, true);    
            X(4,i) = q(1)*length; X(5,i) = q(2)*length; X(6,i) = q(3)*length; X(7,i) = q(4)*length;
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
        end
        % Gripper Opening
        Op = ones(1,size(X,2));
%         Opinit = 0:1/floor(size(X,2)*OpenPerc):1;
%         Opend = 1:-1/floor(size(X,2)*OpenPerc):0;
%         Op(1,1:size(Opinit,2)) = Opinit;
%         Op(1,end-(size(Opend,2)-1):end) = Opend;
%         X(8,:)=Op*length;
%     case 3
%         disp('Cubic'); %Parameters set for cloverleaf
%         length = length;
%         p2 = 3/8*pi; % Coefficients of the cubic
%         p3 = -1/(16*pi^2); % polynomial for gamma(t).
% %         length = poissrnd(100);
%         dtc = length/time;
%         tf = length;        
% %         tf = (time-1)/dt;
% %         dtc = 1/dt;
% %         dtc = ts;
%         ti = 0; % ti is the initial time (zero)
%         % tf = 12.576; % tf is the final time (4*pi seconds)
%         % dt = 0.016; % dt is the time step (0.016 seconds)
%         t = [ti:dtc:tf]'; % Create time vector as column vector (take transpose)
%         gamma = polyval([p3 p2 0 0],t); % Evaluate cubic polynomial for gamma
% %         a = 0.0625; % Parameter a in meters
% %         b = 0.25; % Parameter b in meters
%         a = 6.25; % Parameter a in meters
%         b = 25; % Parameter b in meters
% %         a = 1;
% %         b = 10;
%         x = (b-a)*sin(t)+3*a*sin((t/a)*(b-a)); % x coordinate in meters
%         z = 0.5+(b-a)*cos(t)-3*a*cos((t/a)*(b-a)); % z coordinate in meters
%         % figure;plot3(x,z,t);
%         x=x';z=z';
%         y = zeros(1,size(x,2));
%         X(1,:) = x; X(2,:) = y; X(3,:) = z; 
%         % Rotations
%         for i=1:size(X,2)
%             if i==(size(X,2))
%                 R = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
%             else
%                 x = X(1,i+1)-X(1,i); y = X(2,i+1)-X(2,i); z = X(3,i+1)-X(3,i);
%                 xzlen = sqrt(x*x + z*z);
%                 beta = acos(z/xzlen);
%                 veclen = sqrt(x*x + y*y + z*z);
%                 alpha = acos(xzlen/veclen);
%                 R = rotationMatrx('x',alpha)*rotationMatrx('y',beta);
%             end
%             q = quaternion(R);    
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
%         end
    case 4
        disp('2D Sign');
        length = length/2;
%         T = (time-1)/dt;
%         ts = 1/dt;
%         length = poissrnd(100);
        ts = length/time;
        T = length;
        t = 0:ts:T;
%         x = square(t)*length; % Create square signal.
        x = sin(t)*length;
%         if round(rand(1,1)) == 1
%             i=1;
%         else
%             i=-1;
%         end
%         z = i*x;
        z = zeros(1,size(x,2));
%         plot3(t,x,z) % Plot both signals.\
%         X(1,:) = t/100; X(2,:) = x/100; X(3,:) = z/100; 
        X(1,:) = t; X(2,:) = x; X(3,:) = z; 
        
        %Coordinate frame rotation
        for r=1:size(X,2)
            X(1:3,r) = CoordRot*X(1:3,r);
        end
        
        % Rotations
        for i=1:size(X,2)
%             if i==1
%                 R = eye(3,3);
%             else
            if i==(size(X,2))
%                 R = rpy2r([X(4,i-1),X(5,i-1),X(6,i-1)]);
                R = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
            else

                x = X(1,i+1)-X(1,i); y = X(2,i+1)-X(2,i); z = X(3,i+1)-X(3,i);
                xzlen = sqrt(x*x + z*z);
                beta = acosh(z/xzlen);
                veclen = sqrt(x*x + y*y + z*z);
                alpha = acosh(xzlen/veclen);
                R = rotationMatrx('x',real(alpha))*rotationMatrx('y',real(beta));
            end
%             T = r2t(R);                
%             rpyaw = (tr2rpy(T)*length)/(2*pi);
%             X(4,i) = rpyaw(1); X(5,i)=rpyaw(2); X(6,i)=rpyaw(3);
            q = quaternion(R);    
            X(4,i) = q(1)*length; X(5,i) = q(2)*length; X(6,i) = q(3)*length; X(7,i) = q(4)*length;
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
        end
        % Gripper Opening
        Op = ones(1,size(X,2));
%         Opinit = 0:1/floor(size(X,2)*OpenPerc):1;
%         Opend = 1:-1/floor(size(X,2)*OpenPerc):0;
%         Op(1,1:size(Opinit,2)) = Opinit;
%         Op(1,end-(size(Opend,2)-1):end) = Opend;
%         X(8,:)=Op*length;
        
    case 5 
        disp('Screw')
        theta = 1; % our rotation speed [Hz]
        beta = pi/4; % [rad]
        gamma = pi/4; % [rad]

        b = 0.1; % [T] field strength

        % elapsed_time = 2; % [s]
        % delta = 0.05; % time resolution [s]

        elapsed_time = length;
        delta = length/time;

        omega = [0;0;0;1;0;0]; % our rotation axis for the magnetic field

        v = [1; 0; 0; 0; 0; 0;]; % assume we move forward along the x axis [m/s]
                         % this is to simulate real motion

        H1 = expm(twist(2*pi*theta*omega*delta)); % this is in a local coordiate
                                          % frame based on the last 
                                          % commanded orientation
        H2 = expm(twist([0; 0; 0; 0; beta; 0])); % steering direction
        H3 = expm(twist([0; 0; 0; 0; 0; gamma])); % steering direction
        H4 = expm(twist(v*delta)); % this is to simulate motion in the real world
           
        R = eye(4);

        ix = 0;
        for i=0:delta:elapsed_time,
  
            ix = ix + 1;

            H2 = expm(twist([0; 0; 0; 0; beta; 0])); % steering direction
            H3 = expm(twist([0; 0; 0; 0; 0; gamma])); % steering direction

            R = R*H1; 
            if (1 == ix),
                H(:,:,ix) = H2*H3*R;
            else
                H(:,:,ix) = H2*H3*H4*R;  
                H(1:2,4,ix) = [0;0];
                H(1:3, 4, ix) = H(1:3, 4, ix) + H(1:3, 4, ix-1);
            end    
        end  
        t = H(1:3,4,:);
        X(1:3,:) = reshape(t,[],size(t,3));
        r = H(1:3,1:3,:);
        X(4:7,:) = quaternion(r,true)*length;

%         pos = repmat([1; 1; 1],[1 time]);
%         X(1:3,:) = pos/100; 
%         % Rotations
%         c=1;
%         for i=1:size(X,2)
%             if i==1
%                 R = eye(3,3);
%             else
%                 Rpast = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
%                 Rpast = rpy2r([X(4,i-1),X(5,i-1),X(6,i-1)]);
%                 alpha = c*(180/time)*pi/180;%every i-th degree until max 180 degree
%                 R = Rpast*rotationMatrx('z',alpha);
%             end
% %             T = r2t(R);   
% %             rpyaw = (rpy(R)*length)/(2*pi);
% %             rpy = (tr2rpy(T)*length)/(2*pi);
% %             X(4,i) = rpyaw(1); X(5,i)=rpyaw(2); X(6,i)=rpyaw(3);
%             q = quaternion(R);    
%             X(4,i) = q(1)*length; X(5,i) = q(2)*length; X(6,i) = q(3)*length; X(7,i) = q(4)*length;
% %             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
%         end
%         % Gripper Opening
%         Op = ones(1,size(X,2));
%         Opinit = 0:1/floor(size(X,2)*0.3):1;
%         Opend = 1:-1/floor(size(X,2)*0.3):0;
%         Op(1,1:size(Opinit,2)) = Opinit;
%         Op(1,end-(size(Opend,2)-1):end) = Opend;
%        Op = 0:1/floor(size(X,2)):1;
%        X(8,:)=Op*length;
%     case 6
%         disp('Screw Counter-Clockwise')
%         pos = repmat([1; 1; 1],[1 time]);
%         X(1:3,:) = pos; 
%         % Rotations
%         c=-1;
%         for i=1:size(X,2)
%             if i==1
%                 R = eye(3,3);
%             else
%                 Rpast = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
%                 alpha = c*pi/180;%every 1 degree
%                 R = Rpast*rotationMatrx('z',alpha);
%             end
%             q = quaternion(R);    
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
%         end;
%         
end

end

% 
%         disp('3D Sign');        
% %         T = (time-1)/dt;
% %         ts = 1/dt;
% %         length = poissrnd(100);
%         length = length/2;
%         ts = length/time;
%         T = length;
%         t = 0:ts:T;
% %         x = sawtooth(t)*length; % Create sawtooth signal.
%      
% %         x = sawtooth_w(t)*length;
%         
%         
% %         plot(x, y)
% %         axis([-6 6 -1.5 1.5])
% %         xlabel('x')
% %         ylabel('Sawtooth Wave')
%  
%         
%         x = cos(t)*length; 
%         if round(rand(1,1)) == 1
%             i=1;
%         else
%             i=-1;
%         end
%         
%         z = i*x;
% %         plot3(t,x,z) % Plot both signals.
%         X(1,:) = t; X(2,:) = x; X(3,:) = z; 
%         % Rotations
%         for i=1:size(X,2)
%             if i==(size(X,2))
%                 R = quaternion([X(4,i-1);X(5,i-1);X(6,i-1);X(7,i-1)]);
%             else
%                 x = X(1,i+1)-X(1,i); y = X(2,i+1)-X(2,i); z = X(3,i+1)-X(3,i);
%                 xzlen = sqrt(x*x + z*z);
%                 beta = acos(z/xzlen);
%                 veclen = sqrt(x*x + y*y + z*z);
%                 alpha = acos(xzlen/veclen);
%                 R = rotationMatrx('x',alpha)*rotationMatrx('y',beta);
%             end
%             q = quaternion(R);    
%             X(4,i) = q(1); X(5,i) = q(2); X(6,i) = q(3); X(7,i) = q(4);
%         end
% 
% function y = sawtooth_w(t)
% % We find the period number of every element
% % in the input vector
% tn = ceil((t+pi)/(2*pi)); 
% 
% % and we subtract that corresponding period from
% % the base value. We want final values from -1 to 1
% y = ((t - tn*2*pi) + 2*pi)/pi; 
% end
% %% Hyperbolic Parabloid
% % x = linspace(-4,4);
% % y = linspace(-4,4);
% k = 1;
% time = behaviorsT(behaviorsID(k));
% x = linspace(0,randsample(10,1), time);
% y = linspace(0,randsample(10,1), time);
% [X,Y] = meshgrid(x,y);
% f = (Y.^2) - (X.^2);
% figure;plot3(x(:,:),y(:,:),f(:,10));
% % figure;plot3(x,y,f);
% %% Helixoid
% % timestep = pi/50;
% % T = 10*pi;
% % t = 0:timestep:T;
% % 
% % T = 1000;
% % t = 1:1/10:T/10;
% 
% tf = 1000/10;
% dt = 1/10;
% ti = 0;
% t = [ti:dt:tf];
% x=sin(t); y=cos(t); z=t;
% figure;plot3(x,y,t);
% xlabel('x')
% ylabel('y')
% zlabel('z')
% grid on
% axis square
% 
% %% Cubic
% p2 = 3/8*pi; % Coefficients of the cubic
% p3 = -1/(16*pi^2); % polynomial for gamma(t).
% 
% tf = 1000/10;
% dt = 1/10;
% ti = 0; % ti is the initial time (zero)
% % tf = 12.576; % tf is the final time (4*pi seconds)
% % dt = 0.016; % dt is the time step (0.016 seconds)
% t = [ti:dt:tf]'; % Create time vector as column vector (take transpose)
% gamma = polyval([p3 p2 0 0],t); % Evaluate cubic polynomial for gamma
% a = 0.0625; % Parameter a in meters
% b = 0.25; % Parameter b in meters
% x = (b-a)*sin(t)+3*a*sin((t/a)*(b-a)); % x coordinate in meters
% z = 0.5+(b-a)*cos(t)-3*a*cos((t/a)*(b-a)); % z coordinate in meters
% figure;plot3(x,z,t);
% 
% %% Sawtooth
% T = 10;
% ts = 0.1;
% t = 0:ts:T;
% x = sawtooth(t); % Create sawtooth signal.
% r = randn(T/ts+1,1);
% % y = awgn(x,10,'measured'); % Add white Gaussian noise. 
% y = x + r';
% plot(t,x,t,y) % Plot both signals.
% % plot(t,x) % Plot both signals.
% legend('Original signal','Signal with AWGN');
% 
% %% 3D Sawtooth
% T=time/10;
% % T = 100;
% % ts = 0.1;
% ts = 1/10;
% t = 0:ts:T;
% x = sawtooth(t); % Create sawtooth signal.
% z = x;
% plot3(t,x,z) % Plot both signals.
% 
% %% Square
% T = 10;
% ts = 0.1;
% t = 0:ts:T;
% t = 0:.1:10;
% x = square(t); % Create square signal.
% r = randn(T/ts+1,1);
% % y = awgn(x,10,'measured'); % Add white Gaussian noise. 
% y = x + r';
% plot(t,x,t,y) % Plot both signals.
% legend('Original signal','Signal with AWGN');
% 
% %% 3D Square
% % T = 100;
% % ts = 0.1;
% 
% dt = randsample(20,1);
% T = time/dt;
% ts = 1/dt;
% t = 0:ts:T;
% x = square(t); % Create square signal.
% r = randn(T/ts+1,1);
% % y = awgn(x,10,'measured'); % Add white Gaussian noise. 
% % y = x + r';
% % plot(t,x) % Plot both signals.
% z = zeros(1,size(x,2));
% plot3(t,x,z) % Plot both signals.
% % legend('Original signal','Signal with AWGN');
% 
% 
% %% 3D Graphics - line plot
%     % 3D Graphics - line plot
%     % Dr. P.Venkataraman
%     % The path traced by a particle in 3D space
% 
%     format compact
%     set(gcf,'Menubar','none','Name','3D Line Plot - Path and Velocity', ...
%              'NumberTitle','off','Position',[10,350,400,300]);
%     t = 0:0.1:3;
% 
%     x = 2*sin(2*t);
%     y = 3*cos(2*t);
%     z = t;
%     h1 = plot3(x,y,z, ...
%         'LineWidth',2, ...
%         'Color','k');
%     % animation -----
%     axis([-3 3 -5 6 0 4]);
%     hold on
%     for i=1:length(t)
%         if i> 1
%             delete(h2); % deletes the marker
%             delete(h3); % delete the line
%         end
%         % creats a filled patch of yellow color
%         h2 = plot3(x(i),y(i),z(i),'bo','MarkerFaceColor','r'); % creates a marker
%         h3 = line([x(i) x(i)],[y(i) y(i)],[0 z(i)],'Color','r');
%         pause(0.1) % pauses the plot or you can't see the animation
%     end
%     delete(h3);
%     %-------------------------------------
%     % title in bold face italic in black color
%     title('\bf\it3D Line Plot and Velocity Vector','Color','k', ...
%              'VerticalAlignment','bottom')
%     ylabel('y', ...
%         'FontWeight','b','Color','b', ...
%         'VerticalAlignment','bottom');  % boldface blue
%     xlabel('x', ...
%          'FontWeight','b','Color','b', ...
%          'VerticalAlignment','bottom');  % boldface blue
%     zlabel('z', ...
%          'FontWeight','b','Color','b', ...
%          'VerticalAlignment','bottom');  % boldface blue
% 
%     quiver3(x,y,z,4*cos(4*t),-6*sin(t),1,2,'b')
%     hold off
% 
%     textstr(1)={'x = 2Sin(t)'};
%     textstr(2)={'y = 3Cos(t)'};
%     textstr(3)={'z = t'};
%     text(1.5,4,2,textstr,'FontWeight','b')
%     grid

function mats = getmats(r,q,p0,sampleT,coord_dim,model_type)

% Define system matrices:

switch coord_dim
    case 1
        switch model_type
            case '1D'
                mats.A = 1;
                mats.invA = 1;
                mats.B = 1;
                mats.C = 1;
                mats.Q = q;
            case 'CV'
                mats.A = [1 sampleT; 0 1];
                mats.invA = inv(mats.A);
                mats.B = [0.5*sampleT^2; sampleT];
                mats.C = [1 0]; %[1 0; 0 1];
                mats.Q = q; %q*[1 0 ; 0 1];
            case 'CA'
                mats.A = [1 sampleT 0.5*sampleT^2; 0 1 sampleT; 0 0 1];
                mats.invA = inv(mats.A);
                mats.B = [0.5*sampleT^2; sampleT; 0];
                mats.C = [1 0 0]; %[1 0 0; 0 1 0];
                mats.Q = q*[1 0 0; 0 1 0; 0 0 1];
        end
    case 2
        switch model_type
            case 'CV'
                mats.A = [1 0 sampleT 0; 0 1 0 sampleT; 0 0 1 0; 0 0 0 1];
                mats.invA = inv(mats.A);
                mats.B = [0.5*sampleT^2 0; 0 0.5*sampleT^2; sampleT 0; 0 sampleT];
                mats.C = [1 0 0 0; 0 1 0 0];
            case 'CA'
                mats.A = [1 0 sampleT 0 0.5*sampleT^2 0; 0 1 0 sampleT 0 0.5*sampleT^2; 0 0 1 0 sampleT 0; 0 0  0 1 0 sampleT; 0 0 0 0 1 0; 0 0 0 0 0 1];
                mats.invA = inv(mats.A);
                mats.B = [0.5*sampleT^2 0; 0 0.5*sampleT^2; sampleT 0; 0 0.5*sampleT^2; 1 0; 0 1];
                mats.C = [1 0 0 0 0 0; 0 1 0 0 0 0];
        end
end

dimx = size(mats.A,1);
dimu = size(mats.B,2);
dimy = size(mats.C,1);
mats.Lambda_q = inv(mats.Q);  % process noise in information form
mats.Lambda_r = diag(1./r); % measurement noise in information form
mats.P0 = p0*eye(dimx);    % initial state covariance

return;
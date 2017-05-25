%Returns controllability gramian P for an unstable system and the transformation Tr that splits A
%into stable and unstable parts.
%This program uses the results from the following two publications: 

%[1] K. Zhou, G. Salomon and E. Wu, 'Balanced realization and model
%reduction for unstable systems", International Journal of Robust and
%Nonlinear Control, vol. 9, pp. 183 - 198, 1999.

%[2] S.K. Nagar and S.K. Singh, 'An algorithmic approach for system 
%decomposition and balanced realized model reduction', Journal of the
%Franklin Institute vol 341, pp. 615630, 2004.

%Last updated: 24/09/2012
%Email: chrislbowden@hotmail.com

function [P,Tr] = CtrGram(A,B)

%Check dimensions
[n n0]   = size(A);
if (n ~= n0)
    error('CtrGram:Dim','The state matrix A must be a square matrix')
end
[n1 p1] = size(B);
if (n1 ~= n)
    error('CtrGram:Dim','A and B must have the same number of rows')
end

%Check the number of positive, zero and negative eigenvalues of A
Evals = eig(A);
Pos = 0;
Neg = 0;
Zer = 0;
for i = 1:n
    if real(Evals(i)) > 0
        Pos = Pos +1;
    end 
    if real(Evals(i)) < 0
        Neg = Neg +1;
    end
    if real(Evals(i)) == 0
        Zer = Zer +1;
    end
end

%This method is only valid for systems without poles on the imaginary axis
if (Zer ~= 0)
    error('CtrGram:ImagAxisPole','The state matrix has imaginary axis poles. ')
end
%If A is stable then the function 'gram' from the control system toolbox
%can be used.
if Neg == n
    warning('CtrGram:StabMat','The state matrix is stable. Gramian could also be computed using control system toolbox.')
end

    
%The transformation T in Theorem 1 in Zhou (1999) can be obtained using the
%algorithm of Nagar and Singh (2004)
%Find the Schur form:
[U,T] = schur(A);

%Reorder the Schur factorization A = U*T*U' of  matrix A so that the 
%negative (i.e. stable) cluster of eigenvalues appears in the  
%leading (upper left) diagonal blocks of the Schur matrix T 
%(Nagar and Singh eqn (2))
[US,TS] = ordschur(U,T,'lhp');

%Solve lyapunov equation - Nagar and Singh eqn(3)]
%This will enable us to decouple the system in order to separate into 
%stable and unstable components
A11     =   TS(1:Neg,1:Neg);
A12     =   TS(1:Neg,Neg+1:n);
A22     =   TS(Neg+1:n,Neg+1:n);
S       =   lyap(A11,-A22,A12);

%Use W and it's inverse to find decoupled system - Nagar and Singh eqn(4,5)
W       =   [eye(Neg) S; zeros(n-Neg,Neg) eye(n-Neg)];
Winv    =   [eye(Neg) -S; zeros(n-Neg,Neg) eye(n-Neg)];

%The above can be used to obtain the transformation T (labelled T1 here) in
%Zhou 1999, Theorem 1.
T1      = Winv*US';
T1inv   = US*W;

%To get parts of B corresponding to transformed stable and unstable system:
BB=T1*B;
B1=BB(1:Neg,1:p1);
B2=BB(Neg+1:n,1:p1);

%Gd=[A11 0; 0 A22] where A11 is stable and A22 is unstable
Gd=T1*A*T1inv;

%As and B1 form stable part, Au and Bu form unstable part
As = Gd(1:Neg,1:Neg);
Au = Gd(Neg+1:n,Neg+1:n);

%P1 and P2 are controllability gramians of (As,Bs) and (-Au,Bu)
%- Zhou (1999) p187
P1 = lyapchol(As,B1)'*lyapchol(As,B1);
P2 = lyapchol(-Au,B2)'*lyapchol(-Au,B2);

%P is controllability Gramian of the system, the 'larger' P the
%smaller the required control energy  
%- Zhou (1999) p187 and p195  

P = T1inv*[P1 zeros(Neg,n-Neg); zeros(n-Neg,Neg) P2]*T1inv';
Tr=T1;
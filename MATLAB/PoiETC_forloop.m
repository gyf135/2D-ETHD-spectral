close all; clear variables; clc
% Solve 2D electroconvection in a channel using the Fourier-cheb method
% Navier-Stokes equation is in the vorticity-stream function form
% In the simulation omega is negative vorticity
%% Preprocessing
% addpath('SC\');
addpath('C:\Users\yy\Box\Make video from file\b2r')
addpath('C:\Users\yy\Box\Make video from file\SC')
tic
readTrue = 0;
saveTrue = 1;
NSAVE = 5;
maxit = NSAVE*1;
N = 679; % N+1 has to be even
dt = 1e-5;
% nu = 2.5;%0.025;
% rho = 0.16;
% eps = 1e-4;%1e-4;
% nue = 2.5e-3;%2.5e-5;
mub = 1.0;% Non-dimensionlized %2.5e-3;
umax = 0.0; % This is a free parameter for
umax_sim = umax*1/3;
uw  = umax*2/3; % Velocity at the boundary in the simulation; For Poiseuille flow, umax = -0.5uwall;
%if uw = -1*(2/3), it means that if u = 0 at walls, umax = 1, but in fact,
%uwall = -2/3 and umax(u(y=0)) = 1/3. Also in this case, Q = 4/3, but in
%fact Q = 0; Re = 3Q/4nu = 1/nu. In general, setting uw means setting umax,
%and Q = 4/3*umax [in simulation Q = 0 and the velocity field shift for
%that]. And Re = 3Q/4nu
Q = 4/3*umax; % Flow rate before velocity shift; in simulation Q = 0
uwc = 1;%2.5;
uwv = 1;%1e4;%1e4;
% Re = 2*rho*uw/nu; % Couette definition
% Re = rho*3/2*uw/nu; % Poiseuille definition

% M  = sqrt(eps/rho)/mub
% T  = eps*uwv/(nu*rho*mub) % Rayleight number
% C  = uwc/eps/uwv*4
% Fe = mub*uwv/nue

M = 10;
T = 3000;
C = 2.5*4; % (C corresponding to H = 1, Del_phi = 1 is 4x this)
Fe = 2000;
Ra = 1e9; % (Ra corresponding to H = 1, Del_T = 1 is 16x this)
Pr = 100;

% modeling parameters
nu = M^2/T;
kT = M^2/T/Pr;
Re = 1/nu
ForceE = C*M^2/4;
ForceT = M^4*Ra/T^2/Pr/16;
De = 1/Fe;
eps = 4/C;
DT = M^2/T/Pr;

% Create the Chebyshev-nodes (ascending order)
x = -cos(pi*(0:N)/N);
y = x';
Lx = 4;%2*pi;
NX = 1360;%N*Lx/2;
dx = Lx/NX;
x = linspace(0,Lx-dx,NX);
kx = (2*pi/Lx)*[0:(NX/2) (-NX/2+1):-1];
Kx = ones(N+1,1) * kx;

uwall  = fft(ones(1,NX)*uw);
uwallc = fft(ones(1,NX)*uwc);
uwallv = fft(ones(1,NX)*uwv);

% uwall = -uwall*sin(pi*x);
% uwall = -0.5*uwall*cos(2*pi*x)+0.5;
% Create the tensor product mesh
[X, Y] = meshgrid(x,y);
% Y_hat = fft(Y,[],2)*uw*3/2;
% Build the differentiation matrices
% D = dermatrix(x,2); % This is same as cheb.m
% D1 = D{1};
% D2 = D{2};
D1 = -cheb(N); % negative for ascending order y; 2 for y \in [0,1]
% D2 = D1*D1;
% D4 = D2*D2;

% Boundary points
xx=X(:); yy=Y(:); kkxx = Kx(:);
t_pts  = find(yy==1);
b_pts = find(yy==-1);
bd_pts = find(yy==-1 | yy==1);
i_pts  = find(yy~=-1 & yy~=1);

% Operators in q_hat
I = speye(NX);

spY2 = spdiags(1-y.^2,0,N+1,N+1);
spY = spdiags(y,0,N+1,N+1);

D4y = (spY2*D1^4 - 8*spY*D1^3 - 12*D1^2); % Trefethen style

% spY2 = spdiags((y-1).*y,0,N+1,N+1);
% spY = spdiags(8*y-4,0,N+1,N+1);
%
% D4y = -4*(spY2*D1^4 + spY*D1^3 + 12*D1^2); % Trefethen style

spYY2 = spdiags(1-yy.^2,0,(NX)*(N+1),(NX)*(N+1));

Dxx = spdiags((1i*kkxx).^2,0,(NX)*(N+1),(NX)*(N+1));
Dyy = kron(I,D1^2);
Dc2 = Dxx*(Dyy*spYY2);
Dx4 = spdiags((1i*kkxx).^4,0,(NX)*(N+1),(NX)*(N+1))*spYY2;
Dy4 = kron(I,D4y);

qDel2 = Dyy*spYY2 + spYY2*Dxx;
qDel4 = Dx4  +   Dy4 +   2*Dc2;

% Operators in psi
psiDx = spdiags((1i*kkxx),0,(NX)*(N+1),(NX)*(N+1));
psiDy = kron(I,D1);
psiD4 = D1^4;
psiDel2 = Dxx+Dyy;
psiDel4 = spdiags((1i*kkxx).^4,0,(NX)*(N+1),(NX)*(N+1))+...
    kron(I,psiD4)+2*Dxx*Dyy;
% ==========================================================================================
%                                   OPERATORS
% ==========================================================================================
% Left hand side operator for q
% LHS = qDel2 - (0.5*dt*nu) * qDel4;
LHS = qDel2 - (0.5*dt*M^2/T) * qDel4;
LHS(bd_pts,:) = 0; % Dirichlet boundaries
LHS(bd_pts, bd_pts) = speye(2*NX);
% LHS = full(LHS);
% LHS = inv(LHS);

% LHSC = speye((N+1)*NX) - 0.5*dt*nue*psiDel2;
LHSC = speye((N+1)*NX) - 0.5*dt/Fe*psiDel2;
LHSC(b_pts,:) = 0; % Dirichlet boundaries
LHSC(b_pts, b_pts) = speye(NX);
LHSC(t_pts,:) = psiDy(t_pts,:); % Neumann boundaries
% LHSC = inv(LHSC);

LHSV = psiDel2;
LHSV(bd_pts,:) = 0; % Dirichlet boundaries
LHSV(bd_pts, bd_pts) = speye(2*NX);
% LHSV = inv(LHSV);

LHST = speye((N+1)*NX) - 0.5*dt*kT*psiDel2;
LHST(bd_pts,:) = 0; % Dirichlet boundaries
LHST(bd_pts, bd_pts) = speye(2*NX);

% Break up LHS, psiDy,psiDel2,psiDel4
partition = NX/2+1;
NP = 1;%NX/partition;
for p = 1:partition
    LHS_np{p} = inv(full(LHS((p-1)*NP*(N+1)+1:p*NP*(N+1),(p-1)*NP*(N+1)+1:p*NP*(N+1))));
    LHSC_np{p} = inv(full(LHSC((p-1)*NP*(N+1)+1:p*NP*(N+1),(p-1)*NP*(N+1)+1:p*NP*(N+1))));
    LHSV_np{p} = inv(full(LHSV((p-1)*NP*(N+1)+1:p*NP*(N+1),(p-1)*NP*(N+1)+1:p*NP*(N+1))));
    LHST_np{p} = inv(full(LHST((p-1)*NP*(N+1)+1:p*NP*(N+1),(p-1)*NP*(N+1)+1:p*NP*(N+1))));
    psiDel2_np{p} = full(psiDel2((p-1)*NP*(N+1)+1:p*NP*(N+1),(p-1)*NP*(N+1)+1:p*NP*(N+1)));
    psiDel4_np{p} = full(psiDel4((p-1)*NP*(N+1)+1:p*NP*(N+1),(p-1)*NP*(N+1)+1:p*NP*(N+1)));
end
psiDy_np = full(psiDy(1:N+1,1:N+1));
LHS = [];
LHSC = [];
LHSV = [];
LHST = [];
psiDel2 = [];
psiDel4 = [];
psiDy = [];

save('OperatorsETC.mat','LHS_np','LHSC_np','LHSV_np','LHST_np','psiDel2_np','psiDel4_np','psiDy_np','-v7.3');
% load('OperatorsEC.mat');
% ==========================================================================================

% Initialize
if readTrue==0
    % Initial condition
    % u = cos(pi*X).*sin(pi*Y) + 2*Y;
    % psi = -1/pi*cos(pi*Y)cos(pi*X) + Y.^2 + C
    
    psi  = zeros((N+1),(NX));% + randn((N)*(N+1),1)/100;
    q    = 0*psi;
    q_psi = zeros((N+1),NX);
    % Add perturbation (Couette)
    %     q = cos(2*pi/Lx*X).*sin(pi*Y) - 0.5*uw;
    %     psi = (1-Y.^2).*q;
    % Add perturbation (Poiseuille)
    %         q = 0.5*uw*Y + 0.2*cos(Lx/2*2*pi/Lx*X).*sin(pi*Y);
    q = 0.5*uw*Y + 1e-3*cos(2*pi/Lx*X).*sin(pi*Y); % uupperwall = 0.5uw; ulowerwall = -0.5uw
%     q = 0.5*uw*Y + 0*1e-3*randn(N+1,NX).*(1-Y).*(1+Y); % uupperwall = 0.5uw; ulowerwall = -0.5uw
    psi = (1-Y.^2).*q;
    
    % Add perturbation (RBC)
    %     q = uw*(Y-0.5) + 1*cos(2*pi/Lx*X).*sin(pi*(Y));
    %     %     q = 0.5*uw*Y + 1*cos(2*pi/Lx*X).*sin(0.5*pi*(Y+1));
    %     psi = (1-Y.^2).*q;
    
    %         C = 0.5*uwc*(-Y+1) + 0.00001*cos(Lx/2*2*pi/Lx*X).*sin(0.5*pi*(Y+1));
%     C1 = 0.0*(1 - Y);
%     V1 = 0.5*(1 - Y);
    load('C_ini_10.mat');
    tempn = size(C_ini(:,1),1)-1;
    tempy = -cos(pi*(0:tempn)/tempn).';
    C_ini = interp1(tempy,C_ini(:,1),y);
    V_ini = interp1(tempy,V_ini(:,1),y);
    C_ini = C_ini(:,1)*ones(1,NX);
    V_ini = V_ini(:,1)*ones(1,NX);
    C1 = C_ini;
    V1 = V_ini;
    
    a = 0e-3;
    T1 = a*randn(N+1,NX).*(Y-1).*(Y+1);
    %     C1 = zeros(N+1,NX)+0.0001*cos(2*pi/Lx*X).*sin(0.5*pi*(Y+1));
    %     C1 = q;
    %     V1 = 0.5*uwv * (-Y+1);
    
    time = 0;
    slnW = [];
else
    load(['PoislnvorEC', num2str(N), '.mat']);
    %     psi = reshape(psi,NX*(N+1),1);
    %     q   = reshape(q,NX*(N+1),1);
    %     q_psi = zeros(NX*(N+1),1);
    % Add perturbation
    %     q = 0.001*cos(2*pi/Lx*X).*sin(0.5*pi*(Y+1));
    %     psiPerturb = reshape(fft((1-Y.^2).*q,[],2),(N+1)*NX,1);
    %     psiPrevious_hat = psiPrevious_hat + psiPerturb;
    q_psi = zeros((N+1),NX);
    if (size(psiPrevious_hat,2) == 1)
        psiPrevious_hat = reshape(psiPrevious_hat,N+1,NX);
        psiCurrent_hat = reshape(psiCurrent_hat,N+1,NX);
        psiCurrent_hat = reshape(psiCurrent_hat,N+1,NX);
        V1_hat = reshape(V1_hat,N+1,NX);
        V0_hat = reshape(V0_hat,N+1,NX);
        C1_hat = reshape(C1_hat,N+1,NX);
        C0_hat = reshape(C0_hat,N+1,NX);
        T1_hat = reshape(T1_hat,N+1,NX);
        T0_hat = reshape(T0_hat,N+1,NX);
    end
end
Preproces = toc;
runtime = Preproces;
disp('Preprocessing time...');
disp(Preproces);

% vidfile = VideoWriter('Charge.mp4','MPEG-4');
% open(vidfile);
for it = 1:maxit
    
    if it == 1
        % On the first iteration, w0 will be empty
        %     disp('creating persistent variables in RHS')
        if time == 0
            psiPrevious_hat = fft(psi,[],2);
            psiCurrent_hat  = psiPrevious_hat;
            C1_hat = fft(C1,[],2);
            C0_hat = C1_hat;
            V1_hat = fft(V1,[],2);
            V0_hat = V1_hat;
            T1_hat = fft(T1,[],2);
            T0_hat = T1_hat;
        end
        
        w1_hat = zeros((N+1),(NX));
        u1_hat = zeros((N+1),(NX));
        v1_hat = zeros((N+1),(NX));
        w0_hat = zeros((N+1),(NX));
        u0_hat = zeros((N+1),(NX));
        v0_hat = zeros((N+1),(NX));
        w1x_hat = zeros((N+1),(NX));
        w1y_hat = zeros((N+1),(NX));
        w0x_hat = zeros((N+1),(NX));
        w0y_hat = zeros((N+1),(NX));
        diffu_hat = zeros((N+1),(NX));
        force1x = zeros((N+1),(NX));
        force2y = zeros((N+1),(NX));
        force3y = zeros((N+1),(NX));
        force4x = zeros((N+1),(NX));
        force5x = zeros((N+1),(NX));
        force6y = zeros((N+1),(NX));
        force7x = zeros((N+1),(NX));
        force8y = zeros((N+1),(NX));
        E1x = zeros((N+1),(NX));
        E1y = zeros((N+1),(NX));
        E0x = zeros((N+1),(NX));
        E0y = zeros((N+1),(NX));
        conuC1 = zeros((N+1),(NX));
        convC1 = zeros((N+1),(NX));
        convC0 = zeros((N+1),(NX));
        conuC0 = zeros((N+1),(NX));
        convecCU1 = zeros((N+1),(NX));
        convecCU0 = zeros((N+1),(NX));
        convecCV1 = zeros((N+1),(NX));
        convecCV0 = zeros((N+1),(NX));
        diffuC = zeros((N+1),(NX));
        Fx = zeros((N+1),(NX));
        Fy = zeros((N+1),(NX));
        
        conuT1 = zeros((N+1),(NX));
        convT1 = zeros((N+1),(NX));
        convT0 = zeros((N+1),(NX));
        conuT0 = zeros((N+1),(NX));
        diffuT = zeros((N+1),(NX));
        FT = zeros((N+1),(NX));
        
        w0_hat(:,1) = psiDel2_np{1}*psiPrevious_hat(:,1);
        u0_hat(:,1) = psiDy_np*psiPrevious_hat(:,1);
        v0_hat(:,1) = 0;
        for m = 2:NX/2
            w0_hat(:,m) = psiDel2_np{m}*psiPrevious_hat(:,m);
            w0_hat(:,NX-(m-2)) = conj(w0_hat(:,m));
            
            u0_hat(:,m) = psiDy_np*psiPrevious_hat(:,m);
            u0_hat(:,NX-(m-2)) = conj(u0_hat(:,m));
            
            v0_hat(:,m) = -1i*kx(m)*psiPrevious_hat(:,m);
            v0_hat(:,NX-(m-2)) = conj(v0_hat(:,m));
        end
        w0_hat(:,NX/2+1) = psiDel2_np{NX/2+1}*psiPrevious_hat(:,NX/2+1);
        u0_hat(:,NX/2+1) = psiDy_np*psiPrevious_hat(:,NX/2+1);
        v0_hat(:,NX/2+1) = -1i*kx(NX/2+1)*psiPrevious_hat(:,NX/2+1);
        
    else
        w0_hat = w1_hat;
        u0_hat = u1_hat;
        v0_hat = v1_hat;
    end
    
    w1_hat(:,1) = psiDel2_np{1}*psiCurrent_hat(:,1);
    u1_hat(:,1) = psiDy_np*psiCurrent_hat(:,1);
    v1_hat(:,1) = 0;
    
    C1 = real(ifft(C1_hat,[],2));
    C0 = real(ifft(C0_hat,[],2)); 
    
    % Non-conservation form 
    w1x_hat(:,1) = 0;
    w1y_hat(:,1) = psiDy_np*w1_hat(:,1);
    w0x_hat(:,1) = 0;
    w0y_hat(:,1) = psiDy_np*w0_hat(:,1);
    
    diffu_hat(:,1) = psiDel4_np{1}*psiCurrent_hat(:,1);
    
    %     force1x(:,1) = 0;
    force2y(:,1) = psiDy_np*V1_hat(:,1);
    %     force3y(:,1) = psiDy_np*C1_hat(:,1);
    force4x(:,1) = 0;
    
    %     force5x(:,1) = 0;
    %     force6y(:,1) = psiDy_np*C0_hat(:,1);
    
    force7x(:,1) = 0;
    force8y(:,1) = psiDy_np*V0_hat(:,1);
    
    FT(:,1) = 0;
    
    for m = 2:NX/2
        w1_hat(:,m) = psiDel2_np{m}*psiCurrent_hat(:,m);
        w1_hat(:,NX-(m-2)) = conj(w1_hat(:,m));
        
        u1_hat(:,m) = psiDy_np*psiCurrent_hat(:,m);
        u1_hat(:,NX-(m-2)) = conj(u1_hat(:,m));
        
        v1_hat(:,m) = -1i*kx(m)*psiCurrent_hat(:,m);
        v1_hat(:,NX-(m-2)) = conj(v1_hat(:,m));
        
        w1x_hat(:,m) = 1i*kx(m)*w1_hat(:,m);
        w1x_hat(:,NX-(m-2)) = conj(w1x_hat(:,m));
        
        w0x_hat(:,m) = 1i*kx(m)*w0_hat(:,m);
        w0x_hat(:,NX-(m-2)) = conj(w0x_hat(:,m));
        
        w1y_hat(:,m) = psiDy_np*w1_hat(:,m);
        w1y_hat(:,NX-(m-2)) = conj(w1y_hat(:,m));
        
        w0y_hat(:,m) = psiDy_np*w0_hat(:,m);
        w0y_hat(:,NX-(m-2)) = conj(w0y_hat(:,m));
        
        diffu_hat(:,m) = psiDel4_np{m}*psiCurrent_hat(:,m);
        diffu_hat(:,NX-(m-2)) = conj(diffu_hat(:,m));
        
        %         force1x(:,m) = 1i*kx(m)*C1_hat(:,m);
        %         force1x(:,NX-(m-2)) = conj(force1x(:,m));
        
        force2y(:,m) = psiDy_np*V1_hat(:,m);
        force2y(:,NX-(m-2)) = conj(force2y(:,m));
        
        %         force3y(:,m) = psiDy_np*C1_hat(:,m);
        %         force3y(:,NX-(m-2)) = conj(force3y(:,m));
        
        force4x(:,m) = 1i*kx(m)*V1_hat(:,m);
        force4x(:,NX-(m-2)) = conj(force4x(:,m));
        
        %         force5x(:,m) = 1i*kx(m)*C0_hat(:,m);
        %         force5x(:,NX-(m-2)) = conj(force5x(:,m));
        %
        %         force6y(:,m) = psiDy_np*C0_hat(:,m);
        %         force6y(:,NX-(m-2)) = conj(force6y(:,m));
        
        force7x(:,m) = 1i*kx(m)*V0_hat(:,m);
        force7x(:,NX-(m-2)) = conj(force7x(:,m));
        
        force8y(:,m) = psiDy_np*V0_hat(:,m);
        force8y(:,NX-(m-2)) = conj(force8y(:,m));
        
        % Thermal force
        FT(:,m) = 1i*kx(m)*T1_hat(:,m);
        FT(:,NX-(m-2)) = conj(FT(:,m));  
        
    end
    
    w1_hat(:,NX/2+1) = psiDel2_np{NX/2+1}*psiCurrent_hat(:,NX/2+1);
    u1_hat(:,NX/2+1) = psiDy_np*psiCurrent_hat(:,NX/2+1);
    v1_hat(:,NX/2+1) = -1i*kx(NX/2+1)*psiCurrent_hat(:,NX/2+1);
    
    w1x_hat(:,NX/2+1) = 1i*kx(NX/2+1)*w1_hat(:,NX/2+1);
    w0x_hat(:,NX/2+1) = 1i*kx(NX/2+1)*w0_hat(:,NX/2+1);
    w1y_hat(:,NX/2+1) = psiDy_np*w1_hat(:,NX/2+1);
    w0y_hat(:,NX/2+1) = psiDy_np*w0_hat(:,NX/2+1);
    
    diffu_hat(:,NX/2+1) = psiDel4_np{NX/2+1}*psiCurrent_hat(:,NX/2+1);
    
    %     force1x(:,NX/2+1) = 1i*kx(NX/2+1)*C1_hat(:,NX/2+1); %DelC1/Delx
    force2y(:,NX/2+1) = psiDy_np*V1_hat(:,NX/2+1);      %DelV1/Dely
    %     force3y(:,NX/2+1) = psiDy_np*C1_hat(:,NX/2+1);      %DelC1/Dely
    force4x(:,NX/2+1) = 1i*kx(NX/2+1)*V1_hat(:,NX/2+1); %DelV1/Delx
    %     force5x(:,NX/2+1) = 1i*kx(NX/2+1)*C0_hat(:,NX/2+1); %DelC0/Delx
    %     force6y(:,NX/2+1) = psiDy_np*C0_hat(:,NX/2+1);      %DelC0/Dely
    force7x(:,NX/2+1) = 1i*kx(NX/2+1)*V0_hat(:,NX/2+1); %DelV0/Delx
    force8y(:,NX/2+1) = psiDy_np*V0_hat(:,NX/2+1);      %DelV1/Dely
    
    FT(:,NX/2+1) = 1i*kx(NX/2+1)*T1_hat(:,NX/2+1); %DelT1/Delx
    
    tempFx = fft(C1.*real(ifft(force4x,[],2)),[],2);
    tempFy = fft(C1.*real(ifft(force2y,[],2)),[],2);
    
    Fx(:,1) = 0;
    Fy(:,1) = psiDy_np*tempFx(:,1);
    for m = 2:NX/2
        Fx(:,m) = 1i*kx(m)*tempFy(:,m);
        Fx(:,NX-(m-2)) = conj(Fx(:,m));
        Fy(:,m) = psiDy_np*tempFx(:,m);
        Fy(:,NX-(m-2)) = conj(Fy(:,m));
    end
    
    Fx(:,NX/2+1) = 1i*kx(NX/2+1)*tempFy(:,NX/2+1);
    Fy(:,NX/2+1) = psiDy_np*tempFx(:,NX/2+1);
    
    conu1 = real(ifft(u1_hat,[],2)+uw).*real(ifft(w1x_hat,[],2));
    conv1 = real(ifft(v1_hat,[],2)).*real(ifft(w1y_hat,[],2));
    conu0 = real(ifft(u0_hat,[],2)+uw).*real(ifft(w0x_hat,[],2));
    conv0 = real(ifft(v0_hat,[],2)).*real(ifft(w0y_hat,[],2));
    
    % Conservative form
    u1 = real(ifft(u1_hat,[],2))+uw; 
    u0 = real(ifft(u0_hat,[],2))+uw; 
    v1 = real(ifft(v1_hat,[],2)); 
    v0 = real(ifft(v0_hat,[],2)); 
    w1 = real(ifft(w1_hat,[],2)); 
    w0 = real(ifft(w0_hat,[],2)); 
    
    uw1_hat = fft(u1.*w1,[],2);
    uw0_hat = fft(u0.*w0,[],2);
    vw1_hat = fft(v1.*w1,[],2);
    vw0_hat = fft(v0.*w0,[],2);
    
    convecCU1(:,1) = 0;
    convecCU0(:,1) = 0;
    convecCV1(:,1) = psiDy_np*vw1_hat(:,1);
    convecCV0(:,1) = psiDy_np*vw0_hat(:,1);
    for m = 2:NX/2
        convecCU1(:,m) = 1i*kx(m)*uw1_hat(:,m);
        convecCU1(:,NX-(m-2)) = conj(convecCU1(:,m));
        convecCU0(:,m) = 1i*kx(m)*uw0_hat(:,m);
        convecCU0(:,NX-(m-2)) = conj(convecCU0(:,m));
        
        convecCV1(:,m) = psiDy_np*vw1_hat(:,m);
        convecCV1(:,NX-(m-2)) = conj(convecCV1(:,m));
        convecCV0(:,m) = psiDy_np*vw0_hat(:,m);
        convecCV0(:,NX-(m-2)) = conj(convecCV0(:,m));  
    end
    
    convecCU1(:,NX/2+1) = 1i*kx(NX/2+1)*uw1_hat(:,NX/2+1);
    convecCU0(:,NX/2+1) = 1i*kx(NX/2+1)*uw0_hat(:,NX/2+1);
    convecCV1(:,NX/2+1) = psiDy_np*vw1_hat(:,NX/2+1);
    convecCV0(:,NX/2+1) = psiDy_np*vw0_hat(:,NX/2+1);
    
    convecConser = -1.5*dt*(convecCU1+convecCV1)+0.5*dt*(convecCU0+convecCV0);
    
    %     force = dt/rho*(real(ifft(force1x,[],2)).*real(ifft(force2y,[],2)) - real(ifft(force3y,[],2)).*real(ifft(force4x,[],2)));
    Force = dt*(Fx-Fy)*ForceE;
    FT = -dt*FT*ForceT;
    
    RHS = 0.5*(convecConser+fft(((-1.5)*dt)*(conu1 + conv1) ...
        +(0.5*dt)*(conu0 + conv0),[],2)) ...
        + (0.5*dt*nu)*diffu_hat  ...
        + w1_hat + Force + FT;
    
    
    % Charge density transport
    
    
    %     E1x(:,1) = 0;
    %     E1y(:,1) = psiDy_np*V1_hat(:,1);
    %     E0y(:,1) = psiDy_np*V0_hat(:,1);
    %     E0x(:,1) = 0;
    %
    %     for m = 2:NX/2
    %         E1x(:,m) = 1i*kx(m)*V1_hat(:,m);
    %         E1x(:,NX-(m-2)) = conj(E1x(:,m));
    %
    %         E1y(:,m) = psiDy_np*V1_hat(:,m);
    %         E1y(:,NX-(m-2)) = conj(E1y(:,m));
    %
    %         E0x(:,m) = 1i*kx(m)*V0_hat(:,m);
    %         E0x(:,NX-(m-2)) = conj(E0x(:,m));
    %
    %         E0y(:,m) = psiDy_np*V0_hat(:,m);
    %         E0y(:,NX-(m-2)) = conj(E0y(:,m));
    %     end
    %     E1x(:,NX/2+1) = 1i*kx(NX/2+1)*V1_hat(:,NX/2+1);
    %     E1y(:,NX/2+1) = psiDy_np*V1_hat(:,NX/2+1);
    %     E0y(:,NX/2+1) = psiDy_np*V0_hat(:,NX/2+1);
    %     E0x(:,NX/2+1) = 1i*kx(NX/2+1)*V0_hat(:,NX/2+1);
    
    Eu1 = fft(real(ifft(u1_hat-mub*force4x,[],2)+uw).*C1,[],2);
    Ev1 = fft(real(ifft(v1_hat-mub*force2y,[],2)).*C1,[],2);
    Eu0 = fft(real(ifft(u0_hat-mub*force7x,[],2)+uw).*C0,[],2);
    Ev0 = fft(real(ifft(v0_hat-mub*force8y,[],2)).*C0,[],2);
    
    % (u-mubE)*delC1
    
    %     force1x(:,NX/2+1) = 1i*kx(NX/2+1)*C1_hat(:,NX/2+1); %DelC1/Delx
    %     force2y(:,NX/2+1) = psiDy_np*V1_hat(:,NX/2+1);      %DelV1/Dely
    %     force3y(:,NX/2+1) = psiDy_np*C1_hat(:,NX/2+1);      %DelC1/Dely
    %     force4x(:,NX/2+1) = 1i*kx(NX/2+1)*V1_hat(:,NX/2+1); %DelV1/Delx
    %     force5x(:,NX/2+1) = 1i*kx(NX/2+1)*C0_hat(:,NX/2+1); %DelC0/Delx
    %     force6y(:,NX/2+1) = psiDy_np*C0_hat(:,NX/2+1);      %DelC0/Dely
    %     force7x(:,NX/2+1) = 1i*kx(NX/2+1)*V0_hat(:,NX/2+1); %DelV0/Delx
    %     force8y(:,NX/2+1) = psiDy_np*V0_hat(:,NX/2+1);      %DelV1/Dely
    
    %     conuC1 = fft(real(ifft(u1_hat-mub*force4x,[],2)).*real(ifft(force1x,[],2)),[],2);
    %     convC1 = fft(real(ifft(v1_hat-mub*force2y,[],2)).*real(ifft(force3y,[],2)),[],2);
    %     convC0 = fft(real(ifft(u0_hat-mub*force7x,[],2)).*real(ifft(force5x,[],2)),[],2);
    %     conuC0 = fft(real(ifft(v0_hat-mub*force8y,[],2)).*real(ifft(force6y,[],2)),[],2);
    
    
    conuC1(:,1) = 0;
    convC1(:,1) = psiDy_np*Ev1(:,1);
    convC0(:,1) = psiDy_np*Ev0(:,1);
    conuC0(:,1) = 0;
    diffuC(:,1) = psiDel2_np{1}*C1_hat(:,1);
    
    for m = 2:NX/2
        conuC1(:,m) = 1i*kx(m)*Eu1(:,m);
        conuC1(:,NX-(m-2)) = conj(conuC1(:,m));
        
        convC1(:,m) = psiDy_np*Ev1(:,m);
        convC1(:,NX-(m-2)) = conj(convC1(:,m));
        
        conuC0(:,m) = 1i*kx(m)*Eu0(:,m);
        conuC0(:,NX-(m-2)) = conj(conuC0(:,m));
        
        convC0(:,m) = psiDy_np*Ev0(:,m);
        convC0(:,NX-(m-2)) = conj(convC0(:,m));
        
        diffuC(:,m) = psiDel2_np{m}*C1_hat(:,m);
        diffuC(:,NX-(m-2)) = conj(diffuC(:,m));
    end
    conuC1(:,NX/2+1) = 1i*kx(NX/2+1)*Eu1(:,NX/2+1);
    convC1(:,NX/2+1) = psiDy_np*Ev1(:,NX/2+1);
    convC0(:,NX/2+1) = psiDy_np*Ev0(:,NX/2+1);
    conuC0(:,NX/2+1) = 1i*kx(NX/2+1)*Eu0(:,NX/2+1);
    diffuC(:,NX/2+1) = psiDel2_np{NX/2+1}*C1_hat(:,NX/2+1);
    
    RHSC = ((-1.5)*dt)*(conuC1 + convC1) ...
        +(0.5*dt)*(conuC0 + convC0) ...
        +(0.5*dt/Fe)*diffuC  ...
        + C1_hat;
    
    RHSV = -C1_hat*C/4;
    
    % RHS = ((-1.5)*dt)*((ifft(u1_hat).*ifft(psiDx * w1_hat)) + (ifft(v1_hat).*ifft(psiDy * w1_hat)) ) +...
    %       +(0.5*dt)*((ifft(u0_hat).*ifft(psiDx * w0_hat)) + (ifft(v0_hat).*ifft(psiDy * w0_hat)) ) +...
    %       +(0.5*dt*nu/rho)*ifft(psiDel4 * psiCurrent_hat)  +...
    %       + ifft(w1_hat);
    
    % Temperature transport
    conuT1(:,1) = 0;
    convT1(:,1) = psiDy_np*T1_hat(:,1);
    convT0(:,1) = psiDy_np*T0_hat(:,1);
    conuT0(:,1) = 0;
    diffuT(:,1) = psiDel2_np{1}*T1_hat(:,1);

    for m = 2:NX/2
        conuT1(:,m) = 1i*kx(m)*T1_hat(:,m);
        conuT1(:,NX-(m-2)) = conj(conuT1(:,m));
        
        convT1(:,m) = psiDy_np*T1_hat(:,m);
        convT1(:,NX-(m-2)) = conj(convT1(:,m));
        
        conuT0(:,m) = 1i*kx(m)*T0_hat(:,m);
        conuT0(:,NX-(m-2)) = conj(conuT0(:,m));
        
        convT0(:,m) = psiDy_np*T0_hat(:,m);
        convT0(:,NX-(m-2)) = conj(convT0(:,m));
        
        diffuT(:,m) = psiDel2_np{m}*T1_hat(:,m);
        diffuT(:,NX-(m-2)) = conj(diffuT(:,m));
    end
    conuT1(:,NX/2+1) = 1i*kx(NX/2+1)*T1_hat(:,NX/2+1);
    convT1(:,NX/2+1) = psiDy_np*T1_hat(:,NX/2+1);
    convT0(:,NX/2+1) = psiDy_np*T0_hat(:,NX/2+1);
    conuT0(:,NX/2+1) = 1i*kx(NX/2+1)*T0_hat(:,NX/2+1);
    diffuT(:,NX/2+1) = psiDel2_np{NX/2+1}*T1_hat(:,NX/2+1);
      
    conuT1 = real(ifft(u1_hat,[],2)).*real(ifft(conuT1,[],2));
    convT1 = real(ifft(v1_hat,[],2)).*real(ifft(convT1,[],2));
    conuT0 = real(ifft(u0_hat,[],2)).*real(ifft(conuT0,[],2));
    convT0 = real(ifft(v0_hat,[],2)).*real(ifft(convT0,[],2));
    
    RHST = fft(((-1.5)*dt)*(conuT1 + convT1) ...
        +(0.5*dt)*(conuT0 + convT0),[],2) ...
        +(0.5*dt*kT)*diffuT+dt*v1_hat  ...
        + T1_hat;
    
    % Boundary Condition
    %  S= [1;1-x(2:N).^2;1];
    RHS(bd_pts) = 0;  % no velocity on the boundaries ...
    RHS(t_pts) = (0.5 * (uwall + 0*fft(ones(1,NX)*sin(2*pi/maxit*it))));% ./ S; % except the lid
    %  RHS(b_pts) = -(0.5 * (uwall));% ./ S; % Same for Couette; equal opposite for Poiseuille
    RHS(b_pts) = -(0.5 * (uwall));% ./ S; % Same for Couette; equal opposite for Poiseuille
    
    RHSC(t_pts) = 0;
    RHSC(b_pts) = uwallc;
    
    RHSV(bd_pts) = 0;
    RHSV(b_pts) = uwallv;
    
    RHST(bd_pts) = 0;
    % Break up RHS
    %  RHS_np = reshape(RHS, NP*(N+1),partition);
    %
    %  for p = 1:partition
    %     q_psi((p-1)*NP*(N+1)+1:p*NP*(N+1)) = reshape(LHS_np(p,:,:),NP*(N+1),NP*(N+1)) * RHS_np(:,p);
    % end
    C0_hat = C1_hat;
    V0_hat = V1_hat;
    T0_hat = T1_hat;
    
    q_psi(:,1) = LHS_np{1}*RHS(:,1);
    C1_hat(:,1) = LHSC_np{1}*RHSC(:,1);
    V1_hat(:,1) = LHSV_np{1}*RHSV(:,1);
    T1_hat(:,1) = LHST_np{1}*RHST(:,1);
    for m = 2:NX/2
        q_psi(:,m) = LHS_np{m}*RHS(:,m);
        q_psi(:,NX-(m-2)) = conj(q_psi(:,m));
        C1_hat(:,m) = LHSC_np{m}*RHSC(:,m);
        C1_hat(:,NX-(m-2)) = conj(C1_hat(:,m));
        V1_hat(:,m) = LHSV_np{m}*RHSV(:,m);
        V1_hat(:,NX-(m-2)) = conj(V1_hat(:,m));
        T1_hat(:,m) = LHST_np{m}*RHST(:,m);
        T1_hat(:,NX-(m-2)) = conj(T1_hat(:,m));
    end
    q_psi(:,NX/2+1) = LHS_np{NX/2+1}*RHS(:,NX/2+1);
    C1_hat(:,NX/2+1) = LHSC_np{NX/2+1}*RHSC(:,NX/2+1);
    V1_hat(:,NX/2+1) = LHSV_np{NX/2+1}*RHSV(:,NX/2+1);
    T1_hat(:,NX/2+1) = LHST_np{NX/2+1}*RHST(:,NX/2+1);
    
    psiTemp = ((1-Y.^2)).*q_psi;
    
    % shifting
    psiPrevious_hat = psiCurrent_hat;
    psiCurrent_hat  = psiTemp;
    
    if (mod(it,NSAVE) == 0 || it == maxit)
        disp('Time=');
        disp(time);
        disp(['Iteration/' num2str(maxit)]);
        disp(it);
        disp('Runtime');
        disp(toc-runtime);
        runtime = toc;
        %         figure
        %         contourf(X,Y,real(ifft(reshape(u1_hat,N+1,NX),[],2)));
%         tempW = max(max(sqrt(real(ifft(u1_hat,[],2)+uw).^2 + real(ifft(v1_hat,[],2)).^2)))
        tempW = max(max(real(ifft(u1_hat,[],2)+uw)))
        slnW = [slnW; tempW];
        %         if saveTrue
        %             save(['PoislnvorEC', num2str(N), '.mat'],'psiCurrent_hat','psiPrevious_hat',...
        %                 'C1_hat','C0_hat','V1_hat','V0_hat','time','slnW','-v7.3');
        %         end
        
        %                 z = real(ifft(reshape(C1_hat,N+1,NX),[],2));
        %                 im = sc(z.', 'hot');
        %                 writeVideo(vidfile, im);
    end
    time = time + dt;
end
runtime = toc;
disp('Runtime');
disp(runtime);
% close(vidfile)
%%
psi = real(ifft(reshape(psiCurrent_hat,N+1,NX),[],2));
u1 = real(ifft(reshape(u1_hat,N+1,NX),[],2)) + uw;
v1 = real(ifft(reshape(v1_hat,N+1,NX),[],2));
w1 = real(ifft(reshape(w1_hat,N+1,NX),[],2));
C1 = real(ifft(reshape(C1_hat,N+1,NX),[],2));
V1 = real(ifft(reshape(V1_hat,N+1,NX),[],2));
T1 = real(ifft(reshape(T1_hat,N+1,NX),[],2));
save(['wMATLAB_time=' num2str(time) '.mat'], 'w1','C1','T1')
% contourf(X,Y,u1);
% colormap('jet');
% grid on
% title('u');
% figure
% contourf(X,Y,v1);
% colormap('jet');
% grid on
% title('v');
% figure
% contourf(X,Y,w1);
% colormap('jet');
% grid on
% title('vorticity');
% figure
% contourf(X,Y,psi);
% colormap('jet');
% grid on
% title('\Psi');
% figure
% contourf(X,Y,C1);
% colormap('jet');
% grid on
% title('Charge density');
% figure
% contourf(X,Y,V1);
% colormap('jet');
% grid on
% title('Voltage');
% figure
% contourf(X,Y,real(ifft(Force,[],2)));
% colormap('jet');
% grid on
% title('Boussinesq force');
% 
% 
% figure
% h1 = quiver(X,Y,u1,v1);
% set(h1,'AutoScale','on', 'AutoScaleFactor', 5)
% starty = 0:0.1:1;
% startx = ones(size(starty))*0.5;
% h2=streamline(X,Y,u1,v1,startx,starty);
% set(h2,'LineWidth',1,'Color','k')
% axis([0 Lx -1 1]);
% 

% Save data
if saveTrue
    save(['PoislnvorEC', num2str(N), '.mat'],'psiCurrent_hat','psiPrevious_hat',...
        'C1_hat','C0_hat','V1_hat','V0_hat','T1_hat','T0_hat','time','slnW');
end

% max(max(abs(w1)))
% plot(slnW)
% uana = -0.5*3*uw*y.^2 + 0.5*uw;
% error = uana - u1(:,1);
% figure
% plot(y,error);
% norm(error)
% figure
% plot(y,V1(:,1));
% figure
% plot(y,C_ini(:,1),'or');
% hold on
% plot(y,C1(:,1));
% er = norm(C1(:,1)-C_ini(:,1))
% C_ini = C1;
% V_ini = V1;
% save('C_ini_2p5.mat','C_ini','V_ini');

% current
% Ey = -1*real(ifft(force2y,[],2));
% Current = C1.*Ey;
% current = mean(Current(end,:))

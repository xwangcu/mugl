%% Parameters
random = 1000;
rng(random);
alg = 2; % 1: (LS-)PGD; 2: ADMM (solve VSGL/MUGL)
max_iter = 1e2;

% PGD parameters
step_size = 1e-3;
eps = 1e-2;

% ADMM parameters
t = 1;
tau1 = 1e-4;
tau2 = 1e-3;
eps_primal = 1e-3;
eps_dual = 1e-3;

%% Construct Temperature Graph
currentFolder = pwd;
load([currentFolder,'\data\temperature.mat']);
X = temp_records;
dim = size(X,1);
num = size(X,2);
A = zeros(dim,dim);
for i = 1 : dim
    for j = 1 : dim
        if abs(altitude(i)-altitude(j))<300 && i~=j
            A(i,j) = 1;
        end
    end
end
edge_num = sum(sum(A));
Diag = diag(sum(full(A)));
L_0 = Diag-full(A);
L_0 = 2*dim*L_0/trace(L_0);

%% Run Algorithms
% VSGL
if alg == 1
    [L_NR, ~, ~, ~] = graph_learning_PGD(X, 0, 0, 0, step_size, max_iter, eps);
elseif alg == 2
    [L_NR, ~] = graph_learning_ADMM(X, 0, 0, 0, t, tau1, tau2, max_iter, eps_primal, eps_dual);
else
    fprintf('algorithm type not specified!\n')
end
L_NR(abs(L_NR)<10^(-4))=0;

% GL-SigRep
param.N = dim;
param.max_iter = 50;
param.alpha = 2; 
param.beta = 165;
[L_SIG, ~, ~] = graph_learning_gaussian_dong(X, param);
L_SIG(abs(L_SIG)<10^(-4))=0;

% Log-Model
rho_EPFL = 30; 
alpha_EPFL = 200;
[L_EPFL, fval_EPFL] = graph_learning_ADMM_epfl(X, rho_EPFL, alpha_EPFL, t, tau1, tau2, max_iter, eps_primal, eps_dual);
L_EPFL(abs(L_EPFL)<10^(-4))=0;

% MUGL-o
rho1_DR = 20;
rho2_DR = 600;
if alg == 1
        [L_DR, ~, ~, ~] = graph_learning_PGD(X, rho1_DR, rho2_DR, 0, step_size, max_iter, eps);
    elseif alg == 2
        t_DR = 1e0; tau1_DR = 1e-4; tau2_DR = 1e-3;
        [L_DR, ~] = graph_learning_ADMM(X, rho1_DR, rho2_DR, 0, t_DR, tau1_DR, tau2_DR, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
end
L_DR(abs(L_DR)<10^(-4))=0;

% MUGL-l
rho1_DRL = 1.7; 
rho2_DRL = 630; 
alpha_DRL = 360; 
if alg == 1
    [L_DRL, ~, ~, ~] = graph_learning_LSPGD(X, rho1_DRL, rho2_DRL, alpha_DRL, step_size, 0.1, 0.1, max_iter, eps);
elseif alg == 2
    t_DR = 1e0; tau1_DR = 1e-4; tau2_DR = 1e-3;
    [L_DRL, ~] = graph_learning_ADMM(X, rho1_DRL, rho2_DRL, alpha_DRL, t, tau1, tau2, max_iter, eps_primal, eps_dual);
else
    fprintf('algorithm type not specified!\n')
end
L_DRL(abs(L_DRL)<10^(-4))=0;

% WRGL
eta_WGRL = 0.8; epsilon_WGRL = 12; beta_WGRL = 6;
step_size_WRGL = 1e-5; max_iter_WRGL = 1e6; tol_WRGL = 1e-6;
[L_WRGL, fval] = graph_learning_WRGL(X, eta_WGRL, epsilon_WGRL, beta_WGRL, step_size_WRGL, max_iter_WRGL, tol_WRGL);
L_WRGL(abs(L_WRGL)<10^(-4)) = 0;

%% Evaluation
[precision_NR,recall_NR,Fmeasure_NR,NMI_NR] = graph_learning_perf_eval(L_0, L_NR);
[precision_SIG ,recall_SIG ,Fmeasure_SIG ,NMI_SIG] = graph_learning_perf_eval(L_0, L_SIG);
[precision_EPFL ,recall_EPFL ,Fmeasure_EPFL ,NMI_EPFL] = graph_learning_perf_eval(L_0, L_EPFL);
[precision_DR ,recall_DR ,Fmeasure_DR ,NMI_DR] = graph_learning_perf_eval(L_0,L_DR);
[precision_DRL, recall_DRL, Fmeasure_DRL, NMI_DRL] = graph_learning_perf_eval(L_0, L_DRL);
[precision_WRGL,recall_WRGL,Fmeasure_WRGL,NMI_WRGL,~] = graph_learning_perf_eval(L_0,L_WRGL);

fprintf('temperature data     |   precision	|	recall	|	Fmeasure	|	NMI\n');
fprintf('VSGL       |       %.3f            %.3f            %.3f        %.3f\n ', precision_NR, recall_NR, Fmeasure_NR, NMI_NR);
fprintf('GL-SigRep	|       %.3f            %.3f            %.3f        %.3f\n ', precision_SIG, recall_SIG, Fmeasure_SIG, NMI_SIG);
fprintf('Log-Model	|       %.3f            %.3f            %.3f        %.3f\n ', precision_EPFL, recall_EPFL, Fmeasure_EPFL, NMI_EPFL);
fprintf('MUGL-o     |       %.3f            %.3f            %.3f        %.3f\n ', precision_DR, recall_DR, Fmeasure_DR, NMI_DR);
fprintf('MUGL-l     |       %.3f            %.3f            %.3f        %.3f\n ', precision_DRL, recall_DRL, Fmeasure_DRL, NMI_DRL);
fprintf('WRGL   |           %.3f            %.3f            %.3f        %.3f\n ', precision_WRGL, recall_WRGL, Fmeasure_WRGL, NMI_WRGL);

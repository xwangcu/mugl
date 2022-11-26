%% Setting
random = 1000;
rng(random);
dim = 20;
num = 80;
noise_level = 1;
graph = 2; % 1: Gaussian graph; 2: ER graph; 3: PA graph

%% Algorithm Parameters
nrepeat = 50;
max_iter = 1e8;
alg = 2; % 1: (LS-)PGD; 2: ADMM (solve VSGL/MUGL)

% PGD parameters
step_size = 1e-4;
eps = 1e-4;

% ADMM parameters
t = 1e0;
tau1 = 1e-5;
tau2 = 1e-4;
eps_primal = 1e-4;
eps_dual = 1e-4;

%% Initialize Performance Evaluation
precision_NR = zeros(nrepeat,1);
recall_NR = zeros(nrepeat,1);
Fmeasure_NR = zeros(nrepeat,1);
NMI_NR = zeros(nrepeat,1);

precision_SIG = zeros(nrepeat,1);
recall_SIG = zeros(nrepeat,1);
Fmeasure_SIG = zeros(nrepeat,1);
NMI_SIG = zeros(nrepeat,1);

precision_EPFL = zeros(nrepeat,1);
recall_EPFL = zeros(nrepeat,1);
Fmeasure_EPFL = zeros(nrepeat,1);
NMI_EPFL = zeros(nrepeat,1);

precision_DR = zeros(nrepeat,1);
recall_DR = zeros(nrepeat,1);
Fmeasure_DR = zeros(nrepeat,1);
NMI_DR = zeros(nrepeat,1);

precision_DRL = zeros(nrepeat,1);
recall_DRL = zeros(nrepeat,1);
Fmeasure_DRL = zeros(nrepeat,1);
NMI_DRL = zeros(nrepeat,1);

precision_WRGL = zeros(nrepeat,1);
recall_WRGL = zeros(nrepeat,1);
Fmeasure_WRGL = zeros(nrepeat,1);
NMI_WRGL = zeros(nrepeat,1);

%% Generate Sythetic Graphs
if graph == 1
    [A,XCoords, YCoords] = construct_graph(dim,'gaussian',0.75,0.5);
elseif graph == 2
    [A,XCoords, YCoords] = construct_graph(dim,'er',0.2);
elseif graph == 3
    [A,XCoords, YCoords] = construct_graph(dim,'pa',1);
end

Diag = diag(sum(full(A)));
L_0 = Diag-full(A);
L_0 = L_0/trace(L_0)*2*dim;
[V,D] = eig(full(L_0));

load('rng35_N20_Gaussian_graph.mat');

avg = rand(dim,1) * 1;
covariance = pinv(D);

%% Run Algorithms
for ii = 1 : nrepeat
    
    fprintf('ii=%d\n', ii);
    
    % generate graph signals
    gftcoeff = mvnrnd(zeros(1,dim),covariance,num);
    X = V*gftcoeff' + avg;
    X_noisy = X + noise_level*randn(size(X));
    
    % VSGL
    if alg == 1
        [L_NR, ~, ~, ~] = graph_learning_PGD(X_noisy, 0, 0, 0, step_size, max_iter, eps);
    elseif alg == 2
        [L_NR, ~] = graph_learning_ADMM(X_noisy, 0, 0, 0, t, tau1, tau2, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    
    % GL-SigRep
    param.alpha = 9; param.beta = 120; param.N = dim; param.max_iter = 50;
    [L_SIG, ~, ~] = graph_learning_gaussian_dong(X_noisy, param);
    Lcell = L_SIG;
    L_SIG(abs(L_SIG)<10^(-4)) = 0;
    
    % Log-Model
    rho_EPFL = 25; alpha_EPFL = 150;
    [L_EPFL, ~] = graph_learning_ADMM_epfl(X_noisy, rho_EPFL, alpha_EPFL, t, tau1, tau2, max_iter, eps_primal, eps_dual);
    L_EPFL(abs(L_EPFL)<10^(-4)) = 0;
    
    % MUGL-o
    rho1_DR = 0.15; rho2_DR = 5.6;
    if alg == 1
        [L_DR, ~, ~, ~] = graph_learning_PGD(X_noisy, rho1_DR, rho2_DR, 0, step_size, max_iter, eps);
    elseif alg == 2
        [L_DR, ~] = graph_learning_ADMM(X_noisy, rho1_DR, rho2_DR, 0, t, tau1, tau2, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    L_DR(abs(L_DR)<10^(-4)) = 0;
    
    % MUGL-l
    rho1_DRL = 0.2; rho2_DRL = 6.6; alpha_DRL = 7;
    if alg == 1
        [L_DRL, ~, ~, ~] = graph_learning_LSPGD(X_noisy, rho1_DRL, rho2_DRL, alpha_DRL, step_size, 0.1, 0.1, max_iter, eps);
    elseif alg == 2
        [L_DRL, ~] = graph_learning_ADMM(X_noisy, rho1_DRL, rho2_DRL, alpha_DRL, t, tau1, tau2, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    L_DRL(abs(L_DRL)<10^(-4)) = 0;
    
    % WRGL
    eta_WGRL = 2; epsilon_WGRL = 3; beta_WGRL = 2; step_size_WRGL = 1e-4; max_iter_WRGL = 1e8; tol_WRGL = 1e-4;
    [L_WRGL, fval] = graph_learning_WRGL(X_noisy, eta_WGRL, epsilon_WGRL, beta_WGRL, step_size_WRGL, max_iter_WRGL, tol_WRGL);
    L_WRGL(abs(L_WRGL)<10^(-4)) = 0;
    
    % evaluations
    [precision_NR(ii),recall_NR(ii),Fmeasure_NR(ii),NMI_NR(ii),~] = graph_learning_perf_eval(L_0,L_NR);
    [precision_SIG(ii),recall_SIG(ii),Fmeasure_SIG(ii),NMI_SIG(ii),~] = graph_learning_perf_eval(L_0,L_SIG);
    [precision_EPFL(ii),recall_EPFL(ii),Fmeasure_EPFL(ii),NMI_EPFL(ii),~] = graph_learning_perf_eval(L_0,L_EPFL);
    [precision_DR(ii),recall_DR(ii),Fmeasure_DR(ii),NMI_DR(ii),~] = graph_learning_perf_eval(L_0,L_DR);
    [precision_DRL(ii),recall_DRL(ii),Fmeasure_DRL(ii),NMI_DRL(ii),~] = graph_learning_perf_eval(L_0,L_DRL);
    [precision_WRGL(ii),recall_WRGL(ii),Fmeasure_WRGL(ii),NMI_WRGL(ii),~] = graph_learning_perf_eval(L_0,L_WRGL);
end

%% Calculate and Print Results
mean_precision_NR = mean(precision_NR(:));
mean_recall_NR = mean(recall_NR(:));
mean_Fmeasure_NR = mean(Fmeasure_NR(:));
mean_NMI_NR = mean(NMI_NR(:));
std_precision_NR = std(precision_NR(:));
std_recall_NR = std(recall_NR(:));
std_Fmeasure_NR = std(Fmeasure_NR(:));
std_NMI_NR = std(NMI_NR(:));
rstd_precision_NR = std_precision_NR*100/mean_precision_NR;
rstd_recall_NR = std_recall_NR*100/mean_recall_NR;
rstd_Fmeasure_NR = std_Fmeasure_NR*100/mean_Fmeasure_NR;
rstd_NMI_NR = std_NMI_NR*100/mean_NMI_NR;

mean_precision_SIG = mean(precision_SIG(:));
mean_recall_SIG = mean(recall_SIG(:));
mean_Fmeasure_SIG = mean(Fmeasure_SIG(:));
mean_NMI_SIG = mean(NMI_SIG(:));
std_precision_SIG = std(precision_SIG(:));
std_recall_SIG = std(recall_SIG(:));
std_Fmeasure_SIG = std(Fmeasure_SIG(:));
std_NMI_SIG = std(NMI_SIG(:));
rstd_precision_SIG = std_precision_SIG*100/mean_precision_SIG;
rstd_recall_SIG = std_recall_SIG*100/mean_recall_SIG;
rstd_Fmeasure_SIG = std_Fmeasure_SIG*100/mean_Fmeasure_SIG;
rstd_NMI_SIG = std_NMI_SIG*100/mean_NMI_SIG;

mean_precision_EPFL = mean(precision_EPFL(:));
mean_recall_EPFL = mean(recall_EPFL(:));
mean_Fmeasure_EPFL = mean(Fmeasure_EPFL(:));
mean_NMI_EPFL = mean(NMI_EPFL(:));
std_precision_EPFL = std(precision_EPFL(:));
std_recall_EPFL = std(recall_EPFL(:));
std_Fmeasure_EPFL = std(Fmeasure_EPFL(:));
std_NMI_EPFL = std(NMI_EPFL(:));
rstd_precision_EPFL = std_precision_EPFL*100/mean_precision_EPFL;
rstd_recall_EPFL = std_recall_EPFL*100/mean_recall_EPFL;
rstd_Fmeasure_EPFL = std_Fmeasure_EPFL*100/mean_Fmeasure_EPFL;
rstd_NMI_EPFL = std_NMI_EPFL*100/mean_NMI_EPFL;

mean_precision_DR = mean(precision_DR(:));
mean_recall_DR = mean(recall_DR(:));
mean_Fmeasure_DR = mean(Fmeasure_DR(:));
mean_NMI_DR = mean(NMI_DR(:));
std_precision_DR = std(precision_DR(:));
std_recall_DR = std(recall_DR(:));
std_Fmeasure_DR = std(Fmeasure_DR(:));
std_NMI_DR = std(NMI_DR(:));
rstd_precision_DR = std_precision_DR*100/mean_precision_DR;
rstd_recall_DR = std_recall_DR*100/mean_recall_DR;
rstd_Fmeasure_DR = std_Fmeasure_DR*100/mean_Fmeasure_DR;
rstd_NMI_DR = std_NMI_DR*100/mean_NMI_DR;

mean_precision_DRL = mean(precision_DRL(:));
mean_recall_DRL = mean(recall_DRL(:));
mean_Fmeasure_DRL = mean(Fmeasure_DRL(:));
mean_NMI_DRL = mean(NMI_DRL(:));
std_precision_DRL = std(precision_DRL(:));
std_recall_DRL = std(recall_DRL(:));
std_Fmeasure_DRL = std(Fmeasure_DRL(:));
std_NMI_DRL = std(NMI_DRL(:));
rstd_precision_DRL = std_precision_DRL*100/mean_precision_DRL;
rstd_recall_DRL = std_recall_DRL*100/mean_recall_DRL;
rstd_Fmeasure_DRL = std_Fmeasure_DRL*100/mean_Fmeasure_DRL;
rstd_NMI_DRL = std_NMI_DRL*100/mean_NMI_DRL;

mean_precision_WRGL = mean(precision_WRGL(:));
mean_recall_WRGL = mean(recall_WRGL(:));
mean_Fmeasure_WRGL = mean(Fmeasure_WRGL(:));
mean_NMI_WRGL = mean(NMI_WRGL(:));
std_precision_WRGL = std(precision_WRGL(:));
std_recall_WRGL = std(recall_WRGL(:));
std_Fmeasure_WRGL = std(Fmeasure_WRGL(:));
std_NMI_WRGL = std(NMI_WRGL(:));
rstd_precision_WRGL = std_precision_WRGL*100/mean_precision_WRGL;
rstd_recall_WRGL = std_recall_WRGL*100/mean_recall_WRGL;
rstd_Fmeasure_WRGL = std_Fmeasure_WRGL*100/mean_Fmeasure_WRGL;
rstd_NMI_WRGL = std_NMI_WRGL*100/mean_NMI_WRGL;

fprintf('method     |    precision    |    recall    |    Fmeasure    |    NMI \n');
fprintf('VSGL       |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_NR, rstd_precision_NR, mean_recall_NR, rstd_recall_NR, mean_Fmeasure_NR, rstd_Fmeasure_NR, mean_NMI_NR, rstd_NMI_NR);
fprintf('GL-SigRep  |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_SIG, rstd_precision_SIG, mean_recall_SIG, rstd_recall_SIG, mean_Fmeasure_SIG, rstd_Fmeasure_SIG, mean_NMI_SIG, rstd_NMI_SIG);
fprintf('Log-Model  |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_EPFL, rstd_precision_EPFL, mean_recall_EPFL, rstd_recall_EPFL, mean_Fmeasure_EPFL, rstd_Fmeasure_EPFL, mean_NMI_EPFL, rstd_NMI_EPFL);
fprintf('MUGL-o     |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_DR, rstd_precision_DR, mean_recall_DR, rstd_recall_DR, mean_Fmeasure_DR, rstd_Fmeasure_DR, mean_NMI_DR, rstd_NMI_DR);
fprintf('MUGL-l     |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_DRL, rstd_precision_DRL, mean_recall_DRL, rstd_recall_DRL, mean_Fmeasure_DRL, rstd_Fmeasure_DRL, mean_NMI_DRL, rstd_NMI_DRL);
fprintf('WRGL       |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_WRGL, rstd_precision_WRGL, mean_recall_WRGL, rstd_recall_WRGL, mean_Fmeasure_WRGL, rstd_Fmeasure_WRGL, mean_NMI_WRGL, rstd_NMI_WRGL);

if graph == 1
    fprintf('Gaussian graph | trace(L_0)=%d | dim=%d | num=%d | noise_level=%.2f ｜ nrepeat=%d\n', ...
        trace(L_0), dim, num, noise_level, nrepeat)
elseif graph == 2
    fprintf('ER graph | dim=%d | num=%d | noise_level=%.2f\n', dim, num, noise_level)
elseif graph == 3
    fprintf('PA graph | dim=%d | num=%d | noise_level=%.2f\n', dim, num, noise_level)
else
    fprintf('graph type not specified!\n')    
end

fprintf('model param: eta_WGRL = %g, epsilon_WGRL = %g, beta_WGRL = %g\n', eta_WGRL, epsilon_WGRL, beta_WGRL);
fprintf('algorithm param: step_size_WRGL = %g, max_iter_WRGL = %g, tol_WRGL = %g\n', step_size_WRGL, max_iter_WRGL, tol_WRGL);

fprintf('data param: num = %g, dim = %g, noise_level = %g, nrepeat = %g\n', num, dim, noise_level, nrepeat);
fprintf('model param: rho1_DR = %g, rho2_DR = %g\n', rho1_DR, rho2_DR);
fprintf('algorithm param: step_size = %g, max_iter = %g, eps = %g\n', step_size, max_iter, eps);

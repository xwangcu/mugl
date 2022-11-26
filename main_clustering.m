%% Common Parameters
random = 1000;
rng(random);
img_num_usps = 100;
img_num_coil20 = 200;
dataset = 2; % 1: USPS; 2: COIL20

%% Load Data
currentFolder = pwd;
if dataset == 1
    % get two variables 'fea' and 'gnd'
    load([currentFolder,'\data\USPS.mat']);    
    img_num = img_num_usps;
    signal_num = 256;
    class_num = 10;
elseif dataset == 2
    % get two variables 'fea' and 'gnd'
    load([currentFolder,'\data\COIL20.mat']); 
    img_num = img_num_coil20;
    signal_num = 1024;
    class_num = 20;
else
    fprintf('dataset not specified!\n')
end
img_total = size(fea, 1);

%% Common Algorithm Parameters
nrepeat = 10;
alg = 2; % 1: (LS-)PGD; 2: ADMM (solve VSGL/MUGL)
max_iter = 1e8;
step_size = 1e-3;
eps = 1e-3;
eps_primal = 1e-3;
eps_dual = 1e-3;

%% Initialization
Jaccard_kNNG = zeros(nrepeat,1); FMI_kNNG = zeros(nrepeat,1); RI_kNNG = zeros(nrepeat,1);
Jaccard_NR = zeros(nrepeat,1); FMI_NR = zeros(nrepeat,1); RI_NR = zeros(nrepeat,1);
Jaccard_SigRep = zeros(nrepeat,1); FMI_SigRep = zeros(nrepeat,1); RI_SigRep = zeros(nrepeat,1);
Jaccard_EPFL = zeros(nrepeat,1); FMI_EPFL = zeros(nrepeat,1); RI_EPFL = zeros(nrepeat,1);
Jaccard_DR = zeros(nrepeat,1); FMI_DR = zeros(nrepeat,1); RI_DR = zeros(nrepeat,1);
Jaccard_DRL = zeros(nrepeat,1); FMI_DRL = zeros(nrepeat,1); RI_DRL = zeros(nrepeat,1);
Jaccard_WRGL = zeros(nrepeat,1); FMI_WRGL = zeros(nrepeat,1); RI_WRGL = zeros(nrepeat,1);

%% repeat
tic
for ii = 1 : nrepeat
    
    fprintf('ii = %d\n', ii);
    ind_select = randperm(img_total, img_num);
    X = fea(ind_select,:);
    
    % k-NNG
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 8;
    options.WeightMode = 'HeatKernel';
    options.t = 0.5;
    W_kNNG = constructW(X,options);
    idx_kNNG = spectralcluster(W_kNNG,class_num,'Distance','precomputed');
    [Jaccard_kNNG(ii), FMI_kNNG(ii), RI_kNNG(ii)] = compute_cluster_index(idx_kNNG, gnd(ind_select), img_num);
    
    % GL-SigRep
    param.alpha = 0.05; 
    param.beta = 0.5; 
    param.N = img_num; 
    param.max_iter = 2; 
    [L_SigRep, ~, ~] = graph_learning_gaussian_dong(X, param);
    L_SigRep(abs(L_SigRep)<10^(-4)) = 0;
    w_SigRep = linear_operator_L2w(L_SigRep, img_num);
    W_SigRep = linear_operator_vec2mat(w_SigRep, img_num);
    idx_SigRep = spectralcluster(W_SigRep,class_num,'Distance','precomputed');
    [Jaccard_SigRep(ii), FMI_SigRep(ii), RI_SigRep(ii)] = compute_cluster_index(idx_SigRep, gnd(1:img_num), img_num);

    % VSGL
    if alg == 1
        [L_NR, ~, ~, ~] = graph_learning_PGD(X, 0, 0, 0, step_size, max_iter, eps);
    elseif alg == 2
        t_NR = 1e-2; tau1_NR = 1e-2; tau2_NR = 1e-2;
        [L_NR, ~] = graph_learning_ADMM(X, 0, 0, 0, t_NR, tau1_NR, tau2_NR, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    w_NR = linear_operator_L2w(L_NR, img_num);  
    W_NR = linear_operator_vec2mat(w_NR, img_num);
    idx_NR = spectralcluster(W_NR,class_num,'Distance','precomputed');
    [Jaccard_NR(ii), FMI_NR(ii), RI_NR(ii)] = compute_cluster_index(idx_NR, gnd(ind_select), img_num);
    
    % Log-Model
    rho_EPFL = 1; alpha_EPFL = 500;
    t_EPFL = 1e0; tau1_EPFL = 1e-4; tau2_EPFL = 1e-3;
    [L_EPFL, fval_EPFL] = graph_learning_ADMM_epfl(X, rho_EPFL, alpha_EPFL, t_EPFL, tau1_EPFL, tau2_EPFL, max_iter, eps_primal, eps_dual);
    w_EPFL = linear_operator_L2w(L_EPFL, img_num);
    W_EPFL = linear_operator_vec2mat(w_EPFL, img_num);
    idx_EPFL = spectralcluster(W_EPFL,class_num,'Distance','precomputed');
    [Jaccard_EPFL(ii), FMI_EPFL(ii), RI_EPFL(ii)] = compute_cluster_index(idx_EPFL, gnd(1:img_num), img_num);
    
    % MUGL-o
    rho1_DR = 0.1; rho2_DR = 0.2;
    if alg == 1
        [L_DR, ~, ~, ~] = graph_learning_PGD(X, rho1_DR, rho2_DR, 0, step_size, max_iter, eps);
    elseif alg == 2
        t_DR = 1e-2; tau1_DR = 1e-3; tau2_DR = 1e-2;
        [L_DR, ~] = graph_learning_ADMM(X, rho1_DR, rho2_DR, 0, t_DR, tau1_DR, tau2_DR, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    w_DR = linear_operator_L2w(L_DR, img_num);
    W_DR = linear_operator_vec2mat(w_DR, img_num);
    idx_DR = spectralcluster(W_DR,class_num,'Distance','precomputed');
    [Jaccard_DR(ii), FMI_DR(ii), RI_DR(ii)] = compute_cluster_index(idx_DR, gnd(ind_select), img_num);
    
    % MUGL-l
    rho1_DRL = 0.11; rho2_DRL = 0.18; alpha_DRL = 0.001;
    alg = 1; % 1: (LS-)PGD; 2: ADMM (solve VSGL/MUGL)
    if alg == 1
        [L_DRL, ~, ~, ~] = graph_learning_LSPGD(X, rho1_DRL, rho2_DRL, alpha_DRL, step_size, 0.1, 0.1, max_iter, eps);
    elseif alg == 2
        t_DRL = 1e-2; tau1_DRL = 1e-3; tau2_DRL = 1e-2;
        [L_DRL, ~] = graph_learning_ADMM(X, rho1_DRL, rho2_DRL, alpha_DRL, t_DRL, tau1_DRL, tau2_DRL, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    w_DRL = linear_operator_L2w(L_DRL, img_num);
    W_DRL = linear_operator_vec2mat(w_DRL, img_num);
    idx_DRL = spectralcluster(W_DRL,class_num,'Distance','precomputed');
    [Jaccard_DRL(ii), FMI_DRL(ii), RI_DRL(ii)] = compute_cluster_index(idx_DRL, gnd(ind_select), img_num);
    
    % WRGL
    eta_WGRL = 0.5; epsilon_WGRL = 2; beta_WGRL = 1;
    step_size = 1e-5; eps = 1e-6;
    [L_WRGL, fval] = graph_learning_WRGL(X, eta_WGRL, epsilon_WGRL, beta_WGRL, step_size, max_iter, eps);
    L_WRGL(abs(L_WRGL)<10^(-4)) = 0;
    w_WRGL = linear_operator_L2w(L_WRGL, img_num);  
    W_WRGL = linear_operator_vec2mat(w_WRGL, img_num);
    idx_WRGL = spectralcluster(W_WRGL,class_num,'Distance','precomputed');
    [Jaccard_WRGL(ii), FMI_WRGL(ii), RI_WRGL(ii)] = compute_cluster_index(idx_WRGL, gnd(ind_select), img_num);
end
toc

%% Calculate and Print Results
Jaccard_kNNG_mean = mean(Jaccard_kNNG); Jaccard_kNNG_rstd = std(Jaccard_kNNG)*100/Jaccard_kNNG_mean; 
FMI_kNNG_mean = mean(FMI_kNNG); FMI_kNNG_rstd = std(FMI_kNNG)*100/FMI_kNNG_mean; 
RI_kNNG_mean = mean(RI_kNNG); RI_kNNG_rstd = std(RI_kNNG)*100/RI_kNNG_mean;

Jaccard_NR_mean = mean(Jaccard_NR); Jaccard_NR_rstd = std(Jaccard_NR)*100/Jaccard_NR_mean; 
FMI_NR_mean = mean(FMI_NR); FMI_NR_rstd = std(FMI_NR)*100/FMI_NR_mean; 
RI_NR_mean = mean(RI_NR); RI_NR_rstd = std(RI_NR)*100/RI_NR_mean;

Jaccard_SigRep_mean = mean(Jaccard_SigRep); Jaccard_SigRep_rstd = std(Jaccard_SigRep)*100/Jaccard_SigRep_mean; 
FMI_SigRep_mean = mean(FMI_SigRep); FMI_SigRep_rstd = std(FMI_SigRep)*100/FMI_SigRep_mean; 
RI_SigRep_mean = mean(RI_SigRep); RI_SigRep_rstd = std(RI_SigRep)*100/RI_SigRep_mean;

Jaccard_EPFL_mean = mean(Jaccard_EPFL); Jaccard_EPFL_rstd = std(Jaccard_EPFL)*100/Jaccard_EPFL_mean; 
FMI_EPFL_mean = mean(FMI_EPFL); FMI_EPFL_rstd = std(FMI_EPFL)*100/FMI_EPFL_mean; 
RI_EPFL_mean = mean(RI_EPFL); RI_EPFL_rstd = std(RI_EPFL)*100/RI_EPFL_mean;

Jaccard_DR_mean = mean(Jaccard_DR); Jaccard_DR_rstd = std(Jaccard_DR)*100/Jaccard_DR_mean; 
FMI_DR_mean = mean(FMI_DR); FMI_DR_rstd = std(FMI_DR)*100/FMI_DR_mean; 
RI_DR_mean = mean(RI_DR); RI_DR_rstd = std(RI_DR)*100/RI_DR_mean;

Jaccard_DRL_mean = mean(Jaccard_DRL); Jaccard_DRL_rstd = std(Jaccard_DRL)*100/Jaccard_DRL_mean; 
FMI_DRL_mean = mean(FMI_DRL); FMI_DRL_rstd = std(FMI_DRL)*100/FMI_DRL_mean; 
RI_DRL_mean = mean(RI_DRL); RI_DRL_rstd = std(RI_DRL)*100/RI_DRL_mean;

Jaccard_WRGL_mean = mean(Jaccard_WRGL); Jaccard_WRGL_rstd = std(Jaccard_WRGL)*100/Jaccard_WRGL_mean; 
FMI_WRGL_mean = mean(FMI_WRGL); FMI_WRGL_rstd = std(FMI_WRGL)*100/FMI_WRGL_mean; 
RI_WRGL_mean = mean(RI_WRGL); RI_WRGL_rstd = std(RI_WRGL)*100/RI_WRGL_mean;

Jaccard_kNNG_mean = mean(Jaccard_kNNG); Jaccard_kNNG_rstd = std(Jaccard_kNNG)*100/Jaccard_kNNG_mean; 
FMI_kNNG_mean = mean(FMI_kNNG); FMI_kNNG_rstd = std(FMI_kNNG)*100/FMI_kNNG_mean; 
RI_kNNG_mean = mean(RI_kNNG); RI_kNNG_rstd = std(RI_kNNG)*100/RI_kNNG_mean;

if dataset == 1
    fprintf('USPS\n');
elseif dataset == 2
    fprintf('COIL-20\n');
end

fprintf('method    |    Jaccard    |    FMI    |    RI \n');
fprintf('kNNG      | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_kNNG_mean, Jaccard_kNNG_rstd, FMI_kNNG_mean, FMI_kNNG_rstd, RI_kNNG_mean, RI_kNNG_rstd);
fprintf('VSGL      | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_NR_mean, Jaccard_NR_rstd, FMI_NR_mean, FMI_NR_rstd, RI_NR_mean, RI_NR_rstd);
fprintf('GL-SigRep | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_SigRep_mean, Jaccard_SigRep_rstd, FMI_SigRep_mean, FMI_SigRep_rstd, RI_SigRep_mean, RI_SigRep_rstd);
fprintf('Log-Model | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_EPFL_mean, Jaccard_EPFL_rstd, FMI_EPFL_mean, FMI_EPFL_rstd, RI_EPFL_mean, RI_EPFL_rstd);
fprintf('MUGL-o    | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_DR_mean, Jaccard_DR_rstd, FMI_DR_mean, FMI_DR_rstd, RI_DR_mean, RI_DR_rstd);
fprintf('MUGL-l    | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_DRL_mean, Jaccard_DRL_rstd, FMI_DRL_mean, FMI_DRL_rstd, RI_DRL_mean, RI_DRL_rstd);
fprintf('WRGL      | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%%   | %.3f\x00B1%.2f%% \n', ...
        Jaccard_WRGL_mean, Jaccard_WRGL_rstd, FMI_WRGL_mean, FMI_WRGL_rstd, RI_WRGL_mean, RI_WRGL_rstd);

fprintf('random = %d\n', random); 
fprintf('k = %d\n', options.k);
fprintf('step_size = %g, max_iter = %g, eps = %g\n', step_size, max_iter, eps);
fprintf('model param: eta_WGRL = %g, epsilon_WGRL = %g, beta_WGRL = %g\n', eta_WGRL, epsilon_WGRL, beta_WGRL);

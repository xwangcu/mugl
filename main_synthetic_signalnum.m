%% Common Parameters
random = 1000;
rng(random);
dim = 20;
num = 50;
noise_level = 0.01;
graph = 2; % 1: Gaussian graph; 2: ER graph; 3: PA graph

%% Model Parameters
rho1_DR = 0.11;
rho2_DR =  3.0;

%% Algorithm Parameters
nrepeat = 50;
max_iter = 1e5;
alg = 1; % 1: (LS-)PGD; 2: ADMM (solve VSGL/MUGL)

% PGD parameters
step_size = 1e-3;
eps = 1e-8;

%% Initialize Performance Evaluation
precision_DR = zeros(nrepeat,1);
recall_DR = zeros(nrepeat,1);
Fmeasure_DR = zeros(nrepeat,1);
NMI_DR = zeros(nrepeat,1);

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
avg = rand(dim,1) * 1;
covariance = pinv(D);

%% Run Algorithms
mu_uncer_sum = 0;
Sigma_uncer_sum = 0;
for ii = 1 : nrepeat
    
    fprintf('ii=%d\n', ii);
    
    % generate graph signals
    gftcoeff = mvnrnd(zeros(1,dim),covariance,num);
    X = V*gftcoeff' + avg;
    X_noisy = X + noise_level*randn(size(X));
    
    mu = sum(X_noisy,2)/num;
    mu_uncer = (mu - avg)' * L_0 * (mu - avg);
    mu_uncer_sum = mu_uncer_sum + mu_uncer;
    Sigma = (X_noisy - repmat(mu,[1 num]))*(X_noisy - repmat(mu,[1 num]))'/num;
    Sigma_uncer = norm(Sigma - covariance,"fro");
    Sigma_uncer_sum = Sigma_uncer_sum + Sigma_uncer;

    % MUGL-o
    if alg == 1
        [L_DR, ~, ~, ~] = graph_learning_PGD(X_noisy, rho1_DR, rho2_DR, 0, step_size, max_iter, eps);
    elseif alg == 2
        [L_DR, ~] = graph_learning_ADMM(X_noisy, rho1_DR, rho2_DR, 0, t, tau1, tau2, max_iter, eps_primal, eps_dual);
    else
        fprintf('algorithm type not specified!\n')
    end
    L_DR(abs(L_DR)<10^(-4)) = 0;
    [precision_DR(ii),recall_DR(ii),Fmeasure_DR(ii),NMI_DR(ii),~] = graph_learning_perf_eval(L_0,L_DR);
end
mu_uncer_avg = mu_uncer_sum / nrepeat;
Sigma_uncer_avg = Sigma_uncer_sum / nrepeat;

%% Calculate and Print Results
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

fprintf('method     |    precision    |    recall    |    Fmeasure    |    NMI \n');
fprintf('MUGL-o     |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', mean_precision_DR, rstd_precision_DR, mean_recall_DR, rstd_recall_DR, mean_Fmeasure_DR, rstd_Fmeasure_DR, mean_NMI_DR, rstd_NMI_DR);
fprintf('mu_uncer_avg = %g, Sigma_uncer_avg = %g \n', mu_uncer_avg, Sigma_uncer_avg);

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

fprintf('num = %g, dim = %g, rho1_DR = %g, rho2_DR = %g\n', num, dim, rho1_DR, rho2_DR);

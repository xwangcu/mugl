%% Common Parameters
random = 1000;
rng(random);

perfVSrho = 2; % 1: performance vs rho1; 2: performance vs rho2
xaxis_log = 0;

noise_level = 0.01;
nrepeat = 50;

graph = 2; % 1: Gaussian graph; 2: ER graph; 3: PA graph
dim = 40;
num = 800;

%% Model Parameters
if xaxis_log == 0
    init_rho1 = 0; step_rho1 = 0.5; max_rho1 = 20; num_rho1 = ceil(max_rho1/step_rho1) + 1; rho1 = zeros(1,num_rho1);
    init_rho2 = 0; step_rho2 = 0.2; max_rho2 = 10; num_rho2 = ceil(max_rho2/step_rho2) + 1; rho2 = zeros(1,num_rho2);
elseif xaxis_log == 1
    init_rho1 = -4; step_rho1 = 0.2; max_rho1 = 100; num_rho1 = ceil((log10(max_rho1)-init_rho1)/step_rho1) + 1; rho1 = zeros(1,num_rho1);
    init_rho2 = -4; step_rho2 = 0.2; max_rho2 = 100; num_rho2 = ceil((log10(max_rho2)-init_rho2)/step_rho2) + 1; rho2 = zeros(1,num_rho2);
end

%% Algorithm Parameters
max_iter = 1e8;
alg = 1; % 1: (LS-)PGD; 2: ADMM (solve VSGL/MUGL)
 
% PGD parameters
step_size = 1e-3;
eps = 1e-3;

% ADMM parameters
t = 1e0;
tau1 = 1e-5;
tau2 = 1e-4;
eps_primal = 1e-3;
eps_dual = 1e-3;

%% Generate Sythetic Graphs
if graph == 1
    [A,XCoords, YCoords] = construct_graph(dim,'gaussian',0.75,0.5);
elseif graph == 2
    [A,XCoords, YCoords] = construct_graph(dim,'er',0.08);
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
if perfVSrho == 1
    precision_DR = zeros(nrepeat,num_rho1);
    recall_DR = zeros(nrepeat,num_rho1);
    Fmeasure_DR = zeros(nrepeat,num_rho1);
    NMI_DR = zeros(nrepeat,num_rho1);
    rho2_DR = 0;
    for kk = 1 : num_rho1
        if xaxis_log == 0
            rho1_DR = init_rho1 + step_rho1 * (kk-1);
        elseif xaxis_log == 1
            rho1_DR = 10^(init_rho1 + step_rho1 * (kk-1));
        else
            fprintf('xaxis_log not specified!\n');
        end
        rho1(kk) = rho1_DR;
        
        for ii = 1 : nrepeat
            % generate graph signals
            gftcoeff = mvnrnd(zeros(1,dim),covariance,num);
            X = V*gftcoeff' + avg;
            X_noisy = X + noise_level*randn(size(X));
            
            % MUGL-o
            if alg == 1
                [L_DR, ~, ~, ~] = graph_learning_PGD(X_noisy, rho1_DR, rho2_DR, 0, step_size, max_iter, eps);
            elseif alg == 2
                [L_DR, ~] = graph_learning_ADMM(X_noisy, rho1_DR, rho2_DR, 0, t, tau1, tau2, max_iter, eps_primal, eps_dual);
            else
                fprintf('algorithm type not specified!\n');
            end
            L_DR(abs(L_DR)<10^(-4)) = 0;
            
            % evaluations
            [precision_DR(ii,kk),recall_DR(ii,kk),Fmeasure_DR(ii,kk),NMI_DR(ii,kk),~] = graph_learning_perf_eval(L_0,L_DR);
        end
        
        % Calculate and Print Results
        mean_precision_DR = mean(precision_DR(:,kk));
        mean_recall_DR = mean(recall_DR(:,kk));
        mean_Fmeasure_DR = mean(Fmeasure_DR(:,kk));
        mean_NMI_DR = mean(NMI_DR(:,kk));
        std_precision_DR = std(precision_DR(:,kk));
        std_recall_DR = std(recall_DR(:,kk));
        std_Fmeasure_DR = std(Fmeasure_DR(:,kk));
        std_NMI_DR = std(NMI_DR(:));
        rstd_precision_DR = std_precision_DR*100/mean_precision_DR;
        rstd_recall_DR = std_recall_DR*100/mean_recall_DR;
        rstd_Fmeasure_DR = std_Fmeasure_DR*100/mean_Fmeasure_DR;
        rstd_NMI_DR = std_NMI_DR*100/mean_NMI_DR;
        
        fprintf('rho1 = %.4f, rho2 = %.3f \n', rho1_DR, rho2_DR);
        fprintf('method     |    precision    |    recall    |    Fmeasure    |    NMI \n');
        fprintf('MUGL-o     |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', ...
            mean_precision_DR, rstd_precision_DR, mean_recall_DR, rstd_recall_DR, mean_Fmeasure_DR, ...
            rstd_Fmeasure_DR, mean_NMI_DR, rstd_NMI_DR);
    end
elseif perfVSrho == 2
    precision_DR = zeros(nrepeat,num_rho2);
    recall_DR = zeros(nrepeat,num_rho2);
    Fmeasure_DR = zeros(nrepeat,num_rho2);
    NMI_DR = zeros(nrepeat,num_rho2);
    rho1_DR = 0;
    for kk = 1 : num_rho2
        if xaxis_log == 0
            rho2_DR = init_rho2 + step_rho2 * (kk-1);
        elseif xaxis_log == 1
            rho2_DR = 10^(init_rho2 + step_rho2 * (kk-1));
        else
            fprintf('xaxis_log not specified!\n');
        end
        rho2(kk) = rho2_DR;
        
        for ii = 1 : nrepeat
            % generate graph signals
            gftcoeff = mvnrnd(zeros(1,dim),covariance,num);
            X = V*gftcoeff' + avg;
            X_noisy = X + noise_level*randn(size(X));
            
            % MUGL-o
            if alg == 1
                [L_DR, ~, ~, ~] = graph_learning_PGD(X_noisy, rho1_DR, rho2_DR, 0, step_size, max_iter, eps);
            elseif alg == 2
                [L_DR, ~] = graph_learning_ADMM(X_noisy, rho1_DR, rho2_DR, 0, t, tau1, tau2, max_iter, eps_primal, eps_dual);
            else
                fprintf('algorithm type not specified!\n')
            end
            L_DR(abs(L_DR)<10^(-4)) = 0;
            
            % evaluations
            [precision_DR(ii,kk),recall_DR(ii,kk),Fmeasure_DR(ii,kk),NMI_DR(ii,kk),~] = graph_learning_perf_eval(L_0,L_DR);
        end
        
        % Calculate and Print Results
        mean_precision_DR = mean(precision_DR(:,kk));
        mean_recall_DR = mean(recall_DR(:));
        mean_Fmeasure_DR = mean(Fmeasure_DR(:,kk));
        mean_NMI_DR = mean(NMI_DR(:));
        std_precision_DR = std(precision_DR(:,kk));
        std_recall_DR = std(recall_DR(:));
        std_Fmeasure_DR = std(Fmeasure_DR(:,kk));
        std_NMI_DR = std(NMI_DR(:,kk));
        rstd_precision_DR = std_precision_DR*100/mean_precision_DR;
        rstd_recall_DR = std_recall_DR*100/mean_recall_DR;
        rstd_Fmeasure_DR = std_Fmeasure_DR*100/mean_Fmeasure_DR;
        rstd_NMI_DR = std_NMI_DR*100/mean_NMI_DR;
        
        fprintf('rho1 = %.4f, rho2 = %.3f \n', rho1_DR, rho2_DR);
        fprintf('method     |    precision    |    recall    |    Fmeasure    |    NMI \n');
        fprintf('MUGL-o     |   %.3f\x00B1%.2f%%      %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%%     %.3f\x00B1%.2f%% \n', ...
            mean_precision_DR, rstd_precision_DR, mean_recall_DR, rstd_recall_DR, mean_Fmeasure_DR, ...
            rstd_Fmeasure_DR, mean_NMI_DR, rstd_NMI_DR);
    end
else
    fprintf("perfVSrho not correctly specified!");
end

%% Print & Plot
fprintf('graph = %d, dim = %d, num = %d \n', graph, dim, num);

if perfVSrho == 1
    figure
    boxplot(Fmeasure_DR, rho1, 'Symbol','')
    hold on
    plot(mean(Fmeasure_DR),'r','Linestyle','-','Marker','*','LineWidth',0.6)
    xlabel('$\rho_1$', 'interpreter', 'latex')
    ylabel('F-measure')
    xticks([1:4:41])
    if xaxis_log == 0
%         xticklabels({'0','0.2','0.4','0.6','0.8','1.0','1.2','1.4','1.6','1.8','2'})
        xticklabels({'0','2','4','6','8','10','12','14','16','18','20'})
    elseif xaxis_log == 1
        xaxisproperties= get(gca, 'XAxis');
        xaxisproperties.TickLabelInterpreter = 'latex';
        xticklabels({'$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$1$','$10$','$100$'})
    end
    set(gca,'FontName','Times New Roman','FontSize',15,'LineWidth',1);
elseif perfVSrho == 2
    figure
    boxplot(Fmeasure_DR, rho2, 'Symbol','')
    hold on
    plot(mean(Fmeasure_DR),'r','Linestyle','-','Marker','*','LineWidth',0.6)
    xlabel('$\rho_2$', 'interpreter', 'latex')
    ylabel('F-measure')
    xticks([1:5:51])
    if xaxis_log == 0
        xticklabels({'0','1','2','3','4','5','6','7','8','9','10'})
%         xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'})
    elseif xaxis_log == 1
        xaxisproperties= get(gca, 'XAxis');
        xaxisproperties.TickLabelInterpreter = 'latex';
        xticklabels({'$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$1$','$10$','$100$'})
    end
    set(gca,'FontName','Times New Roman','FontSize',15);
end

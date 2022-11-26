function [L_PGD,fval,fval_diff,time_pgd] = graph_learning_PGD(X_noisy, rho1, rho2, alpha, step_size, max_iter, epsilon)

DIM = size(X_noisy,1); % dim of graph signals
NUM = size(X_noisy,2); % num of graph signals
mean = sum(X_noisy,2) / NUM;
mean_rep = repmat(mean,1,NUM);
mean_meanT = mean * mean';
linadj_mean_meanT = linear_operator_w2L_adjoint(mean_meanT,DIM);
Cov = (X_noisy-mean_rep)*(X_noisy-mean_rep)'/NUM;
linadj_Cov = linear_operator_w2L_adjoint(Cov,DIM);

w = 1/((DIM-1)/2)*ones(DIM*(DIM-1)/2, 1);
fval = zeros(max_iter, 1);
fval_diff = zeros(max_iter, 1);
time_pgd = zeros(max_iter, 1);
for iter = 1 : max_iter
    
    tic
    sqrt_tmp = sqrt(w'*linadj_mean_meanT);
    L_PGD = linear_operator_w2L(w,DIM);
    time1 = toc;
    
    % compute function value
    fval(iter) = w'*linadj_Cov + (sqrt_tmp + rho1)^2 + rho2 * norm(L_PGD,'fro') - alpha*sum(log(diag(L_PGD)));
    
    tic
    hgrad = zeros(DIM,DIM);
    for i = 1:DIM
        ei = zeros(DIM,1); ei(i) = 1;
        hgrad = hgrad + ei*ei'/(L_PGD(i,i)+1e-15);
    end
    grad = linadj_Cov + (sqrt_tmp+rho1) * linadj_mean_meanT / sqrt_tmp ...
           + rho2 * linear_operator_w2L_adjoint(linear_operator_w2L(w,DIM),DIM) / norm(linear_operator_w2L(w,DIM),'fro') ...
           - alpha * linear_operator_w2L_adjoint(hgrad,DIM);
    w_old = w;
    w = w - step_size * grad;
    w = proj2simplex(w, DIM);
    time2 = toc;
    
    % calculate cpu time
    if iter == 1
        time_pgd(iter) = time1 + time2;
    elseif iter > 1
        time_pgd(iter) = time_pgd(iter-1) + time1 + time2;
    end
    
    % stopping criterion
    if iter > 1
        fval_diff(iter-1) = abs(fval(iter) - fval(iter-1));
    end
    diff = norm(w-w_old);
    if diff < epsilon
        break;
    end
end
L_PGD = linear_operator_w2L(w,DIM);

end

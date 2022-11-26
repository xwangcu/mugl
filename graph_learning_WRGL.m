function [L, fval] = graph_learning_WRGL(X_noisy, eta, epsilon, beta, step_size, max_iter, tol)

DIM = size(X_noisy,1); % dim of graph signals
NUM = size(X_noisy,2); % num of graph signals
mean = sum(X_noisy,2) / NUM;
mean_rep = repmat(mean,1,NUM);
Cov = (X_noisy-mean_rep)*(X_noisy-mean_rep)'/NUM;
linadj_Cov = linear_operator_w2L_adjoint(Cov,DIM);

w = 1/((DIM-1)/2)*ones(DIM*(DIM-1)/2, 1);
fval = zeros(max_iter, 1);
fval_diff = zeros(max_iter, 1);

for iter = 1 : max_iter
    L = linear_operator_w2L(w,DIM);
    
    % compute function value
    fval(iter) = w'*linadj_Cov + eta*norm(L,'fro')^2 + epsilon*norm(L,'fro') + beta*(trace(L)-DIM)^2;
    
    grad = linadj_Cov ... 
           + 2*eta*linear_operator_w2L_adjoint(L,DIM) ...
           + epsilon*linear_operator_w2L_adjoint(L,DIM)/norm(L,'fro') ...
           + 2*beta*(trace(L)-DIM)*linear_operator_w2L_adjoint(eye(DIM),DIM);
    w_old = w;
    w = w - step_size * grad;
    w = max(w,0);
    
    % stopping criterion
    if iter > 1
        fval_diff(iter-1) = abs(fval(iter) - fval(iter-1));
    end
    diff = norm(w-w_old);
    if diff < tol
        break;
    end
end
L = linear_operator_w2L(w,DIM);



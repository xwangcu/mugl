function [L_ADMM, fval_admm, fval_diff, time_admm] = graph_learning_ADMM(X, rho1, rho2, alpha, t, tau1, tau2, max_iter, eps_primal, eps_dual)

% min_{w,d} p'*w + (sqrt(q'*w)+rho1)^2 + rho2*||F(w)|| - alpha*1'*log(d)
% s.t.      Sw-d=0, 1'*w=m, w>=0

%% initialization
DIM = size(X,1); % dim of graph signals m
DIMw = DIM*(DIM-1)/2;
NUM = size(X,2); % num of graph signals n
mean = sum(X,2) / NUM;
mean_rep = repmat(mean,1,NUM);
mean_meanT = mean * mean';
Cov = (X-mean_rep)*(X-mean_rep)'/NUM;
p = linear_operator_w2L_adjoint(Cov,DIM);
q = linear_operator_w2L_adjoint(mean_meanT,DIM);
[S, St] = sum_squareform(DIM);
A = [S; ones(1,DIMw)];
At = [St, ones(DIMw,1)];
B = [-eye(DIM); zeros(1,DIM)];
c = [zeros(DIM,1);DIM];

%% iterations
w = ones(DIMw,1);
L_tmp = linear_operator_w2L(w,DIM);
d = diag(L_tmp);
y = ones(DIM+1,1);
fval_admm = zeros(max_iter,1);
fval_diff = zeros(max_iter,1);
time_admm = zeros(max_iter,1);
count = 0;
for k = 1 : max_iter
    
    % compute function value
    fval_admm(k) = p'*w + (sqrt(q'*w)+rho1)^2 + rho2*norm(linear_operator_w2L(w,DIM),'fro') - alpha*sum(log(d));
    
    tic
    
    % update w
    normw = norm(w);
    qtw = q'*w;
    normFw = norm(linear_operator_w2L(w,DIM),'fro');
    if normw == 0
%         fprintf('w = 0 when k = %d\n', k);
        count = count + 1;
        u = w - tau1*(p + At*y + t*At*(A*w+B*d-c));
    else
        u = w - tau1*(p + (sqrt(qtw)+rho1)*q/sqrt(qtw) ...
            + rho2*linear_operator_w2L_adjoint(linear_operator_w2L(w,DIM),DIM) / normFw ...
            + At*y + t*At*(A*w+B*d-c));
    end
    w = max(u, 0);
    
    % update d
    Sw = S*w;
    d_tmp = d;
    v = (1-tau2*t)*d + tau2*t*Sw + tau2*y(1:DIM);
    d = 0.5 * (v + sqrt(v.^2+4*alpha*tau2));
    
    % updata y
    y = y + t*([Sw-d;sum(w)] - c);
    
    rtime = toc;
    
    % calculate cpu time
    if k == 1
        time_admm(k) = rtime;
    elseif k > 1
        time_admm(k) = time_admm(k-1) + rtime;
    end
    
    % stopping criterion
    if k > 1
        fval_diff(k-1) = abs(fval_admm(k) - fval_admm(k-1));
    end
    primal_residual = norm(A*w+B*d-c);
    dual_residual = norm(t*A'*B*(d-d_tmp));
%     fprintf('k = %d, primal_residual = %.10f, dual_residual = %.10f\n', k, primal_residual, dual_residual);
    if primal_residual < eps_primal && dual_residual < eps_dual
%         fprintf('ADMM stopping critertion satisfied when k = %d\n', k);
        break;
    end
end
% fprintf('w=0 occurs for %d times\n', count);

W = linear_operator_vec2mat(w, DIM);
L_ADMM = diag(sum(W,2)) - W;

end

function [L_ADMM, fval_admm] = graph_learning_ADMM_epfl(X, rho, alpha, t, tau1, tau2, max_iter, eps_primal, eps_dual)

% min_{w,d} 2*z'*w - rho*ones'*log(d) + alpha*w'*w
% s.t.      Sw-d=0, w>=0

%% initialization

DIM = size(X,1);
DIMw = DIM*(DIM-1)/2;
Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       Z(i,j) = norm(X(i,:)-X(j,:),2)^2;
    end 
end
% z = linear_operator_mat2vec(Z, DIM);
z = squareform(Z)';
[S, St] = sum_squareform(DIM);

%% iterations

w = randn(DIMw,1);
d = randn(DIM,1);
y = randn(DIM,1);
fval_admm = zeros(max_iter,1);

for k = 1 : max_iter
    
    % update w
    p = w - tau1*(t*St*(S*w-d) + 2*alpha*w - St*y + 2*z);
    w = max(p, 0);
    
    % update d
    d_tmp = d;
    Sw = S*w;
    q = (1-tau2*t)*d + tau2*t*Sw - tau2*y;
    d = 0.5 * (q + sqrt(q.^2+4*rho*tau2));
    
    % updata y
    y = y - t*(Sw - d);
    
    % compute function value
    fval_admm(k) = 2*(z')*w - rho*sum(log(d)) + alpha*(w')*w;
    
    % stopping criterion
    primal_residual = norm(Sw-d);
    dual_residual = norm(-t*S'*(d-d_tmp));
%     fprintf('k = %d, primal_residual = %.10f, dual_residual = %.10f\n', k, primal_residual, dual_residual);
    
    if primal_residual < eps_primal && dual_residual < eps_dual
%         fprintf('stopping critertion satisfied when k = %d\n', k);
        break;
    end
end

W  = linear_operator_vec2mat(w, DIM);
L_ADMM = diag(sum(W,2)) - W;

% %% test
% fprintf('\n Sw-W1 = %f, Sw-d = %f \n', S*w-W*ones(DIM,1), S*w-d);
% fprintf('\n Sw-d norm = %f \n', norm(S*w-d));

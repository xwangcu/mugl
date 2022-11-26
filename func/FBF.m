function [w] = FBF(S, z, m, N, alpha, beta, gamma, maxItr, tol)

w = ones(m,1);
d = ones(N,1);

for iter = 1:maxItr
    
    y = w - gamma * (2*beta*w + S'*d);
    y_bar = d + gamma * (S*(w));
    p = max(0, y-2*gamma*z);
    p_bar = (y_bar-sqrt(y_bar.^2+4*alpha*gamma))/2;
    q = p - gamma * (2*beta*(p) + S'*(p_bar));
    q_bar = p_bar + gamma * (S*(p));
    
    w = w - y + q;
    d = max(d - y_bar + q_bar, 1e-4);
    
    rel_norm_primal = norm(- y + q, 'fro')/norm(w, 'fro');
    rel_norm_dual = norm(- y_bar + q_bar)/norm(d);
    
    fval(iter) = -alpha*sum(log(S*w))+ beta*norm(w)^2+2*z'*w;
    fprintf('iter:  %4d   primal: %.4e  dual: %6.4e  fval: %.4f\n', iter, rel_norm_primal, rel_norm_dual, fval(iter));
      
    if rel_norm_primal < tol && rel_norm_dual < tol
        break
    end
end
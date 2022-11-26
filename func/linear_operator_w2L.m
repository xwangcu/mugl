function [L] = linear_operator_w2L(w, m)

d = size(w,1);
% fprintf('linear_operator_w2L: m = %d, d=%d\n', m, d);
if d ~= m*(m-1)/2
    fprintf('linear_operator_w2L: linear_operator size mismatch\n')
end

L = zeros(m,m);

for i = 1 : m
    for j = 1 : m
        if i > j
            k = i - j + 0.5*(j-1)*(2*m-j);
            L(i,j) = -w(k);
        elseif i < j
            k = j - i + 0.5*(i-1)*(2*m-i);
            L(i,j) = -w(k);
        else
            L(i,i) = 0;
        end
    end
end

d = - sum(L, 2);
for i = 1 : m
    L(i,i) = d(i);
end

end
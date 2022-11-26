function [w] = linear_operator_w2L_adjoint(L, m)

d = m*(m-1)/2;
w = zeros(d,1);

for i = 2 : m
    for j = 1 : i-1
        k = i - j + 0.5*(j-1)*(2*m-j);
        w(k) = L(i,i) - L(i,j) - L(j,i) + L(j,j);
    end
end

end
function [M] = linear_operator_vec2mat(v, m)

d = size(v,1);
% fprintf('linear_operator_w2L: m = %d, d=%d\n', m, d);
if d ~= m*(m-1)/2
    fprintf('linear_operator_w2L: linear_operator size mismatch\n')
end

M = zeros(m,m);

for i = 1 : m
    for j = 1 : m
        if i > j
            k = i - j + 0.5*(j-1)*(2*m-j);
            M(i,j) = v(k);
        elseif i < j
            k = j - i + 0.5*(i-1)*(2*m-i);
            M(i,j) = v(k);
        else
            M(i,i) = 0;
        end
    end
end

end
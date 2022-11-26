function x = proj2simplex(y, scale)
% project an n-dim vector y to the simplex Dn
% Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = scale}

% Algorithm is explained as in the linked document
% http://arxiv.org/abs/1101.6081
% or
% http://ufdc.ufl.edu/IR00000353/
%
% Jan. 14, 2011.

m = length(y); bget = false;
s = sort(y,'descend'); tmpsum = 0;
for ii = 1:m-1
    tmpsum = tmpsum + s(ii);
    tmax = (tmpsum - scale)/ii;
    if tmax >= s(ii+1)
        bget = true;
        break;
    end
end
    
if ~bget 
    tmax = (tmpsum + s(m) -scale)/m; 
end
x = max(y-tmax,0);

end
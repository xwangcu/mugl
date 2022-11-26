function [Jaccard, FMI, RI] = compute_cluster_index(ind1,ind2,m)

SS = 0;
SD = 0;
DS = 0;
DD = 0;
for i = 1 : m-1
    for j = i+1 : m
        if ind1(i) == ind1(j) && ind2(i) == ind2(j)
            SS = SS + 1;
        elseif ind1(i) == ind1(j) && ind2(i) ~= ind2(j)
            SD = SD + 1;
        elseif ind1(i) ~= ind1(j) && ind2(i) == ind2(j)
            DS = DS + 1;
        else
            DD = DD + 1;
        end
    end
end
Jaccard = SS / (SS+SD+DS);
FMI = sqrt( SS^2 / ((SS+SD)*(SS+DS)) );
RI = 2*(SS+DD) / (m*(m-1));

end
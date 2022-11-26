function [precision,recall,f,NMI,num_of_edges,MCC] = graph_learning_perf_eval_mcc(L_0,L)
% evaluate the performance of graph learning algorithms

L_0tmp = L_0-diag(diag(L_0));
edges_groundtruth = squareform(L_0tmp)~=0;

Ltmp = L-diag(diag(L));
edges_learned = squareform(Ltmp)~=0;

num_of_edges = sum(edges_learned);

if num_of_edges > 0
    [precision,recall] = perfcurve(double(edges_groundtruth),double(edges_learned),1,'Tvals',1,'xCrit','prec','yCrit','reca');
    if precision == 0 && recall == 0
        f = 0;
    else
        f = 2*precision*recall/(precision+recall);
    end
    NMI = perfeval_clus_nmi(double(edges_groundtruth),double(edges_learned));
    tp = 0; tn = 0; fp = 0; fn = 0;
    for ii = 1:length(edges_groundtruth)
        if edges_groundtruth(ii) == 1 && edges_learned(ii) == 1
            tp = tp + 1;
        elseif edges_groundtruth(ii) == 0 && edges_learned(ii) == 0
            tn = tn + 1;
        elseif edges_groundtruth(ii) == 1 && edges_learned(ii) == 0
            fn = fn + 1;
        elseif edges_groundtruth(ii) == 0 && edges_learned(ii) == 1
            fp = fp + 1;
        end
    end
    MCC= (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
else
    precision = 0;
    recall = 0;
    f = 0;
    NMI = 0;
    MCC = 0;
end
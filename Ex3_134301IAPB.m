clear;
load('D.mat');

pairwise_dist = pdist(D,'euclidean');
sq_form = squareform(pairwise_dist);

linkage = linkage(pairwise_dist);

c = cophenet(linkage,pairwise_dist);

I = inconsistent(linkage);

clusters = cluster(linkage,'cutoff', 0.95);

dendrogram(linkage);

figure;
scatter3(D(:,1),D(:,2),clusters);
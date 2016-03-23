clear;
load('Student_134301IAPB_21.mat');


X = D(:,1:2);
[n,p] = size(X);
rng(3);

figure;
plot(X(:,1),X(:,2),'.','MarkerSize',15);
title('134301 generated dataset');
xlabel('1');
ylabel('2');

k = 4;
Sigma = {'diagonal','full'};
nSigma = numel(Sigma);
SharedCovariance = {true,false};
SCtext = {'true','false'};
nSC = numel(SharedCovariance);
d = 500;
x1 = linspace(min(X(:,1)) - 2,max(X(:,1)) + 2,d);
x2 = linspace(min(X(:,2)) - 2,max(X(:,2)) + 2,d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];
threshold = sqrt(chi2inv(0.99,2));
options = statset('MaxIter',1000); % Increase number of EM iterations
converged = nan(nSigma,1);

figure;
c = 1;
for i = 1:nSigma;
    for j = 1:nSC;
        gmfit = fitgmdist(X,k,'CovarianceType',Sigma{i},...
            'SharedCovariance',SharedCovariance{j},'Options',options);
        clusterX = cluster(gmfit,X);
        mahalDist = mahal(gmfit,X0);

        %splittingPoint = round(n*0.6);
        %sequenceStart = randperm(n);
        %sequence = transpose(sequenceStart);
            %splitting into two sets using randsample. 60-40
        %clusterXTrain=clusterX(:,sequence(1:splittingPoint));
        %clusterXValidation = clusterX(sequence(splittingPoint+1:end));
        subplot(2,2,c);
        h1 = gscatter(X(:,1),X(:,2),clusterX);
        hold on;
            for m = 1:k;
                idx = mahalDist(:,m)<=threshold;
                Color = h1(m).Color*0.75 + -0.5*(h1(m).Color - 1);
                h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
                uistack(h2,'bottom');
            end
        plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
        title(sprintf('Sigma is %s, SharedCovariance = %s',...
            Sigma{i},SCtext{j}),'FontSize',8)
        hold off
        c = c + 1;
    end
end
 %splitting into two sets using randsample. 60-40
cluster1=X(clusterX == 1);
cluster2=X(clusterX == 2);
cluster3=X(clusterX == 3);
cluster4=X(clusterX == 4);

cluster1N=size(cluster1,1);
cluster2N=size(cluster2,1);
cluster3N=size(cluster3,1);
cluster4N=size(cluster4,1);

splittingPoint1=round(cluster1N*0.5);
splittingPoint2=round(cluster2N*0.5);
splittingPoint3=round(cluster3N*0.5);
splittingPoint4=round(cluster4N*0.5);

sequence1 = transpose(randperm(cluster1N));
sequence2 = transpose(randperm(cluster2N));
sequence3 = transpose(randperm(cluster3N));
sequence4 = transpose(randperm(cluster4N));

cluster1Train=cluster1(sequence1(1:splittingPoint1));
cluster2Train=cluster2(sequence2(1:splittingPoint2));
cluster3Train=cluster3(sequence3(1:splittingPoint3));
cluster4Train=cluster4(sequence4(1:splittingPoint4));

cluster1Validate=cluster1(sequence1(splittingPoint1+1:end));
cluster2Validate=cluster2(sequence2(splittingPoint2+1:end));
cluster3Validate=cluster3(sequence3(splittingPoint3+1:end));
cluster4Validate=cluster4(sequence4(splittingPoint4+1:end));

cluster1Mdl = fitcknn(cluster1Train,cluster1Validate,'NumNeighbors',5,'Standardize',1);
cluster2Mdl = fitcknn(cluster2Train,cluster2Validate,'NumNeighbors',5,'Standardize',1);
cluster3Mdl = fitcknn(cluster3Train,cluster3Validate,'NumNeighbors',5,'Standardize',1);
cluster4Mdl = fitcknn(cluster4Train,cluster4Validate,'NumNeighbors',5,'Standardize',1);






silh = silhouette(D,clusterX);

[idx,C] = kmeans(D,4);
row1 = min(D(:,1)):0.01:max(D(:,1));
row2 = min(D(:,2)):0.01:max(D(:,2));
[row1G,row2G] = meshgrid(row1,row2);
XGrid = [row1G(:),row2G(:)];

idx2Region = kmeans(XGrid,4,'MaxIter',3,'Start',C);

figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title 'Generated set kmeans attempt';
xlabel '1';
ylabel '2';
legend('Region 1','Region 2','Region 3','Region 4','Data','Location','SouthEast');
hold off;
[cidx2,cmeans2] = kmeans(D,4,'dist','sqeuclidean');
[silh2,h] = silhouette(D,cidx2,'sqeuclidean');








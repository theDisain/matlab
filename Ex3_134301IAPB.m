clear;
load('D.mat');

model1=fitlm(D);

anova(model1);
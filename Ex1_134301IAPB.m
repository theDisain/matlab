
clear;
load('C:\Users\rapka\Downloads\Exercises\Exercise1\d_tree_data.mat');

structure = fitctree(x,y,'MinParentSize',1); %Make a classification tree with the minimal parent size of 1 to get all the options. Classification tree because of the binary values represented
view(structure,'Mode','graph');
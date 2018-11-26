%% =========== 1.Loading and Visualizing Data =============
load('ex4data1.mat');
m = size(X, 1);
sel = randperm(m);
sel = sel(1:100);
displayData(X(sel, :));
%% Train a one-vs-all multi-class classifier
function [theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);
theta = zeros(num_labels, n+1);
X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 50);
initial_theta = zeros(n + 1, 1);
for k = 1 : num_labels
	theta(k, :) = fmincg(@(t)(lrCostFunction(X, (y == k), t, lambda)), initial_theta, options);
end

end
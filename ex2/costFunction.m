%% Logistic Regression Cost Function
function [J, gradient] = costFunction(X, y, theta)

m = length(y);
J = 0;
gradient = zeros(size(theta));
J = -1 * sum(y .* log(sigmoid(X * theta)) + (1 - y) .* log(1 - sigmoid(X * theta))) / m;

gradient = (X' * (sigmoid(X * theta - y))) / m

endfunction
%% Regularized Logistic Regression Cost
function [J, gradient] = costFunctionReg (X, y, theta, lambda)

m = length(y);
J = 0;
gradient = zeros(size(theta));
theta_1 = [0; theta(2:end)]; %theta(1) 不参与正则化
J = -1 * sum(y .* log(sigmoid(X * theta)) + (1 - y) .* log(1 - sigmoid(X * theta))) / m + lambda/(2 * m) * theta_1' * theta_1;

gradient = (X' * (sigmoid(X * theta) - y)) / m + lambda / m * theta_1;

endfunction
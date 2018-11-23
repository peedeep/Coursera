%% Logistic regression cost function
function [J, gradient] = lrCostFunction (X, y, theta, lambda)

m = length(y);
J = 0;
gradient = zeros(size(theta));
theta_temp = [0; theta(2:end)];%theta(1) 不参与正则化，所以取零
J = -1 * sum(y .* log(sigmoid(X * theta)) + (1 - y) .* log(1 - sigmoid(X * theta))) / m + lambda * (theta_temp' * theta_temp) / (2 * m);

gradient = (X' * (sigmoid(X * theta) - y)) / m + lambda / m * theta_temp;

gradient = gradient(:);

endfunction
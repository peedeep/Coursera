function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));

theta_1 = [0; theta(2:end)];
J = sum((X * theta - y).^2) / (2 * m) + lambda / (2 * m) * theta_1' * theta_1;

grad = (X' * (X * theta -y)) / m + lambda / m * theta_1;

end
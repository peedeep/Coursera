function [theta, J_history] = gradientDescent (X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);
thetas = theta;
for iter = 1 : num_iters
	theta(1) = theta(1) - alpha / m * sum(X * thetas - y);
	theta(2) = theta(2) - alpha / m * sum((X * thetas - y) .* X(:,2));
	thetas = theta;
	J_history(iter) = computeCost(X, y, theta);
end

endfunction

%% Predict using a one-vs-all multi-class classifier
function p = predictOneVsAll(X, theta)

m = size(X, 1);
k = size(theta, 1);
p = zeros(m, 1);
X = [ones(m, 1) X];
[a, p] = max(sigmoid(X * theta'), [], 2);

end
%% Neural network prediction function
function p = predict(Theta1, Theta2, X)

m = size(X, 1);
k = size(Theta2, 1);
p = zeros(m, 1);

X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2]
a3 = sigmoid(a2 * Theta2');
[a, p] = max(a3, [], 2);

endfunction
%% Neural network prediction function
function p = predict(Theta1, Theta2, X)

m = size(X, 1);
k = size(Theta2, 1);
p = zeros(m, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[a, p] = max(h2, [], 2);

end
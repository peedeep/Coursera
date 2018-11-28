%% Logistic Regression Prediction Function
function p = predict(X, theta)

m = size(X, 1);
p = zeros(m, 1);
g = sigmoid(X * theta);
k = find(g >= 0.5);
p(k) = 1;

end
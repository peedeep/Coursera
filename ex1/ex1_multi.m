%% Exercise 1: Linear regression with multiple variables

%% ================ 1.Feature Normalization ================
clear; close all; clc

fprintf('Loading data ...\n');

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
X_temp = data(:, 1:2);

fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %0.f \n', [X(1:10, :), y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1), X];

%% ================ 2.Gradient Descent ================
fprintf('Running gradient descent...\n');
alpha = 0.01;
num_iters = 8500;
theta = zeros(3, 1);
[theta J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

figure;
plot(1: numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
fprintf('theta from gradient descent\n');
fprintf('theta: %f\n', theta);
fprintf('\n');

price = [1 (([1650 3] - mu) ./ sigma)] * theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%surf(X_temp(:, 1), X_temp(:, 2), repmat(X * theta, numel(X_temp(:, 1), numel(X_temp(:, 2)))));



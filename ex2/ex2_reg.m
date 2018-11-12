%% Octave/MATLAB script for the later parts of the exercise
%% Machine Learning Online Class - Exercise 2: Logistic Regression

clear; close all; clc;

data = load('ex2data2.txt');
X = data(:, [1, 2]);
y = data(:, 3);

plotData(X, y);

hold on;
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');

legend('y = 1', 'y = 0');
hold off;

%% =========== 1.Regularized Logistic Regression ============
X = mapFeature(X(:,1), X(:,2));
initail_theta = zeros(size(X, 2), 1);

lambda = 0;
[J, gradient] = costFunctionReg(X, y, initail_theta, lambda);
fprintf('J at initial theta (zeros): %f\n', J);
fprintf('Expected J (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', gradient(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= 2.Regularization and Accuracies =============
initial_theta = zeros(size(X, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exitFlag] = fminunc(@(t)costFunctionReg(X, y, t, lambda), initial_theta, options);

plotDecisionBoundary(X, y, theta);
hold on;
title(sprintf('lambda = %g', lambda))
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
zlabel('y')
hold off;

p = predict(X, theta);
fprintf('Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');


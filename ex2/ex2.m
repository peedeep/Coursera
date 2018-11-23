%% Machine Learning Online Class - Exercise 2: Logistic Regression

clear; close all; clc;

%% ==================== 1.Plotting ====================
data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

plotData(X, y)
xlabel('Exam 1 score') 
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted');
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============ 2.Compute Cost and Gradient ============
[m, n] =  size(X);
X = [ones(m, 1) X];
fprintf('size: %d', size(X));
initial_theta = zeros(n + 1, 1);
[J, gradient] = costFunction(X, y, initial_theta);

fprintf('Cost at initial theta (zeros): %f\n', J);
fprintf('Expected J (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', gradient);
fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

[J, gradient] = costFunction(X, y, [-24; 0.2; 0.2]);

fprintf('\nCost at test theta: %f\n', J);
fprintf('Expected J (approx): 0.218\n');
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', gradient);
fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============ 3.Optimizing using fminunc ============
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exitFlag] = fminunc(@(t)costFunction(X, y, t), initial_theta, options);

% Print theta to screen
fprintf('exitFlag: %f\n', exitFlag);
fprintf('J at theta found by fminunc: %f\n', J);
fprintf('Expected J (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');

plotDecisionBoundary(X, y, theta);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============ 4.Predict and Accuracies ============

prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

p = predict(X, theta);
fprintf('p: %d\n', p);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n')



%% Octave/MATLAB script that steps you through part 1
%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

clear; close all; clc

%% =========== 1.Loading and Visualizing Data =============
%% 随机画出100个样本
input_layer_size = 400;
num_labels = 10;
load('ex3data1.mat');
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 2.Vectorize Logistic Regression =============
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(X_t, y_t, theta_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 3.Training One-vs-All Logistic Regression =============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[theta] = oneVsAll(X, y, num_labels, lambda);
fprintf('size of theta: %d \n', size(theta));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ 4.Predict for One-Vs-All ============
pred = predictOneVsAll(X, theta);
fprintf('predict size of pred:%d \n', size(pred));
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


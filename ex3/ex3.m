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



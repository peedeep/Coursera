%% Octave/MATLAB script that steps you through part 2 Neural Networks

clear; close all; clc;

input_layer_size = 400;
hiddent_layer_size = 25;
k = 10;

%% =========== 1.Loading and Visualizing Data =============
load('ex3data1.mat');
m = size(X, 1);
sel = randperm(m);
sel = sel(1:100);
displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 2.Loading Pameters ============

load('ex3weights.mat');
% The matrices Theta1 and Theta2 will now be in your Octave
% environment
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

pred = predict(Theta1, Theta2, X);
fprintf('Train data Accuracy: %f\n', mean(double(pred == y)) * 100);

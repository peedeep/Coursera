%% Exercise 1: Linear regression with multiple variables

%% ================ 1.Feature Normalization ================
clear; close all; clc

fprintf('Loading data ...\n');

data = load('ex1data2.txt');
X = data(:, 1:2);
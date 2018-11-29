%% =========== 1.Loading and Visualizing Data =============
load('ex4data1.mat');
m = size(X, 1);
sel = randperm(m);
sel = sel(1:100);
displayData(X(sel, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========== 2.Loading Parameters =============
fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];


%% =========== 3.Compute Cost (Feedforward) =============
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

lambda = 0;

J = nnCostFunction(X, y, nn_params, lambda, input_layer_size, hidden_layer_size, num_labels);
fprintf(['Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========== 4.Implement Regularization =============
fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

lambda = 1;

J = nnCostFunction(X, y, nn_params, lambda, input_layer_size, hidden_layer_size, num_labels);
fprintf('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)\n', J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========== 5.Sigmoid Gradient =============
fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f \n\n', g);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 6.Initializing Pameters =============
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =========== 7.Implement Backpropagation =============
fprintf('\nChecking Backpropagation... \n');

checkNNGradients;

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 8.Implement Regularization =============
fprintf('\nChecking Backpropagation Regularization... \n');
lambda = 3;
checkNNGradients(lambda);

debug_J = nnCostFunction(X, y, nn_params, lambda, input_layer_size, hidden_layer_size, num_labels);
fprintf('debug J %f\n should be 0.576051\n', debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 9.Training NN =============
fprintf('\nTraining NN... \n');
options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p)nnCostFunction(X, y, p, lambda, input_layer_size, hidden_layer_size, num_labels);

[nn_params, J] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1: hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)):end), num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 10.Visualize Weights =============
fprintf('\nVisualize Weights... \n');
displayData(Theta1(:, 2:end));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== 11.Implement Predict =============
fprintf('\nImplement Predict... \n');

pred = predict(Theta1, Theta2, X);
fprintf('\n Training set Accuracy: %f', mean(double(pred == y)) * 100);



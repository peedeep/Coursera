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
%% =========== 8.Implement Regularization =============
%% =========== 9.Training NN =============
%% =========== 10.Visualize Weights =============
%% =========== 11.Implement Predict =============

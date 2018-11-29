%% Function to help check your gradients
function checkNNGradients(lambda)
%Creates a small neural network to check the backpropagation gradients
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;
    
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

X = debugInitializeWeights(m, input_layer_size - 1);
y = 1 + mod(1:m, num_labels)';

nn_params = [Theta1(:) ; Theta2(:)];

costFunc = @(p) nnCostFunction(X, y, p, lambda, input_layer_size, hidden_layer_size, num_labels);
[J, gradient] = costFunc(nn_params);

numgradient = computeNumerialGradient(costFunc, nn_params);
disp([numgradient gradient]);

diff = norm(numgradient - gradient) / norm(numgradient + gradient);
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
     
end
%% Neural network cost function
function [J, gradient] = nnCostFunction(X, y, nn_params, lambda, input_layer_size, hidden_layer_size, num_labels)

Theta1 = reshape(nn_params(1: hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;

%% 对y进行处理 Y(find(y==3))= [0 0 1 0 0 0 0 0 0 0]; 用于 Feedforward cost function 1�?2   
Y=[];
E = eye(num_labels);  
for i = 1 : num_labels
    Y0 = find(y == i);    % 找到等于y=i的序列号,替换向量
    Y(Y0,:) = repmat(E(i,:), size(Y0,1), 1);
end

%% regularized Feedforward cost function lambda=1
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1'); 
a2 = [ones(m, 1) a2];      
a3 = sigmoid(a2 * Theta2');  

Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];

temp1 = sum(Theta1_temp .^ 2);
temp2 = sum(Theta2_temp .^ 2);

cost = Y .* log(a3) + (1 - Y) .* log((1 - a3));
J= -1 / m * sum(cost(:)) + lambda/(2*m) * (sum(temp1(:)) + sum(temp2(:))); 

%% Graident
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for t = 1:m
	% step 1
	a_1 = X(t, :)';
	z_2 = Theta1 * a_1;
	a_2 = sigmoid(z_2);
	a_2 = [1; a_2];
	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);
	% step 2
	err_3 = zeros(num_labels, 1);
	for k = 1:num_labels
		err_3(k) = a_3(k) - (y(t) == k);
	end
	% step 3
	err_2 = Theta2' * err_3;
	err_2 = err_2(2:end) .* sigmoidGradient(z_2);
	% step 4
	delta_2 = delta_2 + err_3 * a_2';
	delta_1 = delta_1 + err_2 * a_1';
end

% step 5
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta1_grad = 1 / m * delta_1 + lambda/m * Theta1_temp;
Theta2_grad = 1 / m * delta_2 + lambda/m * Theta2_temp;

% Unroll gradients
gradient = [Theta1_grad(:) ; Theta2_grad(:)];

end
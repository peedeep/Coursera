%% Exercise 1: Linear Regression

%% ========== 1.Plotting ==========
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);
plotData(X, y);
fprintf('Program paused, Press enter to continue.\n');
pause;

%% ========== 2.Cost and Gradient descent ==========
X = [ones(m, 1), data(:, 1)];
theta = zeros(2, 1);
iterations = 2000;
alpha = 0.01;
J = computeCost(X, y, theta);
fprintf('size of X = %f\n', size(X));
fprintf('size of theta = %f\n', size(theta));
fprintf('With theta = [0; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

J = computeCost(X, y, [-1; 2]);
fprintf('With theta = [-1; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');
fprintf('Program paused, Press enter to continue.\n');
pause

[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');

hold on; % keep previous plot visible
plot(X(:,2), X * theta, '-');
legend('Training data', 'Linear regression');
hold off;

fprintf('%f\n', size(J_history));
figure; %open a new figure window
plot([1:size(J_history)], J_history);
pause;

predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;
fprintf('predict1:%f \n', predict1);
fprintf('predict2:%f \n', predict2);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========= 3.Visualizing J(theta_0, theta_1) ==========
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
	for j = 1:length(theta1_vals)
		t = [theta0_vals(i); theta1_vals(j)];
		J_vals(i,j) = computeCost(X, y, t);
	end
end

J_vals = J_vals'; %由于meshgrids在surf命令中的工作方式，我们需要在调用surf之前调换J_vals，否则轴会被翻转
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals); % surf命令绘制得到的是着色的三维曲面。
xlabel('\theta_0');ylabel('\theta_1');
% Contour plot
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20)); % contour用来绘制矩阵数据的等高线
xlabel('\theta_0');ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


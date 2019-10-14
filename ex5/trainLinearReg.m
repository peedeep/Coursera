function [theta] = trainLinearReg(X, y, lambda)

initail_theta = zeros(size(X, 2), 1);

costFunction = @(t)linearRegCostFunction(X, y, t, lambda);

options = optimset('MaxIter', 200, 'GradObj', 'on');

theta = fmincg(costFunction, initail_theta, options);

end
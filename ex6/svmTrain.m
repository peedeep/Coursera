function [model] = svmTrain(X, Y, C, kernelFunction, tol, max_passes)
if ~exist('tol', 'var') || isempty(tol)
    tol = le-3;
end
if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

end
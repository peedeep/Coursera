function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
fprintf(' X_norm %f  \n', X_norm);
sigma = std(X_norm);
fprintf('sigma  %f  \n', sigma);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end

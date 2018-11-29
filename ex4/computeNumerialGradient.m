%% Numerically compute gradients
function numgradient = computeNumerialGradient(J, theta)

numgradient = zeros(size(theta));
perturb = zeros(size(theta));

epsilon = 1e-4;
for i = 1:numel(theta)
    perturb(i) = epsilon;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    numgradient(i) = (loss2 - loss1) / (2*epsilon);
    perturb(i) = 0;
end

end
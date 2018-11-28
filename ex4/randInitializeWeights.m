%% Randomly initialize weights
function W = randInitializeWeights(L_in, L_out)

init_epsilon = 0.12;

W = zeros(L_out, L_in + 1);
W = rand(L_out, L_in + 1) * (2 * init_epsilon) - init_epsilon;

end
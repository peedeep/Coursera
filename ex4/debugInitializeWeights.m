%% Function for initializing weights
function W = debugInitializeWeights(L_out, L_in)
    
W = zeros(L_out, L_in + 1);
W = reshape(sin(1:numel(W)), size(W)) / 10;

end
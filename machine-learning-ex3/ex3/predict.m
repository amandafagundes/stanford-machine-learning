function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% First Layer (the input attributes and bias)
L1 = [ones(m, 1) X];

% Calculates the Z used in the hypotesis function
Z2 = L1 * Theta1';

% Calculates the result of the hypothesys function using Z (sigmoid)
H2 = 1 ./ (1 + exp(-Z2));

% The second layer inputs (the previously result and bias)
L2 = [ones(m, 1) H2];

% Calculates the Z using the theta parameters
Z3 = L2 * Theta2';

% Calculates the output hypothesis function (result)
H3 = 1 ./ (1 + exp(-Z3));

% Gets the index with the largest value (indicating the probabilty of the example belongs to that class)
[_,p] = max(H3, [], 2);

end

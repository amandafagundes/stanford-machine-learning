function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Each line represents a output vector of 0s and 1s (5000 x 10)
y_matrix = eye(num_labels)(y,:);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X];
% The weighted sum of input units (5000 x 401 * 401 x 25)
% The rows represents the examples and each column is the weighted sum of the attributes and the weights for each next layer unit
z2 =  a1 * Theta1';
% The sigmoig function of z2
a2 = 1 ./ (1 + exp(-z2));
% add 1 to the a2 first columns (5000 x 26)
a2 = [ones(size(a2,1), 1) a2];
% The witghted sum of hidden layer (5000 x 26 * 26 x 10)
z3 = a2 * Theta2';
% The sigmoig function of z3
a3 = 1 ./ (1 + exp(-z3));
% The output layer error
d3 = a3 - y_matrix;
% The hidden layer error
d2 = (d3 * Theta2(:,[2:end])) .* a2(:,[2:end]) .* (1 - a2(:,[2:end]));

% Calculates the cost function
for i = 1 : m
    % Get the expected vector values from y_matrix
    expected = y_matrix(i,:);
    % Get the calculated values from a3 layer (output)
    predicted = a3(i,:);
    % Sum of errors 
    % If the expected value is 1, the second part of the expression will be anulated
    % and if it is 0, the first part.
    % The predicted values contains the probability of the example belongs for each class
    %   ~> For big values, the log is lower
    %   ~> For small values, the log is higher
    %   ~> Thus, when higher the probability, lower the error
    J += -expected * log(predicted)' - (1 - expected) * log(1 - predicted)';
end

% The bias unit isn't regularized
Theta1_grad(:,1) = 1/m * (d2' * a1(:,1));
% Calculates the Theta1 gradient values
Theta1_grad(:, [2:end]) = 1/m * (d2' * a1(:, [2:end])) + (lambda/m) * Theta1(:, [2:end]);

% The bias unit isn't regularized
Theta2_grad(:,1) = 1/m * (d3' * a2(:,1));
% Calculates the Theta2 gradient values
Theta2_grad(:, [2:end]) = 1/m * (d3' * a2(:, [2:end])) + (lambda/m) * Theta2(:, [2:end]);

J = J/m + lambda/(2 * m) * (sum(sum((Theta1(:,[2:end]) .^2))) + sum(sum((Theta2(:,[2:end]) .^2))));


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

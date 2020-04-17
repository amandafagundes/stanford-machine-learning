function [J, grad, teste] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
z = X * theta;
h = 1 ./ (1 + exp(-z));

J = (1/m * (-y' * log(h)  - (1 - y)' * log(1 - h))) + lambda/(2 * m) * sum(theta([2:end]).^2);

grad(1) = 1/m * ((h - y)' * X(:, 1));
grad(2:end, 1) = (1/m * (X(:,2:end)' * (h - y))) + lambda/m .* theta(2:end);

% =============================================================

end
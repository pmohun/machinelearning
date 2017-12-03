function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta and set J to the cost.
% Compute the partial derivatives and set grad to the partial derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta); % hypothesis
J = (1/m)*((-y)'*log(h)-(1-y)'*log(1-h)); % cost function

grad = (1/m)*(X'*(h-y)); % grad should have the same dimensions as theta


end

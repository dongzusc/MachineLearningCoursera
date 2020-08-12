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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

J0 = 0; % unregularized cost function

h1 = sigmoid([ones(m, 1) X] * Theta1'); % a2
h2 = sigmoid([ones(m, 1) h1] * Theta2'); % predictions, = a3

for c = 1:num_labels
  
  y_p = h2(:,c); % prediction of class c in OneVsAll
  
  J0 = J0 + [(y==c)' * log(y_p) + (1-(y==c))' * log(1-y_p)]; %adding cost for each class
  
endfor

   % regularization part  
ThetaSqr = nn_params' * nn_params - Theta1(:,1)' * Theta1(:,1) - Theta2(:,1)' *Theta2(:,1);


J = -1.0/m * J0 + lambda/(2*m) * [ ThetaSqr ];
  
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for t = 1:m  % looping all training examples
  a1 = [1 X(t,:)]; % training data and biased term 1 by (feature+1);
  a2 = [1 h1(t,:)]; % the t-th example, using previous spaces, 1 by (hid + 1)
  a3 = h2(t,:); % (1 by K)
  yt = y(t); % true label for the t-th example, 1~K
  
  delta3 = a3; 
  delta3(yt) = delta3(yt) - 1; % delta3 = a3 - y_k, 1 by K
  
  delta2 = Theta2' * delta3' .* (a2.*(1-a2))'; % (hid+1) by K times K by 1 .* (hid+1) by 1;
  delta2 = delta2(2:end); % change to (hid) by 1
  
  Delta2 = Delta2 + delta3' * a2; % K by 1 times 1 by (hid+1)
  Delta1 = Delta1 + delta2 * a1; % (hid) by 1 times 1 by (feature+1)
  
endfor

Theta1_grad = 1.0/m * Delta1; 
Theta2_grad = 1.0/m * Delta2; % unregulated

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% removing the theta_0 terms
Theta1_removed = [zeros(hidden_layer_size,1) Theta1(:,2:end)];

Theta2_removed = [zeros(num_labels,1) Theta2(:,2:end)]; 

Theta1_grad = Theta1_grad + lambda/m * Theta1_removed;

Theta2_grad = Theta2_grad + lambda/m * Theta2_removed;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

%% classical SVM by Hinge loss
function [w, b, numSupportVectors] = Hinge_SVM(X, Y, learning_rate, epochs)
    % X - input feature vectors; assume rows are samples and columns are features
    % Y - true labels; a column vector of 1's and -1's
    % learning_rate - step size for gradient descent
    % epochs - number of passes over the dataset

    % Initialize weights and bias
    [n_samples, n_features] = size(X);
    w = zeros(n_features, 1);  % column vector
    b = 0;

    % Initialize support vectors count
    numSupportVectors = 0;

    % Gradient descent to minimize hinge loss
    for epoch = 1:epochs
        % Reset support vectors count for each epoch
        numSupportVectors = 0;
        
        for i = 1:n_samples
            if Y(i) * (X(i,:) * w + b) < 1
                % Misclassified or within margin, update weights and bias
                w = w + learning_rate * (X(i,:)' * Y(i));
                b = b + learning_rate * Y(i);
                
                % Increment the support vector count
                numSupportVectors = numSupportVectors + 1;
            end
        end
    end
end
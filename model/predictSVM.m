function predictions = predictSVM(X, w, b)
    % X - new data to predict on
    % w - learned weight vector from the SVM
    % b - learned bias term from the SVM

    predictions = sign(X * w + b);
end
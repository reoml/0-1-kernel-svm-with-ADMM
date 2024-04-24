function f_xj = computeFunction(xj, X, w, b, sigma)
%% function description:
%  the fcomputeFunction is to calculate the value of plane function
%     %input: 
%         X:[samples,feature]
%         xj:[samples,feature]
%       para:
%         w:[samples,1]
%         b:[1,1]
%         sigma:[1,1]
%     %output:
%         f_xj:[1,1]
    m = size(X, 1); % The number of data points in X
    % Compute the kernel values and perform the weighted sum
    sum_term = 0; % Initialize the sum to zero
    for i = 1:m
  %      sum_term = sum_term + w(i) * polynomial_kernel(xj, X(i, :), sigma);
        sum_term = sum_term + w(i) * gaussianKernel(xj, X(i, :)', sigma);
    end
    % Add the bias term
    f_xj = sum_term + b;
end
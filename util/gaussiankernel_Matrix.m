function K = gaussiankernel_Matrix(X, sigma)
%% function description:
% the gaussiankernel_Matrix function get the vector X and para sigma for
% input, get the kernel Matrix of X using gaussian kernel.
%input para:
    %X [samples,features]
    %sigma[1,1]
    % sigma is the standard deviation of the Gaussian kernel.
%output para:
    %K[samples,samples] K is the Gaussian kernel_Martix of X
    % Euclidean distances can be computed using the identity:
    % ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y'
    % Vectorized computation of squared Euclidean distance matrix
    XX = sum(X.^2, 2);
    YY = XX'; % Transpose for correct broadcasting
    distances = XX + YY - 2 * (X * X');
    % Compute the Gaussian kernel matrix
    K = exp(-distances / (2 * sigma^2));
end
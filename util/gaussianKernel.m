function K = gaussianKernel(x1, x2, sigma)
    % x1 and x2 are the vectors between which you want to compute the kernel.
    % sigma is the standard deviation of the Gaussian kernel.
    
    % Ensure that the vectors are column vectors
    x1 = x1(:);
    x2 = x2(:);
    
    % Compute the squared euclidean distance between the vectors
    distanceSquared = sum((x1 - x2) .^ 2);
    
    % Compute the Gaussian kernel
    K = exp(-distanceSquared / (2 * sigma^2));
end
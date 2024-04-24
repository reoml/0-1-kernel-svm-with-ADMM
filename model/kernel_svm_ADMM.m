%0/1 kernel SVM ADMM
function [w,b,u,lambda,tol_touple] = kernel_svm_ADMM(x,y,w_k,b_k,u_k,lambda_k,sigma,C,neta,para)
%% function description:
% the kernel_svm_ADMM function take para for input and get the output
% update para for a loop using ADMM algorithm for 0/1 svm
%     %input: 
%         x:[samples,feature]
%         y:[samples,1]
%         update para:
%           w_k:[samples,1]
%           b_k:[1,1]
%           u_k:[samples,1]
%           lambda_k:[samples,1]
%         Hyperpara:
%           sigma:[1,1]
%           para:[1,1]
%           C:[1,1]
%           neta:[1,1]
%     %output:
%         w:[samples,1]
%         b:[1,1]
%         u:[samples,1]
%         lambda:[sampes,1];
%         tol_touple:[4,1] save the tolerate condition
    %% data_process
    [samples,~] = size(x);
    I = ones(samples,1);
    K = gaussiankernel_Matrix(x,para);    
%     K = polynomialkernel_Matrix(x,para);  
    Dy = diag(y);
    A = Dy*K;
    z_k = I-A*w_k-b_k*y-lambda_k/sigma;
    %% get work set;
    index = get_support_vector(z_k,C,sigma);  
%     index_length = size(index,1)
    %% get u_k+1   %% prox method
        u        = z_k;                         %Tk complement part;
        u(index,:) = 0;                         %Tk part;
    %% get w_k+1    %% gradient
     v_k = -(u-I+b_k*y+lambda_k/sigma);
     w = sigma*(eye(samples)+sigma*K)\(Dy*v_k);
    %% get b_k+1    %% gradient
        b = dot(y,(I-u-A*w-lambda_k/sigma)) / dot(y,y);
    %% get lambda_k+1  %%dual ascent
        lambda = lambda_k + neta*sigma*(u-I+A*w+b*y);
        temp_array = zeros(size(lambda));
        temp_array(index,:) = lambda(index,:);
        lambda = temp_array;
    %% tolerate condition 
        tol_one = norm(w-w_k);
        tol_two = norm(u-u_k);
        tol_three =  norm(lambda-lambda_k);
        tol_four =  norm(b-b_k);
        tol_touple = [tol_one,tol_two,tol_three,tol_four];
end


function [ACC2]=test_iteration(X,y,Xt,yt,C,sigma,para,neta)
%% function description:
% the test_iteration function use the input to tarin the 0/1 kernel SVM
% and then use Xt,yt to test the acc of the model.
%     %input: 
%         X:[samples,feature]
%         y:[samples,1]
%         Xt:[samples,feature]
%         yt:[samples,1]
%         Hyperpara:
%           sigma:[1,1]
%           para:[1,1]
%           C:[1,1]
%           neta:[1,1]
%     %output:
%         ACC2:[1,1]
%% init para set
    [samples,feature]  = size(X);  
    max_iteration = 1000;
    tol_level = 0.001;
    index_test = [];
    [w_init,b_init,u_init,lambda_init] = deal(ones(samples,1)/100,0,zeros(samples,1),zeros(samples,1));
    [w_k,b_k,u_k,lambda_k] = deal(w_init,b_init,u_init,lambda_init);
    [tol_one,tol_two,tol_three,tol_four] = deal(100000,100000,100000,100000);
    %% iteration
    for k =1:max_iteration
        [w,b,u,lambda,tol_touple] = kernel_svm_ADMM(X,y,w_k,b_k,u_k,lambda_k,sigma,C,neta,para);
        if max(tol_touple)<tol_level
            disp('误差够小')
            break;
        end
    end
    ACC2    = accuracy(Xt,yt,w,b,X,y,para);
    fprintf('0/1核SVM： | %5.2f  |  %5.2f  |   %5.2f  | %3d  | \n',...
    C, sigma,para, ACC2);

end
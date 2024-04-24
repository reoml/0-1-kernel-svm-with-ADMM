function ACC = accuracy(X,y,w,b,x_train,y_train,para)
%% function description:
% the accuracy function is to calculate the acc of the 0/1 kernel svm.
%     %input: 
%         x_train:[samples,feature]
%         y_train:[samples,1]
%         X:[samples,feature]  test data
%         y:[samples,1]        test data
%         para:
%           w:[samples,1]
%           b:[1,1]
%         Hyperpara:
%           para:[1,1]
%     %output:
%         ACC1:[1,1]
[m,n] = size(X) ;
error = 0;
for j = 1:m
    f_xj = computeFunction(X(j,:), x_train, w, b, para);
    error= error +nnz(sign(f_xj)-y(j,:));
end
ACC = 1-1/(m)*(error);
end
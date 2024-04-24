function [ACC1] = classical_SVM(X,y,Xt,yt)
%% function description:
% the function classical_SVM use the input X,y to train the Hinge SVM,and then
% use Xt,yt to test the trained model.lastly give the acc for output.
%     %input: 
%         train data
%         X:[samples,feature]
%         y:[samples,1]
%         test data
%         Xt:[samples,feature]
%         yt:[samples,1]
%     %output:
%         ACC1:[1,1] acc of hinge svm
%% train model
[w_hinge,b_hinge, numSupportVectors] = Hinge_SVM(X, y, 1.618,1000);
%% test model
label= predictSVM(Xt,w_hinge,b_hinge);
%% calculate acc
error = 0;
for j = 1:size(yt,1)
    error= error +nnz(label(j,:)-yt(j,:));
end
ACC1 = 1-1/(size(yt,1))*(error);
fprintf('HingeSVM：  %3d   \n',...
   ACC1);
end
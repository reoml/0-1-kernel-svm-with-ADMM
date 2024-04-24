clear all; clc; close all;
load ionosphere
ionosphere = array2table(X);
ionosphere.Group = Y;
Z = cell2mat(Y);
Y1 = ones(size(Z,1),1);
for i =1:size(Z)
    if Z(i)=='g'
        Y1(i)=1;
    else
        Y1(i)=-1;
    end
end
X_ionosphere = X;
Y_ionosphere = Y1;
save('ionosphere.mat',"X_ionosphere","Y_ionosphere");
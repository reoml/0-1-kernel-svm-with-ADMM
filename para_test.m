clear all; clc; close all;
warning('off', 'all');
addpath(genpath(pwd));
type = {'liver','Austrain','ionosphere','sonar','wpbc'};
data_type = type{3};    
converse_svp =1;        %% whether converse support vector point
if strcmp(data_type, 'liver')   %%random seed
   rng(1);
elseif strcmp(data_type, 'Austrain')     
   rng(0);
elseif strcmp(data_type,'ionosphere')
   rng(0);
elseif strcmp(data_type,'sonar') 
   rng(0);
elseif strcmp(data_type,'wpbc')  
   rng(1);
end
%%acc save
ACC_array1 = zeros(10,1);
ACC_array2 = zeros(10,1);
for mean_iteration = 1:10
    %% dataset para set
    if strcmp(data_type, 'liver')      
        load liver.mat
        X = x_liver;
        Y = y_liver;
        [C,sigma,para,neta] = deal(2,0.500000000000000,4.00000000000000,1.61800000000000);
    elseif strcmp(data_type, 'Austrain')      
        load Austrain.mat
        [C,sigma,para,neta] = deal(32,0.500000000000000,2.82842712474619,1.61800000000000);
    elseif strcmp(data_type,'ionosphere')
        load ionosphere.mat
        X = X_ionosphere;
        X(:,2) = [];
        Y = Y_ionosphere;
        [C,sigma,para,neta] = deal(16,0.353553390593274,2.00000000000000,1.61800000000000);
    elseif strcmp(data_type,'sonar') 
        load sonar_dataset_x.mat
        load sonar_dataset_y.mat
        X = X_sonar;
        Y = cell2mat(Y_sonar);
        Z = ones(size(Y,1),1);
        for i =1:size(Y,1)
            if Y(i,1)=='R'
                Z(i,1)=-1;
            end
        end
        Y=Z;
        [C,sigma,para,neta] = deal(128,0.353553390593274,0.707106781186548,1.61800000000000);  %poly\gaussian 11.3137084989848 
    elseif strcmp(data_type,'wpbc')   % rng(1);
        load wpbc.mat
        y_wpbc = zeros(size(y,1),1);
        for i =1:size(y,1)
            if y{i}=='N'
                y_wpbc(i)= 1;
            else 
                y_wpbc(i)=-1;
            end
        end
        Y = y_wpbc;
        [C,sigma,para,neta] = deal(4,0.250000000000000,8.00000000000000,1.61800000000000); 
    end


%% data progress: delete Nan data
    nan_positions = isnan(X);
    linear_indices = find(isnan(X));
    [row, col] = find(isnan(X));
    X(row,:)=[];
    Y(row,:)=[];
%% norm and split train and test data
    X      = normalization(X,2);
    y      = Y;  
    y(y~=1)= -1;  
    [M,n]  = size(X);         

    % randomly split the data into training and testing data
    m  = ceil(0.9*M);  mt = M-m;       I  = randperm(M);
    Tt = I(1:mt);      Xtest = X(Tt,:);   ytest = y(Tt);   % testing  data 
    T  = I(1+mt:end);  Xtrain  = X(T,:);    ytrain  = y(T,:);  % training data
    [samples,feature]  = size(Xtrain);  
    if converse_svp
        converse_numbers = cast(samples*0.1,'uint8');
        rand_indice = randperm(m,converse_numbers);
        y(rand_indice,:)=-y(rand_indice,:);
    end
    [ACC1] = classical_SVM(Xtrain,ytrain,Xtest,ytest);
    [ACC2] = test_iteration(Xtrain,ytrain,Xtest,ytest,C,sigma,para,neta);
    ACC_array1(mean_iteration) =ACC1;
    ACC_array2(mean_iteration) =ACC2;
end                                %%mean_iteration
fprintf('hinge svm mean acc:  %3d \n', mean(ACC_array1));
fprintf('0/1 kernel svm mean acc: %3d \n',mean(ACC_array2))


% 
            
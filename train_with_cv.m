clear all; clc; close all;
warning('off', 'all');
addpath(genpath(pwd));
type = {'liver','Austrain','ionosphere','sonar','wpbc'};
data_type = type{3};
%% data load
if strcmp(data_type, 'liver')
    load liver.mat
    X = x_liver;
    Y = y_liver;
    [C,sigma,para,neta] = deal(2,0.500000000000000,11.3137084989848,1.61800000000000);
elseif strcmp(data_type, 'Austrain')      %%right
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
    [C,sigma,para,neta] = deal(128,0.353553390593274,0.707106781186548,1.61800);  %poly\gaussian 11.3137084989848 
elseif strcmp(data_type,'wpbc')
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
    [C,sigma,para,neta] = deal(4,0.250000000000000,11.3137084989848,1.61800000000000);
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
X      = normalization(X,2); % normalize the data
% randomly split the data into training and testing data
m  = ceil(0.9*M);  mt = M-m;       I  = randperm(M);
Tt = I(1:mt);      Xt = X(Tt,:);   yt = y(Tt);   % testing  data 
T  = I(1+mt:end);  X  = X(T,:);    y  = y(T,:);  % training data


%% para set
[samples,feature]  = size(X);  
numbers = linspace(0.01,0.2,30);
init_set = 0;
temp_ACC = 0;
C_scope = 7;
Sigma_scope = 7;
para_scope = 7;
K_fold = 10;
neta =  1.618;
sizeof_kfold =samples-floor(samples/K_fold);
[start_wk,start_bk,start_uk,start_lambdak] = deal(ones(sizeof_kfold,1)/100,0,zeros(sizeof_kfold,1),zeros(sizeof_kfold,1));
fprintf(' ------------------------------------------------------------------------\n');
fprintf('      C      sigma   para    ACC\n');
fprintf(' ------------------------------------------------------------------------\n');


for i          = -C_scope:1:C_scope                                        %%i    para of C
    C     = 2^i;
    for  j     = -Sigma_scope:1:Sigma_scope                                %%j    para of sigma
        sigma = sqrt(2)^j;
        for m = -para_scope:1:para_scope                                    %%m    para of para(Gussian para) 
            %% fold para set
            para = sqrt(2)^m;
            ACC_fold = zeros(K_fold,1);
            SV_array = zeros(K_fold,1);
            for k_fold =1:K_fold                                           %%fold
                %% fold slide
                cv = cvpartition(samples, 'KFold', K_fold);
                train_indices = cv.training(k_fold);
                validation_indices = cv.test(k_fold);
                X_train = X(train_indices, :);
                y_train = y(train_indices, :);
                X_val = X(validation_indices, :);
                y_val = y(validation_indices, :);
                %% para set
                [w_init,b_init,u_init,lambda_init] = deal(ones(size(X_train,1),1)/100,0,zeros(size(X_train,1),1),zeros(size(X_train,1),1));
                max_iteration = 1000;
                tol_level = 0.0001;
                [w_k,b_k,u_k,lambda_k] = deal(w_init,b_init,u_init,lambda_init);
                [tol_one,tol_two,tol_three,tol_four] = deal(100000,100000,100000,100000);
                %% main iteration
                for k =1:max_iteration                  
                  
                    [w,b,u,lambda,tol_touple] = kernel_svm_ADMM(X_train,y_train,w_k,b_k,u_k,lambda_k,sigma,C,neta,para);
                    if max(tol_touple)<tol_level
                        SV_point = size(find(u==0),1);
                        disp(['误差够小，支持向量的数量是: ', num2str(SV_point)]);
                        break;
                    end
                   
                end
                
                %%update para
                [w_k,b_k,u_k,lambda_k] = deal(w,b,u,lambda);
                if k==1000
                    SV_point = size(find(u==0),1);
                    disp(['到达1000次，支持向量的数量是: ', num2str(SV_point)]);
                    SV_array(k_fold) = SV_point;
                end
                

                ACC_fold(k_fold) =accuracy(X_val,y_val,w,b,X_train,y_train,para);
                fprintf(' Accuracy Across %d Folds: %5.2f\n', K_fold,ACC_fold(k_fold));
            end                                                            %%fold

            mean_ACC = mean(ACC_fold);
            mean_SV = mean(SV_array);
            fprintf('Average Accuracy Across %d Folds: %5.2f\n', k, mean_ACC);
            %% save max acc and para
            if temp_ACC <=mean_ACC
                temp_ACC = mean_ACC;
                touple = {w_k,b_k,u_k,lambda_k,C,sigma,para,neta};
            end
        end                                                                %%m      para of para(Gussian para)
    end                                                                    %%j      para of sigma
end                                                                        %%i      para of C


%% get index function
function index = get_support_vector(data,para_C,para_sigma)
%% function description: 
% the get_support_vector is to find the Tk set point(SV point) index by the
% formula
%%input para:
    %data [samples,features]
    %para_C constant [1,1]
    %para_sigma constant [1,1]
%%output para:
    %index:[Tk,1]
    [samples,~] = size(data);
    Nm = 1:samples;
    lower_bound = 0;
    upper_bound = sqrt(2 * para_C / para_sigma);
    index = Nm(data > lower_bound & data <= upper_bound);
    index = index';
end
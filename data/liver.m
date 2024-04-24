clear all; clc; close all;
load x_liver.mat
load y_liver.mat
x_liver{1,2}
for i = 1:size(y_liver,1)
    if strcmp(x_liver{i,2}, 'Female')
        x_liver{i,2} = 1;
    else
        x_liver{i,2}= -1;
    end
end
converted_x_liver = x_liver;  % Copy the original cell array

% Loop through each element of the cell array
for i = 1:numel(converted_x_liver)
    if isnumeric(converted_x_liver{i})
        % If the element is numeric, convert it to double
        converted_x_liver{i} = double(converted_x_liver{i});
    end
    % If the element is not numeric, leave it unchanged
end


x_liver = cell2mat(converted_x_liver);
y_liver = double(y_liver);
save('liver.mat',"x_liver","y_liver");
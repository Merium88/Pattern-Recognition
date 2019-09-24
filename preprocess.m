function [norm,log_data,bin1,bin2] = preprocess(data)
%Normalize data
norm = zscore(data,1,1);

% Take log of data
log_data = logarithm(data);

%Binarize data
bin1 = binarize(data);
%Additional pre-processing strategy which binarizes normalized data
bin2 = binarize(norm);
end

function log_data = logarithm(data)
for i=1:1:length(data(:,1))
   log_data(i,:) = log(data(i,:)+0.1); 
end
end

function bin = binarize(norm_data)
for i=1:1:length(norm_data(:,1))
    for j=1:1:length(norm_data(1,:))
        if(norm_data(i,j)>0)
            bin(i,j) = 1;
        else
            bin(i,j) = 0;
        end
    end
end
end
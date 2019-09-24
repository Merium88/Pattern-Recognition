function [etest,sen,spec] = KNN (xtest,xtrain,ytrain,ytest,bin_flag)
%Calculate euclidean distance/hamming distance based on bin_flag for all testing data
counter=1;
if(bin_flag==0)
distance = pdist2(xtest,xtrain,'euclidean');
else
distance= pdist2(xtest,xtrain,'hamming');
end

%Sort the rows of matrix
[Sorted,sorted_index] = sort(distance,2,'ascend'); %1536x3065 matrix (each column represents neighbours based on distance)

for K=1:1:10
  %Sum of selected indices from ytrain will give Kc
     class1_score = sum(ytrain(sorted_index(:,1:K)),2)/K;
     class2_score = 1-class1_score;
   %compare class1 and class2 score and 
     Y_test(counter,:) = double(gt(class1_score,class2_score));
     
%Estimate Error
etest(counter,:) = error_est(Y_test(counter,:),ytest');
CP = confusionmat(Y_test(counter,:),ytest');
sen(counter,:) = CP(1,1)*100/(CP(1,1)+CP(2,1));
spec(counter,:) = CP(2,2)*100/(CP(1,2)+CP(2,2));
counter = counter+1;
end

for K=15:5:100
  %Sum of selected indices from ytrain will give Kc
     class1_score = sum(ytrain(sorted_index(:,1:K)),2)/K;
     class2_score = 1-class1_score;
   
     
      Y_test(counter,:) = double(gt(class1_score,class2_score));
  
    
%Estimate Error
etest(counter,:) = error_est(Y_test(counter,:),ytest');
CP = confusionmat(Y_test(counter,:),ytest');
sen(counter,:) = CP(1,1)*100/(CP(1,1)+CP(2,1));
spec(counter,:) = CP(2,2)*100/(CP(1,2)+CP(2,2));
counter = counter+1;
end

% 
 end
function [etest,etrain,sen,spec] = Gaussian_Naive(x_test,x_train,ytrain,ytest)
%% TRAINING OF NAIVE BAYES CLASSIFIER
%Separate data for spam and non-spam emails
class1 = find(ytrain==1);
class2 = find(ytrain==0);

%Use ML to calculate prior for both classes Y=0, Y=1
%P = NB_YPrior(ytrain)
P = length(class1)/length(ytrain); 

%Estimate mean and variance for each feature of X using ML
avg_class1 = sum(x_train(class1,:),1)/length(class1); %mean of all features in x, belonging to class 1
avg_class2 = sum(x_train(class2,:),1)/length(class2); 
for i =1:1:length(x_train(1,:))
var_class1(i) = sum((x_train(class1,i)- avg_class1(i)).^2,1)/length(class1);%var for each feature with class 1
var_class2(i) = sum((x_train(class2,i)- avg_class2(i)).^2,1)/length(class2);
end

%% TESTING OF NAIVE BAYES CLASSIFIER
%Calculate Y_Hat=YGivenX(D,P,binary_x_train)for test data
% Estimate Posterior for Training Data
Y_Test = GNB_testing(x_test,avg_class1,avg_class2,var_class1,var_class2,P);
%Calculate Y_Hat=YGivenX(D,P,binary_x_train)for train data
Y_Train = GNB_testing(x_train,avg_class1,avg_class2,var_class1,var_class2,P);


%% Error estimation with testing data
%Estimate Error
etest = error_est(Y_Test,ytest');
etrain = error_est(Y_Train,ytrain');
CP = confusionmat(Y_Test,ytest');
sen = CP(1,1)*100/(CP(1,1)+CP(2,1));
spec = CP(2,2)*100/(CP(1,2)+CP(2,2));
end

%Gaussian Naive Bayes Testing Function
function Y_hat = GNB_testing(x_data,avg_class1,avg_class2,var_class1,var_class2,P)
%%Posterior Prediction for test data
for i =1:1:length(x_data(:,1))
    for j=1:1:length(x_data(1,:))
      class1_prob(j) = (-0.5*((x_data(i,j)- avg_class1(j)).^2)/var_class1(j)) - log((2*pi*var_class1(j))^0.5);
      class2_prob(j) = (-0.5*((x_data(i,j)- avg_class2(j)).^2)/var_class2(j)) - log((2*pi*var_class2(j))^0.5);
    end
      class1_score = sum(class1_prob,2) + log(P);
      class2_score = sum(class2_prob,2) + log(1-P); 
      
      if class1_score>class2_score 
        Y_hat(i) = 1;
      else
        Y_hat(i) = 0;
      end
end
end
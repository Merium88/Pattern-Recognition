function [etest,etrain,sen,spec] = BB_Naive(x_test,x_train,ytrain,ytest,alpha)
%% TRAINING OF NAIVE BAYES CLASSIFIER
%Separate data for spam and non-spam emails
class1 = find(ytrain==1);
class2 = find(ytrain==0);

%Use ML to calculate prior for both classes Y=0, Y=1
%P = NB_YPrior(ytrain)
P = length(class1)/length(ytrain); 
a = sum(x_train(class1,:),1);
%Estimate likelihood of x given theta, with a beta prior distribution
%D = NB_XGivenY(binary_x_train,yTrain) with beta distribution where a=b
D = [(sum(x_train(class1,:),1)+alpha)/(length(class1)+(2*alpha));
    (sum(x_train(class2,:),1)+alpha)/(length(class2)+(2*alpha))];
ans = max(D);
%% TESTING OF NAIVE BAYES CLASSIFIER
%Calculate Y_Hat=YGivenX(D,P,binary_x_train)for test data
% Estimate Posterior for Training Data
Y_Test = NB_testing(x_test,D,P);
%Calculate Y_Hat=YGivenX(D,P,binary_x_train)for train data
Y_Train = NB_testing(x_train,D,P);


%% Error estimation with testing data
%Estimate Error
etest = error_est(Y_Test,ytest');
etrain = error_est(Y_Train,ytrain');
CP = confusionmat(Y_Test,ytest');
sen = CP(1,1)*100/(CP(1,1)+CP(2,1));
spec = CP(2,2)*100/(CP(1,2)+CP(2,2));
% plotconfusion(Y_Test,ytest');

end

function Y_hat = NB_testing(x_data,D,P)
for i=1:1:length(x_data(:,1))
    %First estimate posterior mean of thetajc for each new testing data x 
  class1_prob = x_data(i,:).*log(D(1,:)+eps) + (1-x_data(i,:)).*log(1-D(1,:)+eps); 
  class2_prob = x_data(i,:).*log(D(2,:)+eps) + (1-x_data(i,:)).*log(1-D(2,:)+eps); 
  
  %Second take log of posterior mean of thetajc AND log of posterior mean
  %of class pi_c and sum
  ans = sum(class1_prob,2);
  class1_score = sum(class1_prob,2)+log(P); %
  class2_score = sum(class2_prob,2)+log(1-P);
  
  %Estimate class label based on predicted scores, count is a counter for
  %each value of alpha
  if class1_score>class2_score
     Y_hat(i) = 1;
  else
     Y_hat(i) = 0;
  end
end
end
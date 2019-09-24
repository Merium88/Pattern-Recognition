function [etest,etrain,sen,spec] = logistic_regression(x_test,x_train,ytrain,ytest,lambda)
%Concatenate 1 to the start of X
 norm_XTrain = [ones(length(x_train(:,1)),1),x_train];
 norm_XTest = [ones(length(x_test(:,1)),1),x_test];

 %Set Initial values of weight
 weight = zeros(length(norm_XTrain(1,:)),1);
 error = 1;
 %Newtons Method
while error > 0.0001
sigmoid = exp(-weight'*norm_XTrain');
for i=1:1:length(norm_XTrain(:,1))
sigm(i) = 1/(1+sigmoid(i));
end

%Determine g and H
g = norm_XTrain'*(sigm'-ytrain) + [0;weight(2:end).*lambda];
S = diag(sigm.*(ones(size(sigm))-sigm));
lam = lambda*eye(length(norm_XTrain(1,:)));
lam(1,1) = 0;
H = norm_XTrain'*S*norm_XTrain + lam;

%Update weights    
weight_new = weight - inv(H)*g;
error =mean( abs((weight_new-weight)./weight));
weight = weight_new;
end

% Estimate Posterior for Training Data
Y_Test = posterior(norm_XTest,weight);
%Estimate Posterior of Testing Data
Y_Train = posterior(norm_XTrain,weight);

%Estimate Error
etest = error_est(Y_Test,ytest');
etrain = error_est(Y_Train,ytrain');
%Confusion Matrix
CP = confusionmat(Y_Test,ytest');
sen = CP(1,1)*100/(CP(1,1)+CP(2,1));
spec = CP(2,2)*100/(CP(1,2)+CP(2,2));
end

function Y_hat = posterior(x_data,weight)
sigmoid = exp(-weight'*x_data');
%For each sample of X compute Y
for i=1:1:length(x_data(:,1))
sigm_test(i) = 1/(1+sigmoid(i));

class1_score = sigm_test(i);
class2_score = 1-sigm_test(i);

%Classify Y
if class1_score>class2_score%class2_score 
  Y_hat(i) = 1;
else
  Y_hat(i) = 0;
end

end
end



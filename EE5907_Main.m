clear all; clc
%Load Data
data=load('spamData.mat');
%% Data Preprocessing
% 1. Preprocessing of X_train data and X_test data
[norm_train,log_train,bin1_train,bin2_train] = preprocess(data.Xtrain);
[norm_test,log_test,bin1_test,bin2_test] = preprocess(data.Xtest);

%% Q1. Implement Beta-Bernoulli Naive Bayes 
count=1;
for alpha = 0:0.5:100
[BB_Naive_eTest(count,:),BB_Naive_eTrain(count,:),BB_Sensitivity(count,:),BB_Specificity(count,:)] = BB_Naive(bin1_test,bin1_train,data.ytrain,data.ytest,alpha);
count = count +1;
end

% Plot Results for Q1
plot((0:0.5:100),BB_Naive_eTest);
hold on
plot((0:0.5:100),BB_Naive_eTrain);
title('Plots of training and test error rates versus alpha')
xlabel('0 < alpha < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot((0:0.5:100),BB_Sensitivity);
hold on
plot((0:0.5:100),BB_Specificity);
title('Plots of training and test error rates versus alpha')
xlabel('0 < alpha < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')

%% Q2. Implement Gaussian Naive Bayes
%Implement Gaussian Naive Bayes for normalized data 
[Gaus_Naive_eTest_norm,Gaus_Naive_eTrain_norm,Gaus_Sensitivity_norm,Gaus_Specificity_norm] = Gaussian_Naive(norm_test,norm_train,data.ytrain,data.ytest);

%Implement Gaussian Naive Bayes for normalized data 
[Gaus_Naive_eTest_log,Gaus_Naive_eTrain_log,Gaus_Sensitivity_log,Gaus_Specificity_log] = Gaussian_Naive(log_test,log_train,data.ytrain,data.ytest);

%% Q3. Implement Logistic Regression
% Implement for normalized data varying values of lambda
for lambda = 1:1:10
[Logistic_eTest_norm(lambda,:),Logistic_eTrain_norm(lambda,:),Logistic_Sensitivity_norm(lambda,:),Logistic_Specificity_norm(lambda,:)] = logistic_regression(norm_test,norm_train,data.ytrain,data.ytest,lambda);
end
counter = 11;
for lambda = 15:5:100
[Logistic_eTest_norm(counter,:),Logistic_eTrain_norm(counter,:),Logistic_Sensitivity_norm(counter,:),Logistic_Specificity_norm(counter,:)] = logistic_regression(norm_test,norm_train,data.ytrain,data.ytest,lambda);
counter=counter +1;
end

% Implement for logarithm data varying values of lambda
for lambda = 1:1:10
[Logistic_eTest_log(lambda,:),Logistic_eTrain_log(lambda,:),Logistic_Sensitivity_log(lambda,:),Logistic_Specificity_log(lambda,:)] = logistic_regression(log_test,log_train,data.ytrain,data.ytest,lambda);
end
counter = 11;
for lambda = 15:5:100
[Logistic_eTest_log(counter,:),Logistic_eTrain_log(counter,:),Logistic_Sensitivity_log(counter,:),Logistic_Specificity_log(counter,:)] = logistic_regression(log_test,log_train,data.ytrain,data.ytest,lambda);
counter=counter +1;
end

% Implement for binarized data varying values of lambda
for lambda = 1:1:10
[Logistic_eTest_bin(lambda,:),Logistic_eTrain_bin(lambda,:),Logistic_Sensitivity_bin(lambda,:),Logistic_Specificity_bin(lambda,:)] = logistic_regression(bin1_test,bin1_train,data.ytrain,data.ytest,lambda);
end
counter = 11;
for lambda = 15:5:100
[Logistic_eTest_bin(counter,:),Logistic_eTrain_bin(counter,:),Logistic_Sensitivity_bin(counter,:),Logistic_Specificity_bin(counter,:)] = logistic_regression(bin1_test,bin1_train,data.ytrain,data.ytest,lambda);
counter=counter +1;
end

% Plot Results for Q3
figure
plot([(1:1:10),(15:5:100)],Logistic_eTest_norm);
hold on
plot([(1:1:10),(15:5:100)],Logistic_eTrain_norm);
xlabel('0 < lambda < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot([(1:1:10),(15:5:100)],Logistic_eTest_log);
hold on
plot([(1:1:10),(15:5:100)],Logistic_eTrain_log);
xlabel('0 < lambda < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot([(1:1:10),(15:5:100)],Logistic_eTest_bin);
hold on
plot([(1:1:10),(15:5:100)],Logistic_eTrain_bin);
xlabel('0 < lambda < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot([(1:1:10),(15:5:100)],Logistic_Sensitivity_norm);
hold on
plot([(1:1:10),(15:5:100)],Logistic_Specificity_norm);
xlabel('0 < lambda < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')

figure
plot([(1:1:10),(15:5:100)],Logistic_Sensitivity_log);
hold on
plot([(1:1:10),(15:5:100)],Logistic_Specificity_log);
xlabel('0 < lambda < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')

figure
plot([(1:1:10),(15:5:100)],Logistic_Sensitivity_bin);
hold on
plot([(1:1:10),(15:5:100)],Logistic_Specificity_bin);
xlabel('0 < lambda < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')
%% Implement KNN Algorithm 

%Estimate error in classification for normalized data
bin_flag = 0;
[KNN_eTest_norm,KNN_Sensitivity_norm,KNN_Specificity_norm] = KNN(norm_test,norm_train,data.ytrain,data.ytest,bin_flag);
KNN_eTrain_norm = KNN(norm_train,norm_train,data.ytrain,data.ytrain,bin_flag);

%Estimate error in classification for logarithm data
[KNN_eTest_log,KNN_Sensitivity_log,KNN_Specificity_log] = KNN(log_test,log_train,data.ytrain,data.ytest,bin_flag);
KNN_eTrain_log = KNN(log_train,log_train,data.ytrain,data.ytrain,bin_flag);

%Estimate error in classification for binarized data
bin_flag = 1;
[KNN_eTest_bin,KNN_Sensitivity_bin,KNN_Specificity_bin] = KNN(bin1_test,bin1_train,data.ytrain,data.ytest,bin_flag);
KNN_eTrain_bin = KNN(bin1_train,bin1_train,data.ytrain,data.ytrain,bin_flag);

% Plot Results for Q4
figure
plot([(1:1:10),(15:5:100)],KNN_eTest_norm);
hold on
plot([(1:1:10),(15:5:100)],KNN_eTrain_norm);
xlabel('0 < K < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot([(1:1:10),(15:5:100)],KNN_eTest_log);
hold on
plot([(1:1:10),(15:5:100)],KNN_eTrain_log);
xlabel('0 < K < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot([(1:1:10),(15:5:100)],KNN_eTest_bin);
hold on
plot([(1:1:10),(15:5:100)],KNN_eTrain_bin);
xlabel('0 < K < 100') % x-axis label
ylabel('Error') % y-axis label
legend('Testing Error','Training Error')

figure
plot([(1:1:10),(15:5:100)],KNN_Sensitivity_norm);
hold on
plot([(1:1:10),(15:5:100)],KNN_Specificity_norm);
xlabel('0 < K < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')

figure
plot([(1:1:10),(15:5:100)],KNN_Sensitivity_log);
hold on
plot([(1:1:10),(15:5:100)],KNN_Specificity_log);
xlabel('0 < K < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')

figure
plot([(1:1:10),(15:5:100)],KNN_Sensitivity_bin);
hold on
plot([(1:1:10),(15:5:100)],KNN_Specificity_bin);
xlabel('0 < K < 100') % x-axis label
ylabel('Percentage') % y-axis label
legend('Sensitivity','Specificty')

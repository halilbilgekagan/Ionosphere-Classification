
%% setup
rng(7);

clc;
clear;
Data = xlsread('IonosphereDataSet.xlsx');

[row colm] = size(Data);

train_rate=0.7;
shuffled = Data(randperm(size(Data, 1)), :);
ft_train = shuffled(1:(row.*train_rate), 1:(colm-1));
y_train = shuffled(1:(row.*train_rate), colm);
ft_test = shuffled((row.*train_rate+1):row, 1:(colm-1));
y_test = shuffled((row.*train_rate+1):row, colm);

%% logi
% f1 calc for each feature
for feature =1:(colm-1)
    mdlFTL=fitglm(ft_train(:,feature),y_train,"Distribution","binomial");
     pred_ytest= predict(mdlFTL, ft_test(:,feature));
     predLabels_test1=pred_ytest > 0.5;
     
    
    TP = sum((predLabels_test1 == 1) & (y_test == 1));
    TN = sum((predLabels_test1 == 0) & (y_test == 0));
    FP = sum((predLabels_test1 == 1) & (y_test == 0));
    FN = sum((predLabels_test1 == 0) & (y_test == 1));
    
    Accuracy_fl = (TP + TN) / (TP + TN + FP + FN);
    Precision_fl = TP / (TP + FP);
    Recall_fl = TP / (TP + FN) ;
    F1_Score_fl(feature,1) = 2 * ((Precision_fl * Recall_fl) / (Precision_fl + Recall_fl));

end

cvmdl = cvpartition(y_train, 'KFold', 10);
errors = zeros(cvmdl.NumTestSets, 1);

for i = 1:cvmdl.NumTestSets
    trainIdx = cvmdl.training(i);
    testIdx = cvmdl.test(i);
    mdlCV = fitglm(ft_train(trainIdx, :), y_train(trainIdx), 'Distribution', 'binomial');
    predY = predict(mdlCV, ft_train(testIdx, :));
    predLabels = predY > 0.5;
    errors(i) = mean(predLabels ~= y_train(testIdx));
end

mdl = fitglm(ft_train, y_train, 'Distribution', 'binomial');

meanError = mean(errors);
Data0=Data(1:120,1:32);
Data1=Data(121:340,1:32);
predProb_0 = predict(mdl, Data0);
predProb_1= predict(mdl,Data1);

linear_comb_Data0 = [ones(size(Data0, 1), 1) Data0] * mdl.Coefficients.Estimate;
linear_comb_Data1 = [ones(size(Data1, 1), 1) Data1] * mdl.Coefficients.Estimate;

% Visualize the decision boundary
figure;
scatter(linear_comb_Data0, predProb_0, 'b', 'DisplayName', 'Data0');
hold on;
scatter(linear_comb_Data1, predProb_1, 'r', 'DisplayName', 'Data1');
xlabel('Linear Combination of Features');
ylabel('Predicted Probability');
title('Logistic Regression Decision Boundary');
legend;
grid on;
%set threshold 0.3 from plot
predY_test1 = predict(mdl, ft_test);
predLabels_test = predY_test1 > 0.3;

testError = mean(predLabels_test ~= y_test);

TP = sum((predLabels_test == 1) & (y_test == 1));
TN = sum((predLabels_test == 0) & (y_test == 0));
FP = sum((predLabels_test == 1) & (y_test == 0));
FN = sum((predLabels_test == 0) & (y_test == 1));

Accuracy = (TP + TN) / (TP + TN + FP + FN);
Precision = TP / (TP + FP);
Recall = TP / (TP + FN) ;
F1_Score = 2 * ((Precision * Recall) / (Precision + Recall));

%lda model training

mdlLDA = fitcdiscr(ft_train, y_train);
pred_lda=predict(mdlLDA,ft_test);

confMat = confusionmat(y_test, pred_lda);
TPlda = confMat(1,1);
FPlda = confMat(1,2);
FNlda = confMat(2,1);
precisionlda = TPlda / (TPlda + FPlda);
recallda = TPlda / (TPlda + FNlda);
f1_scorelda = 2 * (precisionlda * recallda) / (precisionlda + recallda);

%pca model training

[coeff, score_train, ~, ~, explained] = pca(ft_train);
explained_variance = cumsum(explained);
num_components = find(explained_variance >= 95, 1);  
ft_train_pca = score_train(:, 1:num_components);
score_test = ft_test * coeff;
ft_test_pca = score_test(:, 1:num_components);
mdlpca = fitlm(ft_train_pca, y_train);
pred_testpca = predict(mdlpca, ft_test_pca);
thresholdpca = 0.5;  
pred_test_labelspca = double(pred_testpca > thresholdpca);  
% F1 score calculation
confMatpca = confusionmat(double(y_test), pred_test_labelspca);  
TPpca = confMatpca(2,2);
FPpca = confMatpca(1,2);
FNpca = confMatpca(2,1);
precisionpca = TPpca / (TPpca + FPpca);
recallpca = TPpca / (TPpca + FNpca);
f1_scorepca = 2 * (precisionpca * recallpca) / (precisionpca + recallpca);


%% knn classifier


maxK = 20;  % set maximum k value
errors = zeros(maxK, 1);

for k=1:maxK
    mdlKNN = fitcknn(ft_train, y_train, 'NumNeighbors', k);
    cvmdlKNN = crossval(mdlKNN, 'KFold', 10);
    knnLoss = kfoldLoss(cvmdlKNN);
    errors(k) = knnLoss;
end

%visualize for optimal k value
figure;
plot(1:maxK, errors, '-o');
title('Elbow Method for Optimal k');
xlabel('Number of Neighbors k');
ylabel('Cross-validated Error');
grid on;

[~, optimalK] = min(errors);
%train model based on optimalK
mdlKNN = fitcknn(ft_train, y_train, 'NumNeighbors', optimalK);
predY_test2 = predict(mdlKNN, ft_test);

TPK = sum((predY_test2 == 1) & (y_test == 1));
TNK = sum((predY_test2 == 0) & (y_test == 0));
FPK = sum((predY_test2 == 1) & (y_test == 0));
FNK = sum((predY_test2 == 0) & (y_test == 1));

AccuracyK = (TPK + TNK) / (TPK + TNK + FPK + FNK);
PrecisionK = TPK / (TPK + FPK);
RecallK = TPK / (TPK + FNK);
F1_ScoreK = 2 * ((PrecisionK * RecallK) / (PrecisionK + RecallK));


% f1 calc for each feature
for feature =1:(colm-1)
    mdlFTK=fitcknn(ft_train(:,feature),y_train,'NumNeighbors', optimalK);
    predY_test1 = predict(mdlFTK, ft_test(:,feature));
    
    TP = sum((predY_test1 == 1) & (y_test == 1));
    TN = sum((predY_test1 == 0) & (y_test == 0));
    FP = sum((predY_test1 == 1) & (y_test == 0));
    FN = sum((predY_test1 == 0) & (y_test == 1));
    
    Accuracy_fk = (TP + TN) / (TP + TN + FP + FN);
    Precision_fk = TP / (TP + FP);
    Recall_fk = TP / (TP + FN) ;
    F1_Score_fk(feature,1) = 2 * ((Precision_fk * Recall_fk) / (Precision_fk + Recall_fk));

end

%% Bayes for each feature
  
% f1 calc for each feature
for feature =1:(colm-1)
    mdlFTB=fitcnb(ft_train(:,feature),y_train);
    predY_test1 = predict(mdlFTB, ft_test(:,feature));
    
    TP = sum((predY_test1 == 1) & (y_test == 1));
    TN = sum((predY_test1 == 0) & (y_test == 0));
    FP = sum((predY_test1 == 1) & (y_test == 0));
    FN = sum((predY_test1 == 0) & (y_test == 1));
    

    Accuracy_fb = (TP + TN) / (TP + TN + FP + FN);
    Precision_fb = TP / (TP + FP);
    Recall_fb = TP / (TP + FN) ;
    F1_Score_fb(feature,1) = 2 * ((Precision_fb * Recall_fb) / (Precision_fb + Recall_fb));

end

%% setup for bayes
rng(7);
Data = xlsread('IonosphereDataSet.xlsx');
[row colm] = size(Data);
train_rate=1;
shuffled = Data(randperm(size(Data, 1)), :);
ft_train = shuffled(1:(row.*train_rate), 1:(colm-1));
y_train = shuffled(1:(row.*train_rate), colm);
ft_test = shuffled((row.*train_rate+1):row, 1:(colm-1));
y_test = shuffled((row.*train_rate+1):row, colm);

%% bayes

numDataPoints = size(ft_train, 1);
predY_test3 = zeros(numDataPoints, 1);

for loo = 1:numDataPoints
    tempData = ft_train;
    Val = tempData(loo, :);
    tempLabels = y_train;
    ValLabel = tempLabels(loo);
    tempData(loo, :) = [];
    tempLabels(loo) = [];
    mdlbys = fitcnb(tempData, tempLabels);
    % Prediction
    predY_test3(loo) = predict(mdlbys, Val);
end

TPBb = sum((predY_test3 == 1) & (y_train == 1));
TNBb = sum((predY_test3 == 0) & (y_train == 0));
FPBb = sum((predY_test3 == 1) & (y_train == 0));
FNBb = sum((predY_test3 == 0) & (y_train == 1));

AccuracyBb = (TPBb + TNBb) / (TPBb + TNBb + FPBb + FNBb);
PrecisionBb = TPBb / (TPBb + FPBb);
RecallBb = TPBb / (TPBb + FNBb);
F1_ScoreBb = 2 * ((PrecisionBb * RecallBb) / (PrecisionBb + RecallBb));
%% ALL classification Comparisation
% Plot F1 Scores for Each Classifier
figure;
x = 1:(colm - 1);

% Plot F1 Scores for Logistic Regression
plot(x, F1_Score_fl, "-o", 'DisplayName', 'Logistic Regression');
hold on;

% Plot F1 Scores for KNN
plot(x, F1_Score_fk, "-x", 'DisplayName', 'KNN');

% Plot F1 Scores for Naive Bayes
plot(x, F1_Score_fb, "-s", 'DisplayName', 'Naive Bayes');

title('Comparison of F1 Scores for Each Feature Across Different Classifiers');
xlabel('Features');
ylabel('F1 Scores');
legend('Location', 'best');
grid on;
hold off;

[sorted_fl,I]=sort(F1_Score_fl,'descend');
[sorted_fk,m]=sort(F1_Score_fk,'descend');
[sorted_fb,n]=sort(F1_Score_fb,'descend');
%Showing F1 scores for corresponding feature
figure;

subplot(3,1,1);
plot(sorted_fl, '-o');
xticks(1:32);
xticklabels(I);
title('F1 Scores For Each Feature (Logistic Regression)');
xlabel('Features');
ylabel('F1 Scores');
grid on;

subplot(3,1,2);
plot(sorted_fk, '-s');
xticks(1:32);
xticklabels(m);
title('F1 Scores For Each Feature (KNN)');
xlabel('Features');
ylabel('F1 Scores');
grid on;

subplot(3,1,3);
plot(sorted_fb, '-x');
xticks(1:32);
xticklabels(n);
title('F1 Scores For Each Feature (Bayes)');
xlabel('Features');
ylabel('F1 Scores');
grid on;

% Overall title for the figure
sgtitle('F1 Scores For Each Feature');

% Average F1 Score Calculation
figure;
sumF1 = F1_Score_fb + F1_Score_fk + F1_Score_fl;
avgF1 = sumF1 / 3;
plot(x, avgF1, "-o", 'DisplayName', 'AVG F1');
title('Average F1 Scores for Each Feature');
xlabel('Features');
ylabel('F1 Scores');
legend('Location', 'best');
grid on;

% Overall F1 Scores for Different Models
F1_Score = 2 * ((Precision * Recall) / (Precision + Recall));
F1_ScoreK = 2 * ((PrecisionK * RecallK) / (PrecisionK + RecallK));
F1_ScoreBb = 2 * ((PrecisionBb * RecallBb) / (PrecisionBb + RecallBb));

models = {'Logistic Regression', 'KNN', 'Logistic Regression with PCA','Naive Bayes','Logistic Regression with LDA'};
f1_scores = [F1_Score, F1_ScoreK,f1_scorepca, F1_ScoreBb,f1_scorelda]; 

% Create a bar plot for the F1 scores
figure;
bar(f1_scores);

% Set the x-axis labels
set(gca, 'XTickLabel', models);

% Add title and labels
title('Overall F1 Scores for Different Models');
xlabel('Models');
ylabel('F1 Score');

% Display the values on top of the bars
for i = 1:length(f1_scores)
    text(i, f1_scores(i), num2str(f1_scores(i), '%.3f'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

grid on;

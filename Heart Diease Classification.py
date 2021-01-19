import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import tree
from urllib.request import urlopen # Get data from UCI Machine Learning Repository
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

Cleveland_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
Longbeach_data_URL ='https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
np.set_printoptions(threshold=np.nan) #see a whole array when we output it

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(urlopen(Hungarian_data_URL), names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
LongbeachHeartDisease = pd.read_csv(urlopen(Longbeach_data_URL), names = names) #gets Longbeach data
datatemp = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease,LongbeachHeartDisease] #combines all arrays into a list

heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()
##pre_processing
#in the report, heart disease ran from 1-4. Therefore, we make the classification problem easier by dividing into 2 diagnosis: 0 and 1
#We are going to convert the predictor column into 1 for " heart disease is present" and 0 for " heart disease is not present"
#change ? to nan value

heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes

#nomalization
for item in heartDisease: #converts everything to floats
    heartDisease[item] = pd.to_numeric(heartDisease[item],downcast='float')
def normalize(heartDisease, toNormalize): #normalizes
    result = heartDisease.copy()
    for item in heartDisease.columns:
        if (item in toNormalize):
            max_value = heartDisease[item].max()
            min_value = heartDisease[item].min()
            result[item] = (heartDisease[item] - min_value) / (max_value - min_value)
    return result
toNormalize = ['age','cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal'] #columns to normalize
heartDisease = normalize(heartDisease, toNormalize)
heartDisease = heartDisease.dropna()
heartDisease.head()

# 1 for "heart disease is present" and 0 for "heart disease is not present."
for i in range(1,5):
    heartDisease['heartdisease'] = heartDisease['heartdisease'].replace(i,1)

train, test = train_test_split(heartDisease, test_size = 0.20, random_state = 42)


training_set = train.ix[:, train.columns != 'heartdisease']
# Next we create the class set
class_set = train.ix[:, train.columns == 'heartdisease']

# Next we create the test set doing the same process as the training set
test_set = test.ix[:, test.columns != 'heartdisease']
test_class_set = test.ix[:, test.columns == 'heartdisease']

##feature selection
## Using gini importance to choose important feature in feature selection
#print("**THIS IS RANDOM FOREST METHOD-FEATURE SELECTION METHOD**")
#fitRF = RandomForestClassifier(random_state = 42,
#                                criterion='gini',
#                                n_estimators = 500,
#                                )
# Print the name and gini importance of each feature
#fitRF.fit(training_set, class_set['heartdisease'])
#for feature in zip(names, fitRF.feature_importances_):
#    print(feature)
## identify and select most important feature

# features that have an importance of more than 0.069
#sfm= SelectFromModel(fitRF,threshold=0.069)
#sfm.fit(training_set, class_set['heartdisease']) # Train the selector
#SelectFromModel(estimator= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=500, n_jobs=1, oob_score=False, random_state=42,
#            verbose=0, warm_start=False), prefit= False,threshold=0.069)
# Print the names of the most important features
#for feature_list_index in sfm.get_support(indices=True):
#    print(names[feature_list_index])

### Create a data subset with only the most important feature
# Transform the data to create a new dataset containing only the most important features
#training_set_important = sfm.transform(training_set)
#test_set_important = sfm.transform(test_set)
## PCA method
print("**THIS IS PCA METHOD-FEATURE EXTRACTION**")
#feature component
# choose the 6 principal component
pca = PCA()
X_train = pca.fit_transform(training_set)
X_test = pca.transform(test_set)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
pca1 = PCA(n_components=7)
principalComponents1 = pca1.fit_transform(training_set)
principalComponents2 = pca1.transform(test_set)
print("This is important principal components")
print(pca1.explained_variance_ratio_)
principalDf = pd.DataFrame(data = principalComponents1
             , columns = ['principal component 1', 'principal component 2','principal component 3'
                          ,'principal component 4', 'principal component 5', 'principal component 6'
                          ,'principal component 7'])

#decision tree method
print("****THIS PART IS DECISION TREE CLASSIFICATION****")
data = tree.DecisionTreeClassifier()
data = data.fit(train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']], train['heartdisease'])
predictions_data = data.predict(test[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])
predictright = 0
predictions_data.shape[0]
for i in range(0,predictions_data.shape[0]-1):
    if (predictions_data[i]== test.iloc[i][13]):
        predictright +=1
accuracy = predictright/predictions_data.shape[0]
print("accuracy of decision tree method with full feature classification")
print(accuracy* 100 ,'%')
print("Table comparing actual vs. predicted values for my test set:\n",
     pd.crosstab(predictions_data, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
#ROC curve calculation
fpr2, tpr2, _ = roc_curve(predictions_data, test_class_set)

#AUC curve calcuation
auc_dt = auc(fpr2, tpr2)
##decision tree method with limited feature classifier by using feature selection
#data1 = tree.DecisionTreeClassifier()
#data1 = data.fit(training_set_important, class_set['heartdisease'])
#predictions_data1 = data1.predict(test_set_important)
#predictright1 = 0
#predictions_data1.shape[0]
#for i in range(0,predictions_data1.shape[0]-1):
#    if (predictions_data1[i]== test.iloc[i][13]):
#        predictright1 +=1
#accuracy1 = predictright1/predictions_data1.shape[0]
#print("accuracy of decision tree method  with limited feature classifier by using feature selection ")
#print(accuracy1* 100 ,'%')
#print("Table comparing actual vs. predicted values for my test set:\n",
#     pd.crosstab(predictions_data1, test_class_set['heartdisease'],
#                  rownames=['Predicted Values'],
#                  colnames=['Actual Values']))
#fpr8, tpr8, _ = roc_curve(predictions_data1, test_class_set)
#AUC curve calcuation
#auc_rf = auc(fpr8, tpr8)
##decision tree method with limited feature classifier by using feature extraction
classifierDCT = tree.DecisionTreeClassifier()
classifierDCT.fit(principalComponents1, class_set['heartdisease'])
# Predicting the Test set results
y_pred = classifierDCT.predict(principalComponents2)
print("accuracy of decision tree method with limited feature classifier by using feature extraction:\n",
     '%.3f' % (accuracy_score(test_class_set, y_pred)* 100), '%')
print(pd.crosstab(y_pred, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
fpr12, tpr12, _ = roc_curve(y_pred, test_class_set)
#AUC curve calcuation
auc_rf = auc(fpr12, tpr12)
#SVM method
print("****THIS PART IS SVM CLASSIFICATION ****")
svmtest = svm.SVC()
svmfit = svmtest.fit(train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']], train['heartdisease'])
svmPredictions = svmtest.predict(test[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])
predictrightsvm = 0
for i in range(0,svmPredictions.shape[0]-1):
    if (svmPredictions[i]== test.iloc[i][13]):
        predictrightsvm +=1
accuracysvm = predictrightsvm/svmPredictions.shape[0]
print("accuracy of SVM method with full feature classification")
print(accuracysvm* 100 ,'%')
print(pd.crosstab(svmPredictions, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
#ROC curve calculation
fpr4, tpr4, _ = roc_curve(svmPredictions, test_class_set)
#AUC curve calcuation
auc_svm = auc(fpr4, tpr4)
## SVM with limited classifier method by using feature selection

#svmtest1 = svm.SVC()
#svmfit1 = svmtest1.fit(training_set_important, class_set['heartdisease'] )
#svmPredictions1 = svmtest1.predict(test_set_important)
#predictrightsvm1 = 0
#for i in range(0,svmPredictions1.shape[0]-1):
#    if (svmPredictions1[i]== test.iloc[i][13]):
#        predictrightsvm1 +=1
#accuracysvm1 = predictrightsvm1/svmPredictions1.shape[0]
#print("accuracy of SVM method with limited feature classification by using feature selection")
#print(accuracysvm1* 100 ,'%')
#print(pd.crosstab(svmPredictions1, test_class_set['heartdisease'],
#                  rownames=['Predicted Values'],
#                  colnames=['Actual Values']))
#fpr7, tpr7, _ = roc_curve(svmPredictions1, test_class_set)
#AUC curve calcuation
#auc_rf = auc(fpr7, tpr7)
## SVM with limited classifier method by using feature extraction
classifierSVM = svm.SVC()
classifierSVM.fit(principalComponents1, class_set['heartdisease'])
# Predicting the Test set results
SVM_pred = classifierDCT.predict(principalComponents2)
print("accuracy of SVM method with limited feature classifier by using feature extraction:\n",
     '%.3f' % (accuracy_score(test_class_set, SVM_pred)* 100), '%')
print(pd.crosstab(SVM_pred, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
fpr11, tpr11, _ = roc_curve(SVM_pred, test_class_set)
#AUC curve calcuation
auc_rf = auc(fpr11, tpr11)
# Kth nearest Neighbors
print(" THIS PART IS KTH NEAREST NEIGHBORS CLASSIFICATION")
heartDiseaseKnn = KNeighborsClassifier(n_neighbors=13)
heartDiseaseKnn.fit(training_set, class_set['heartdisease'])
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=13, p=2,
           weights='uniform')

predictions = heartDiseaseKnn.predict(test_set)

# Let's compare the predictions vs. the actual values
print(pd.crosstab(predictions, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
# Let's get the accuracy of our test set
accuracy = heartDiseaseKnn.score(test_set, test_class_set['heartdisease'])
print("accuracy of Kth nearest Neighbors method with full feature classification :")
print('%.3f' % (accuracy * 100), '%')
fpr3, tpr3, _ = roc_curve(predictions, test_class_set)
auc_knn = auc(fpr3, tpr3)
##KTH NEAREST NEIGHBORS METHOD with limited feature
#heartDiseaseKnn1 = KNeighborsClassifier(n_neighbors=10)
#heartDiseaseKnn1.fit(training_set_important, class_set['heartdisease'] )
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
#           weights='uniform')

#predictions1 = heartDiseaseKnn1.predict(test_set_important)

# Let's compare the predictions vs. the actual values
#print(pd.crosstab(predictions1, test_class_set['heartdisease'],
#                  rownames=['Predicted Values'],
#                  colnames=['Actual Values']))
# Let's get the accuracy of our test set
#accuracyKNN = heartDiseaseKnn1.score(test_set_important, test_class_set['heartdisease'])
#print("accuracy of Kth nearest Neighbors method with limited feature classification by using feature selection:")
#print('%.3f' % (accuracyKNN * 100), '%')
#fpr6, tpr6, _ = roc_curve(predictions1, test_class_set)
#AUC curve calcuation
#auc_rf = auc(fpr6, tpr6)
## KNN with limited classifier method by using feature extraction
classifierKNN = KNeighborsClassifier(n_neighbors=10)
classifierKNN.fit(principalComponents1, class_set['heartdisease'])
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
           weights='uniform')
# Predicting the Test set results
KNN_pred = classifierKNN.predict(principalComponents2)

print(pd.crosstab(KNN_pred, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
print("accuracy of KNN method with limited feature classifier by using feature extraction:\n",
     '%.3f' % (accuracy_score(test_class_set, KNN_pred)* 100), '%')
fpr10, tpr10, _ = roc_curve(KNN_pred, test_class_set)
#AUC curve calcuation
auc_rf = auc(fpr10, tpr10)
#Random forest method
print(" THIS PART IS RANDOM FOREST CLASSIFICATION")
fitRF_important = RandomForestClassifier(random_state=42,
                                criterion='gini',
                                n_estimators = 500,
                                )
#Train the new classifier on the new dataset containing the most important features
#fitRF_important.fit(training_set_important, class_set['heartdisease'])
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=500, n_jobs=1, oob_score=False, random_state=42,
#            verbose=0, warm_start=False)
#print("***Compare the accuracy of my full feature classifier to our limited feature classifier****")
#predictions_RF = fitRF.predict(test_set) # apply full featured classifier to the test data
#print(pd.crosstab(predictions_RF, test_class_set['heartdisease'],
#                  rownames=['Predicted Values'],
#                  colnames=['Actual Values']))

#print("Here is my accuracy of random forest method with full featured classifier to the test data:\n",
#     '%.3f' % (accuracy_score(test_class_set, predictions_RF)* 100), '%')
#predictions_RF1 = fitRF_important.predict(test_set_important) #apply limited feature classifier to our test data
#print(pd.crosstab(predictions_RF1, test_class_set['heartdisease'],
#                  rownames=['Predicted Values'],
#                  colnames=['Actual Values']))

#print("Here is my accuracy of random forest method with limited featured classifier by using feature selection to the test data:\n",
#     '%.3f' % (accuracy_score(test_class_set, predictions_RF1)* 100), '%')


#ROC curve calculation
#fpr1, tpr1, _ = roc_curve(predictions_RF, test_class_set)
##AUC curve calcuation
#auc_rf = auc(fpr1, tpr1)
#fpr5, tpr5, _ = roc_curve(predictions_RF1, test_class_set)
#AUC curve calcuation
#auc_rf = auc(fpr5, tpr5)
## RANDOM FOREST WITH LIMITED FEATURE BY USING FEATURE EXTRACTION
classifier = RandomForestClassifier(max_depth= None, random_state=42)
classifier.fit(principalComponents1, class_set['heartdisease'])
# Predicting the Test set results
y_pred1 = classifier.predict(principalComponents2)
print(pd.crosstab(y_pred, test_class_set['heartdisease'],
                  rownames=['Predicted Values'],
                  colnames=['Actual Values']))
print("Here is my accuracy of random forest method with limited feature classification by using feature extraction:\n",
     '%.3f' % (accuracy_score(test_class_set, y_pred1)* 100), '%')
fpr9, tpr9, _ = roc_curve(y_pred1, test_class_set)
#AUC curve calcuation
auc_rf = auc(fpr9, tpr9)

## THIS IS ROC CURVE with full feature
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(fpr2, tpr2,label='Decision Trees ROC Curve with full feature(area = %.4f)' % auc_dt,
         color = 'navy',
         linewidth=2)
#plt.plot(fpr1, tpr1,label='Random Forest ROC Curve with full feature (area = %.4f)' % auc_rf,
#         color = 'red',
#         linewidth=2)
plt.plot(fpr3, tpr3,label='Kth Nearest Neighbor ROC Curve with full feature (area = %.4f)' % auc_knn,
         color = 'green',
         linewidth=2)
plt.plot(fpr4, tpr4,label='SVM ROC Curve (area = %.4f)' % auc_svm,
         color = 'pink',
         linewidth=2)

ax.set_facecolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For All Models')
plt.legend(loc="lower right")

plt.show()
##ROC Curve with limited feature by using feature selection
#fig, ax = plt.subplots(figsize=(10, 10))
#plt.plot(fpr8, tpr8,label='Decision Trees ROC Curve with limited feature by using feature selection(area = %.4f)' % auc_dt,
#         color = 'navy',
#         linewidth=2)
#plt.plot(fpr5, tpr5,label='Random Forest ROC Curve with limited feature by using feature selection (area = %.4f)' % auc_rf,
#         color = 'red',
#         linewidth=2)
#plt.plot(fpr6, tpr6,label='Kth Nearest Neighbor ROC Curve with limited feature by using feature selection (area = %.4f)' % auc_knn,
#         color = 'green',
#         linewidth=2)
#plt.plot(fpr7, tpr7,label='SVM ROC Curve with limited feature by using feature selection (area = %.4f)' % auc_svm,
#         color = 'pink',
#         linewidth=2)

#ax.set_facecolor('#fafafa')
#plt.plot([0, 1], [0, 1], 'k--', lw=2)
#plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
#plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
#plt.xlim([-0.01, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve Comparison For All Models')
#plt.legend(loc="lower right")

#plt.show()
## ROC CURVE WITH limited feature by using feature extraction
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(fpr12, tpr12,label='Decision Trees ROC Curve with limited feature by using feature extraction(area = %.4f)' % auc_dt,
         color = 'navy',
         linewidth=2)
plt.plot(fpr9, tpr9,label='Random Forest ROC Curve with limited feature by using feature extraction (area = %.4f)' % auc_rf,
         color = 'red',
         linewidth=2)
plt.plot(fpr10, tpr10,label='Kth Nearest Neighbor ROC Curve with limited feature by using feature extraction(area = %.4f)' % auc_knn,
         color = 'green',
         linewidth=2)
plt.plot(fpr11, tpr11,label='SVM ROC Curve with limited feature by using feature extraction (area = %.4f)' % auc_svm,
         color = 'pink',
         linewidth=2)

ax.set_facecolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For All Models')
plt.legend(loc="lower right")

plt.show()

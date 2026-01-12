##################
#EEE419 Project 2
#Caitlyn Blythe
##################

#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from warnings import filterwarnings
filterwarnings('ignore')

#create dataframe and specify which data is relevant
df_sonar = pd.read_csv('sonar_all_data_2.csv',header=None)
df_sonar.drop(61,axis=1,inplace=True)
x = df_sonar.iloc[:,:60]    #uses data up to column 60
y = df_sonar.iloc[:,60]    #extracts the classification column (contains the "answer key" for our model)

#split the data and standardize it:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

y_pred = []    #initialize y_pred variable as a matrix, will store predictions for the # of components 1-60
accuracy = []    #initialize the accuracy variable as a matrix, will store accuracies for predictions in y_pred
comp_loop = np.arange(1,61)    #initialize variable comp_loop to loop over number of components

model = MLPClassifier(hidden_layer_sizes=(150),activation='logistic',max_iter=2000,alpha=0.00001,solver='adam',tol=0.0001)
for num_comp in comp_loop:    #loop over the number of components
    pca = PCA(n_components=num_comp)    #pca, looping through 60 values of components
    x_train_pca = pca.fit_transform(x_train_std)    #train on standardized x_train data
    x_test_pca = pca.transform(x_test_std)    #test on standardized x_test data
    model.fit(x_train_pca, y_train)    #use MLPClassifier on pca x training data
    y_pred.append(model.predict(x_test_pca))    #add the predictions into y_pred array

y_pred = np.array(y_pred)    #total array for y_pred
for index in range(len(comp_loop)):    #find accuracies for each prediction
    accuracy.append(accuracy_score(y_test, y_pred[index]))
    print('Number of components: ', comp_loop[index])
    print('Accuracy: %2f' % (accuracy_score(y_test, y_pred[index])))
acc_index = accuracy.index(max(accuracy))    #find the index for the maximum accuracy
print('Maximum accuracy is %.2f' % max(accuracy), 'using',comp_loop[acc_index],'components')
cmat = confusion_matrix(y_test, y_pred[acc_index])    #confusion matrix on test vs predicted y data
print(cmat)
#plot the accuracy vs num components:
plt.plot(comp_loop,accuracy)
plt.title('Accuracy vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.show()



import csv
import random
from time import time

from sklearn.mixture import GaussianMixture

# Set seed
random.seed(time())

subjects={}
with open('DSL-StrongPasswordData.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for row in reader:
        
        if row[0] != "subject":
            if subjects.get(row[0]) is None:
                data={}
                data['KD']=[]
                data['DDKL']=[]
                data['UUKL']=[]

                subjects[row[0]] = data
            
            subjects[row[0]]['KD'].append(row[3])
            subjects[row[0]]['DDKL'].append(row[4])
            subjects[row[0]]['UUKL'].append(row[3]+row[4])

train_data={}
test_data={}
# Create a list of 80 random different data for each subject
random_number=random.sample(range(0,400),80)
for subject in subjects:

    data_train = {}
    data_train['KD']=[]
    data_train['DDKL']=[]
    data_train['UUKL']=[]

    data_test = {}
    data_test['KD']=[]
    data_test['DDKL']=[]
    data_test['UUKL']=[]

    train_data[subject]=data_train
    test_data[subject]=data_test

    for i in range(0,400):
        
        if i in random_number:
            test_data[subject]['KD'].append(subjects[subject]['KD'][i])
            test_data[subject]['DDKL'].append(subjects[subject]['DDKL'][i])
            test_data[subject]['UUKL'].append(subjects[subject]['UUKL'][i])
        else:
            train_data[subject]['KD'].append(subjects[subject]['KD'][i])
            train_data[subject]['DDKL'].append(subjects[subject]['DDKL'][i])
            train_data[subject]['UUKL'].append(subjects[subject]['UUKL'][i])

GMM=[]
for subject in train_data:
    GMM.append(GaussianMixture(n_components=3, covariance_type='full').fit((subject[1]['KD'],subject[1]['DDKL'],subject[1]['UUKL'])))


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
            
            subjects[row[0]]['KD'].append((float(row[3]),float(row[6]),float(row[9]),float(row[12]),float(row[15]),float(row[18]),float(row[21]),float(row[24]),float(row[27]),float(row[30]),float(row[33])))
            subjects[row[0]]['DDKL'].append((float(row[4]),float(row[7]),float(row[10]),float(row[13]),float(row[16]),float(row[19]),float(row[22]),float(row[25]),float(row[28]),float(row[31]),))
            subjects[row[0]]['UUKL'].append((float(row[3])+float(row[4]),float(row[6])+float(row[7]),float(row[9])+float(row[10]),float(row[12])+float(row[13]),float(row[15])+float(row[16]),float(row[18])+float(row[19]),float(row[21])+float(row[22]),float(row[24])+float(row[25]),float(row[27])+float(row[28]),float(row[30])+float(row[31])))

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

GMM={}

for subject in train_data:
    GMM[subject] = {}
    print("Training subject "+subject)
    GMM[subject]['KD'] = GaussianMixture(n_components=2, covariance_type='full').fit(train_data[subject]['KD'])
    GMM[subject]['DDKL'] = GaussianMixture(n_components=2, covariance_type='full').fit(train_data[subject]['DDKL'])
    GMM[subject]['UUKL'] = GaussianMixture(n_components=2, covariance_type='full').fit(train_data[subject]['UUKL'])

#Make prediction on the test data
correct_prediction_KD=0
correct_prediction_DDKL=0
correct_prediction_UUKL=0

for subject in test_data:
    print("Testing subject "+subject)

    prediction={}
    prediction["KD"]={}
    prediction["DDKL"]={}
    prediction["UUKL"]={}

    for gaussin in GMM:
        prediction["KD"][gaussin] = GMM[gaussin]["KD"].score(test_data[subject]["KD"])
        prediction["DDKL"][gaussin] = GMM[gaussin]["DDKL"].score(test_data[subject]["DDKL"])
        prediction["UUKL"][gaussin] = GMM[gaussin]["UUKL"].score(test_data[subject]["UUKL"])

    max_gaussian_KD=max(prediction["KD"].items(), key=lambda x: x[1])
    max_gaussian_DDKL=max(prediction["DDKL"].items(), key=lambda x: x[1])
    max_gaussian_UUKL=max(prediction["UUKL"].items(), key=lambda x: x[1])

    if max_gaussian_KD[0] == subject:
        correct_prediction_KD+=1
    if max_gaussian_DDKL[0] == subject:
        correct_prediction_DDKL+=1
    if max_gaussian_UUKL[0] == subject:
        correct_prediction_UUKL+=1

print("KD: "+str(correct_prediction_KD/len(test_data)))
print("DDKL: "+str(correct_prediction_DDKL/len(test_data)))
print("UUKL: "+str(correct_prediction_UUKL/len(test_data)))
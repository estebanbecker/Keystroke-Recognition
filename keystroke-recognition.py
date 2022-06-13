import csv

data=[]
with open('DSL-StrongPasswordData.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for row in reader:
        

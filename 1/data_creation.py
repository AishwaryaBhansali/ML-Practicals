# Assignment 1	Create the dataset for Placement Cell Perform following Preprocessing of data:							
# 	(dataset should include students' name, 10th, 12th, FE to TE %, certifications completed, no of projects completed, internships etc. Dependent variable placed or not placed, predict your batch sample case whether she will be placed or not.							
# 	Normalization 							
# 	Manage missing values,							
# 	Standardization							
# 	Scaling							
# 	Handling duplicate values							
# 	Handling outliers							
# 	Build the model, create an UI, test samples from 2024-25 batch		

import pandas as pd
import random

names = ['Aishwarya', 'Rahul', 'Sneha', 'Vikram', 'Neha', 'Siddharth', 'Tina', 'Ravi', 'Pooja', 'Karan']

def generate_student_data(n=50):
    data = []
    for i in range(n):
        name = random.choice(names) + " " + chr(65 + random.randint(0, 25)) + "."
        tenth = round(random.uniform(60, 95), 2)
        twelfth = round(random.uniform(60, 95), 2)
        fe = round(random.uniform(55, 90), 2)
        se = round(random.uniform(55, 90), 2)
        te = round(random.uniform(55, 90), 2)
        certs = random.randint(0, 5)
        projects = random.randint(1, 4)
        internships = random.randint(0, 2)

        # Simple logic for placement (you can make it smarter):
        placed = 'Yes' if te >= 70 and projects >= 2 and internships >= 1 else 'No'

        data.append([name, tenth, twelfth, fe, se, te, certs, projects, internships, placed])
    
    df = pd.DataFrame(data, columns=[
        'Name', 'Tenth %', 'Twelfth %', 'FE %', 'SE %', 'TE %',
        'Certifications', 'Projects', 'Internships', 'Placed'
    ])
    return df

df = generate_student_data()
df.to_csv("placement_data.csv", index=False)
print(df.head())
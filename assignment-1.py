# Can & Nour

# k-NN Dropping All Missing Values

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("thyroidDF.csv")
df.head()

df.isnull().sum()/len(df)*100

# Correlation matrix

df.corr()

# Histogram

faulty = df[ df['age'] > 130 ].index
df.drop(faulty , inplace=True)
df.hist(bins=30, figsize=(15, 10))

# drop columns "patient_id","TBG_measured","TBG" and rows with NaN values

df = df.drop(["patient_id","TBG_measured","TBG"], axis=1)
print(len(df))
print(len(df.dropna()))
df = df.dropna()

# Encoder

from sklearn.preprocessing import LabelEncoder
cat_var = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych", "referral_source", "target", "sex", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured", "TSH_measured"]
for var in cat_var:
    df[var] = LabelEncoder().fit_transform(df[var])
df.head()

# Predict and print accuracy

y = df["target"]
x = df.drop(["target"], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=10)

model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score

pred = model.predict(x_test)
print('Test Accuracy     : {:.3f}'.format(accuracy_score(y_test, pred)))

# k-NN Imputing Missing Values with 0

df = pd.read_csv("thyroidDF.csv")
df = df.dropna(subset=['sex'])
df = df.fillna(0)
df = df.drop(["patient_id"], axis=1)

cat_var = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych", "referral_source", "target", "sex", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured", "TSH_measured", "TBG_measured"]
for var in cat_var:
    df[var] = LabelEncoder().fit_transform(df[var])
df.head()

y = df["target"]
x = df.drop(["target"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

model = KNeighborsClassifier(n_neighbors=10)

model.fit(x_train,y_train)

pred = model.predict(x_test)
print('Test Accuracy     : {:.3f}'.format(accuracy_score(y_test, pred)))

# k-NN Imputing Missing Value with Mean

df = pd.read_csv("thyroidDF.csv")
df = df.dropna(subset=['sex'])
df = df.drop(["patient_id"], axis=1)

df['TSH'].fillna(df['TSH'].mean(), inplace=True)
df['T3'].fillna(df['T3'].mean(), inplace=True)
df['TT4'].fillna(df['TT4'].mean(), inplace=True)
df['T4U'].fillna(df['T4U'].mean(), inplace=True)
df['FTI'].fillna(df['FTI'].mean(), inplace=True)
df['TBG'].fillna(df['TBG'].mean(), inplace=True)

cat_var = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych", "referral_source", "target", "sex", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured", "TSH_measured", "TBG_measured"]
for var in cat_var:
    df[var] = LabelEncoder().fit_transform(df[var])

y = df["target"]
x = df.drop(["target"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

model = KNeighborsClassifier(n_neighbors=10)

model.fit(x_train,y_train)

pred = model.predict(x_test)
print('Test Accuracy     : {:.3f}'.format(accuracy_score(y_test, pred)))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("thyroidDF.csv")
df = df.dropna(subset=['sex'])
df = df.drop(["patient_id"], axis=1)

df['TSH'].fillna(df['TSH'].mean(), inplace=True)
df['T3'].fillna(df['T3'].mean(), inplace=True)
df['TT4'].fillna(df['TT4'].mean(), inplace=True)
df['T4U'].fillna(df['T4U'].mean(), inplace=True)
df['FTI'].fillna(df['FTI'].mean(), inplace=True)
df['TBG'].fillna(df['TBG'].mean(), inplace=True)

cat_var = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych", "referral_source", "target", "sex", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured", "TSH_measured", "TBG_measured"]
for var in cat_var:
    df[var] = LabelEncoder().fit_transform(df[var])

y = df["target"]
x = df.drop(["target"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

clf = GaussianNB()
clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

pred = model.predict(x_test)
print('Test Accuracy     : {:.3f}'.format(accuracy_score(y_test, pred)))

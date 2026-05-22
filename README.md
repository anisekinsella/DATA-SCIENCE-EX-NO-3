## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="362" height="436" alt="image" src="https://github.com/user-attachments/assets/03555e69-2de3-4b2f-b695-1115c600b18f" />


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="182" height="277" alt="image" src="https://github.com/user-attachments/assets/e16b96a7-4142-4db0-b5b3-fcd2373bc5f9" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="410" height="438" alt="image" src="https://github.com/user-attachments/assets/ab14f248-fe55-4776-960c-e47c32b83d76" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="413" height="466" alt="image" src="https://github.com/user-attachments/assets/e31e22d1-ef5e-46fb-948c-9b00c65b2e21" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
<img width="922" height="138" alt="image" src="https://github.com/user-attachments/assets/6c4f8f93-dae9-49db-9253-210957a988f8" />

```
df2=pd.concat([df2,enc],axis=1)
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="420" height="448" alt="image" src="https://github.com/user-attachments/assets/eb4bc8d3-a600-4837-b010-7be2366f3f16" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="673" height="432" alt="image" src="https://github.com/user-attachments/assets/4324b1b2-0210-4860-bf2d-51f62230c94a" />

```
pip install --upgrade category_encoders
```
<img width="605" height="431" alt="image" src="https://github.com/user-attachments/assets/adfc1295-8119-4308-ad6a-39d157bd183a" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="996" height="440" alt="image" src="https://github.com/user-attachments/assets/c289dcd7-ed6d-4a73-9f9e-45ac41dc32af" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
<img width="871" height="450" alt="image" src="https://github.com/user-attachments/assets/677e9a99-b248-4757-9388-36eb6dee9f52" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="695" height="437" alt="image" src="https://github.com/user-attachments/assets/1ef69e1d-04b8-4e06-939a-cdb17833d9fc" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="916" height="557" alt="image" src="https://github.com/user-attachments/assets/5f74dd63-6b9b-47bb-b2e3-959df9fdbbf4" />

```
df.skew()
```
<img width="456" height="127" alt="image" src="https://github.com/user-attachments/assets/4e0e6ed3-51a0-4d58-b70b-0f9b7eb2cab6" />

```
np.log(df["Highly Positive Skew"])
```
<img width="700" height="312" alt="image" src="https://github.com/user-attachments/assets/8987a26e-2c85-431c-9b2f-0544f14d0482" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="730" height="311" alt="image" src="https://github.com/user-attachments/assets/43d621d6-3905-45a0-86e5-2bfaf925fab2" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="696" height="315" alt="image" src="https://github.com/user-attachments/assets/ebe6a8ac-eecb-4132-8107-8386070656b3" />

```
np.square(df["Highly Positive Skew"])
```
```
df["Highly Positve Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="922" height="547" alt="image" src="https://github.com/user-attachments/assets/44d401b8-23dc-4a4c-b033-8f73c106412b" />

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
<img width="556" height="186" alt="image" src="https://github.com/user-attachments/assets/ac9b4abc-2de2-4e7f-a95d-bcd4907961e2" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="917" height="292" alt="image" src="https://github.com/user-attachments/assets/17b2e793-eea4-4b70-be9b-a1b078ddfd9c" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="905" height="651" alt="image" src="https://github.com/user-attachments/assets/ddcdab46-8a5a-43ff-b417-4e8c0f8d9381" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="905" height="645" alt="image" src="https://github.com/user-attachments/assets/13d6b4ac-2b28-409e-af58-98e3e8ce9b6f" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="890" height="653" alt="image" src="https://github.com/user-attachments/assets/f96d703e-8a78-4e9c-a63f-ce904b1a49eb" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="878" height="646" alt="image" src="https://github.com/user-attachments/assets/99ad941e-1798-4c4f-90d4-f5e0522779fe" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="891" height="646" alt="image" src="https://github.com/user-attachments/assets/f1d4333b-53cd-47a5-b2a8-0cd08d79eb1f" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="892" height="653" alt="image" src="https://github.com/user-attachments/assets/cddfca3a-0bde-43f8-9f2d-273abd1a1a1b" />

# RESULT:
   Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       

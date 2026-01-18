"""
An internshipproject.
Linear regression model using kaggle dataset
Dataset used = "https://www.kaggle.com/code/manassehibrahim/house-prices-prediction-using-ml-techniques/input?select=train.csv"
"""



#Modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.metrics import r2_score , root_mean_squared_error
import random as rand

#def for formatting
def section(title):
    print("\n" + "="*80)
    print(title.upper())
    print("="*80 + "\n")

def sub(title):
    print("\n" + "-"*80)
    print(title)
    print()

#beginning
section("Model for House Price prediction from Kaggle DataSet")

#loading of data with eror handling
try:
    df=pd.read_csv(r"train.csv")
    print("Data loaded Successfully")
    print(f"Dimensions of the DataSet:{df.shape}")
    print(f"Total entries:{len(df)}")
except:
    print("Failed to load Dataset")
    print("Ensure the enterd file path is correct or not")
    exit()

#data description
section("Basic Data Analysis")

sub("DataSet info:")
print(df.info())

sub("First five rows:")
print(df.head())

sub("Basic Statistcal Data:")
print(df.describe())

#independent var features
print("\n" + "-" * 80)
print("Mean price:", df["TARGET(PRICE_IN_LACS)"].mean())
print("Median price:", df["TARGET(PRICE_IN_LACS)"].median())
print("Mode price:", df["TARGET(PRICE_IN_LACS)"].mode()[0])
print("Minimum price:", df["TARGET(PRICE_IN_LACS)"].min())
print("Maximum price:", df["TARGET(PRICE_IN_LACS)"].max())
print("Standard Deviation:", df["TARGET(PRICE_IN_LACS)"].std())

#dependent vra features
sub("Features of Deciding Factors:")
print(f"Posted By: {dict(df["POSTED_BY"].value_counts())}")
print(f"CONSTRUCTION Status: {dict(df["UNDER_CONSTRUCTION"].value_counts())}")
print(f"RERA: {dict(df["RERA"].value_counts())}")
print(f"BHK NO.: {dict(df["BHK_NO."].value_counts())}")
print(f"BHK OR RK: {dict(df["BHK_OR_RK"].value_counts())}")
print(f"READY TO MOVE: {dict(df["READY_TO_MOVE"].value_counts())}")
print(f"RESALE: {dict(df["RESALE"].value_counts())}")

sub("Handling of Null values:")
missing=df.isnull().sum()

if missing.sum() == 0:
    print("No Null values found.")
else:
    print("Misiing values:")
    print(missing[missing>0])
    df=df.fillna(df.median(numeric_only=True))#fixes numeric data
    df=df.dropna()#drops non numeric nan values

section('Processing of Data.')

#adding a new coloumn city in place of address as encoding of address wil look petty horrific.
df['CITY'] = df['ADDRESS'].str.split(',').str[-1].str.strip()
df=df.drop(columns="ADDRESS")

#copying data for data model
df_model = df.copy()

print("Checking for Invalid Data Entries...")
print()

initial_count=len(df_model)
df_model = df_model[df_model['SQUARE_FT'] > 0]#removes any data less than 0
df_model = df_model[df_model['TARGET(PRICE_IN_LACS)'] > 0]
final_count = len(df_model)
removed = initial_count - final_count
if removed > 0:
    print(f"Removed {removed} rows with invalid values")
else:
    print("No invalid values found")

sub("Encoding categorical variables...")

#using one hot encoder for more meaningfull encoding of categorical data
ohe=OneHotEncoder(handle_unknown="ignore",sparse_output=False).set_output(transform="pandas")
ohetransform=ohe.fit_transform(df_model[["POSTED_BY","BHK_OR_RK","CITY"]])
df_model=pd.concat([df_model,ohetransform],axis=1).drop(columns=["POSTED_BY","BHK_OR_RK","CITY"])

# Safe encoded feature counts (after OneHotEncoding)
print(f"Encoded POSTED_BY: {len(ohe.categories_[0])} unique values")
print(f"Encoded BHK_OR_RK: {len(ohe.categories_[1])} unique values")
print(f"Encoded CITY: {len(ohe.categories_[2])} unique values")


section("FEATURE CORRELATION ANALYSIS")

print("\nFeature Correlations with Price:")

numerical_features = [   # well this was imp as..bhul gaya ha well due to one hot encoding there were numerous city coloumns
    'BHK_NO.',
    'SQUARE_FT',
    'UNDER_CONSTRUCTION',
    'RERA',
    'READY_TO_MOVE',
    'RESALE',
    'LONGITUDE',
    'LATITUDE',
    'POSTED_BY_Builder',
    'POSTED_BY_Dealer',
    'POSTED_BY_Dealer',
    'BHK_OR_RK_BHK',
    'BHK_OR_RK_RK',
    'TARGET(PRICE_IN_LACS)']
corr=df_model[numerical_features].corr(numeric_only=True)["TARGET(PRICE_IN_LACS)"].sort_values(ascending=False)
#df_model.to_csv("output.csv", index=False) was checking my fucked up dataset
print(corr)

#Preparing data for traning
section('Prepare Data for Training')

#only for show features as due to one hot encoding the daatset is well not so readable
Features = [
    'BHK_NO.',
    'SQUARE_FT',
    'UNDER_CONSTRUCTION',
    'RERA',
    'READY_TO_MOVE',
    'RESALE',
    'LONGITUDE',
    'LATITUDE',
    'POSTED_BY_ENCODED',
    'BHK_OR_RK_ENCODED',
    'ADDRESS_ENCODED'
]

X = df_model.drop("TARGET(PRICE_IN_LACS)",axis=1)#input features

y = df_model["TARGET(PRICE_IN_LACS)"]#independent variable

print(f"\nFeatures Selected: {len(Features)}")
for n,feature in enumerate(Features):
    print(f"  {n}. {feature}")

#Splitting of Data for test and train 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\nSplitting of Data Completed...")
print(f"Training Set: {len(X_train)} samples ({((len(X_train)/len(X)))*100 :.1f}%)")
print(f"Testing Set: {len(X_test)} samples ({((len(X_test)/len(X)))*100 :.1f}%)")

#scaling of Data
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print("\nData scaled succefully using Standard Scaler")

#model training
section("Prepare Data for Training")
#Model Training
model=LinearRegression()
model.fit(X_train_scaled,y_train)

print("Model Training Sucessfull")
print(f"Intercept: ₹{model.intercept_:.2f} Lacs.")


section('Model Evaluation')

#evaluation of model using predict
y_pred=model.predict(X_test_scaled)

sub("Test Set Performance:")

score=r2_score(y_test,y_pred)
RMSE=root_mean_squared_error(y_test,y_pred)
print(f"R² Score: {score:.5f}")
print(f"Root Mean Squared Error: ₹{RMSE} Lacs.")

#Sample prediction comparisions
section("SAMPLE PREDICTIONS")

sample_size=5
sample_indices=[]

print(f"\nShowing {sample_size} random predictions from test set:\n")#priting before else it will be 0

while sample_size>0:
    index=rand.randint(0,len(y_test))
    if index in sample_indices:
         pass
    else:
        sample_indices.append(index)
        sample_size-=1

for i in sample_indices:
    actual=y_test.iloc[i]
    predicted=y_pred[i]
    error=abs(actual - predicted)
    error_percent = (error/actual)*100

    print(f"Sample number {i}:")
    print(f"  Actual Price:    ₹{actual:.2f} Lacs")
    print(f"  Predicted Price: ₹{predicted:.2f} Lacs")
    print(f"  Error:           ₹{error:.2f} Lacs ({error_percent:.2f}%)")
    print("-"*80)

#PLOTTING OF GRAPHS
section("Plotting of Graphs")

#graph one for accuracy 
difference=y_test - y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'c--', lw=1)  # Perfect prediction line
plt.xlabel("Actual Price (Lacs)", fontsize=12)
plt.ylabel("Predicted Price (Lacs)", fontsize=12)
plt.title("Actual vs Predicted Prices", fontsize=14)
plt.grid(True, alpha=0.3)

#saving the fig 1
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=300)
plt.show()
plt.close()
print("Actual vs Predicted plot saved as 'actual_vs_predicted.png' in results folder")



#graph 2 for effects of top 10 coefficients
coeff_df = pd.DataFrame({ "Feature": X.columns, "Coefficient": model.coef_ }).sort_values(by="Coefficient", key=abs, ascending=False)
top_n = 10
coeff_df_top = coeff_df.head(top_n)

plt.figure(figsize=(17,6))
plt.barh(coeff_df_top["Feature"], coeff_df_top["Coefficient"], color='steelblue', edgecolor='black')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title(f"Top {top_n} Features by Coefficient")
plt.grid(True,alpha=0.3)

#saving the fig 2
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.show()
plt.close() 
print("Feature importance plot saved as 'feature_importance.png' in results folder")

section("MODEL SUMMARY")
print(f"Model Type: Linear Regression")
print(f"Number of Features Used: {X_train.shape[1]}")
print(f"Test Set R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Test Set RMSE: {root_mean_squared_error(y_test, y_pred):.2f} Lacs")

print("\nTop 5 Features by Coefficient:")
for i, row in coeff_df.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Coefficient']:.2f}")

section("Model Training Complete")

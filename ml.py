import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy import stats
from scipy.stats import skew
from scipy.stats import pearsonr
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

data = pd.read_excel('Petiole.xlsx')

data.head()

d=data.drop(['Sample Name'],axis=1)
for item in d.columns:
  print(item,"  ",skew(data[item].astype('float')))

data.columns

"""# **Lab Potassium prediction**"""

data.columns

data_K=data[['Lab K (%)','XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
       'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)','XRF Cu (ppm)']]

"""Removing Outlier values"""

Q1 = data_K['Lab K (%)'].quantile(0.25)
Q3 = data_K['Lab K (%)'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
x1=Q1 - 1.5 * IQR
x2=Q3 + 1.5 * IQR
print(x1)
print(x2)
data_K=data_K[(data_K['Lab K (%)']>=x1) & (data_K['Lab K (%)']<=x2)]

print(data_K.shape)
print(data.shape)

"""Correlation of Lab K with other independent variables"""

d=data_K.drop(['Lab K (%)'],axis=1)

for item in d.columns:
  corr, _ = pearsonr(data_K['Lab K (%)'],d[item]) 
  print("corelation between Lab K and ",item," = ",corr*100,"%")

"""Taking only important variables"""

data_K=data_K[['XRF K (%)','XRF Ca (%)','XRF Mg (%)','Lab K (%)']]

"""Standardizing the variables"""

d=data_K.drop(['Lab K (%)'],axis=1)
cols=d.columns

mean_arr = []
stddev_arr = []

for item in cols:
  data_K[item]=(data_K[item]-data_K[item].mean())/statistics.stdev(data_K[item])
  mean_arr.append(data_K[item].mean())
  stddev_arr.append(statistics.stdev(data_K[item]))

"""Spliting Dataset into training and testing data"""

y=data_K['Lab K (%)']
X=data_K.drop(['Lab K (%)'],axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=100)

RF=XGBRegressor(n_estimators=500,learning_rate=0.01)
RF.fit(X_train,y_train)

stat = {"mean": mean_arr, "std_dev": stddev_arr}

import pickle
with open('petiole_potassium_model.pkl', 'wb') as f:
    pickle.dump(RF, f)
with open('petiole_potassium_stat.txt', 'wb') as f:
    pickle.dump(stat, f)


##########################################################################

# """# **Lab Calcium prediction**"""

# data_Ca=data[['Lab Ca (%)','XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)', 'XRF Cu (ppm)']]

# """Histogram plot of Lab Ca"""

# sns.set_theme(style="whitegrid")
# plt.hist(data_Ca['Lab Ca (%)'])
# plt.title("Distribution of Ca")
# plt.xlabel("Ca")

# print("Min value: ",data_Ca['Lab Ca (%)'].min())
# print("Max Value: ",data_Ca['Lab Ca (%)'].max())
# print("Mean Value: ",data_Ca['Lab Ca (%)'].mean())
# print("Median Value: ",data_Ca['Lab Ca (%)'].median())
# print("skewness value: ",skew(data_Ca['Lab Ca (%)']))
# print("log skewness Value: ",skew(np.log(data_Ca['Lab Ca (%)'])))

# """Visualizing using Boxplot and removing outliers using IQR"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_Ca['Lab Ca (%)'])
# ax.set_title("Boxplot of Lab Ca")
# ax.set_xlabel('Lab Ca')



# Q1 = data_Ca['Lab Ca (%)'].quantile(0.25)
# Q3 = data_Ca['Lab Ca (%)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)

# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)

# data_Ca=data_Ca[(data_Ca['Lab Ca (%)']>=x1) & (data_Ca['Lab Ca (%)']<=x2)]

# data_Ca.shape

# data.shape

# """Correlation of all independent variables with the target variable"""

# d=data_Ca.drop(['Lab Ca (%)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_Ca['Lab Ca (%)'],d[item]) 
#   print("corelation between Lab Ca and ",item," = ",corr*100,"%")

# data_Ca.columns

# """HeatMap Plot to see multi-correlation"""

# plt.figure(figsize=(14,7))

# plt.title("Correlation of different variables")
# sns.heatmap(data=data_Ca.corr(), annot=True,cmap="YlGnBu")

# plt.xlabel("Heat-Map")



# # from statsmodels.stats.outliers_influence import variance_inflation_factor
# # from patsy import dmatrices

# # X=data_Ca.drop(['Lab Ca (%)'],axis=1)
# # vif_data = pd.DataFrame()
# # vif_data["feature"] = X.columns
# # vif_data["VIF"] = [variance_inflation_factor(X.values, i)
# #                           for i in range(len(X.columns))]

# # print(vif_data)

# """Removing variables with low correlation"""

# data_Ca=data_Ca.drop(['XRF Cu (ppm)','XRF Fe (ppm)','XRF Zn (ppm)','XRF S(%)','XRF K (%)'],axis=1)

# data_Ca.columns

# """Standardising the dataset"""

# d=data_Ca.drop(['Lab Ca (%)'],axis=1)
# cols=d.columns

# for item in cols:
#   data_Ca[item]=(data_Ca[item]-data_Ca[item].mean())/statistics.stdev(data_Ca[item])



# """Dividing into Train and Test dataset"""

# y=data_Ca['Lab Ca (%)']
# X=data_Ca.drop(['Lab Ca (%)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=0)

# """Modelling"""



# def optimise_pls_cv(X, y, n_comp):
#     # Define PLS object
#     pls = PLSRegression(n_components=n_comp)

#     # Cross-validation
#     y_cv = cross_val_predict(pls, X, y, cv=10)

#     # Calculate scores
#     r2 = r2_score(y, y_cv)
#     mse = mean_squared_error(y, y_cv)
#     rpd = y.std()/np.sqrt(mse)
    
#     return (y_cv, r2, mse, rpd)
    



# r2s = []
# mses = []
# rpds = []
# xticks = np.arange(1,8)
# for n_comp in xticks:
#     y_cv, r2, mse, rpd = optimise_pls_cv(X_train, y_train, n_comp)
#     r2s.append(r2)
#     mses.append(mse)
#     rpds.append(rpd)



# # Plot the mses
# def plot_metrics(vals, ylabel, objective):
#     with plt.style.context('ggplot'):
#         plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
#         if objective=='min':
#             idx = np.argmin(vals)
#         else:
#             idx = np.argmax(vals)
#         plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

#         plt.xlabel('Number of PLS components')
#         plt.xticks = xticks
#         plt.ylabel(ylabel)
#         plt.title('PLS')

#     plt.show()


# plot_metrics(mses, 'MSE', 'min')

# plot_metrics(rpds, 'RPD', 'max')

# """R2 Score and RMSE of PLSR model"""

# pls=PLSRegression(n_components=4)
# pls.fit(X_train,y_train)


# print("MODEL = PLSR")
# print("Train data :- ")
# pred_1=pls.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=pls.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("PLS_Ca_Train")
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("PLS_Ca_Test")
# plt.show()

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("Ca_PLS")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")











# data.columns



# """# **Lab Magnesium Prediction**"""

# data.columns
# data_mg=data[['Lab Mg (%)','XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)', 'XRF Cu (ppm)']]

# """Histogram of Lab Mg"""

# sns.set_theme(style="whitegrid")
# plt.hist(data_mg['Lab Mg (%)'])
# plt.title("Distribution of Mg")
# plt.xlabel("Mg")

# """Boxplot to visualize outliers"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_mg['Lab Mg (%)'])
# ax.set_title("Boxplot of Lab Mg")
# ax.set_xlabel('Lab Mg')

# """Removing Outliers"""

# Q1 = data_mg['Lab Mg (%)'].quantile(0.25)
# Q3 = data_mg['Lab Mg (%)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)
# data_mg=data_mg[(data_mg['Lab Mg (%)']>=x1) & (data_mg['Lab Mg (%)']<=x2)]

# print(data_mg.shape)
# print(data.shape)

# """Correlation of independent variables with target variable"""

# d=data_mg.drop(['Lab Mg (%)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_mg['Lab Mg (%)'],d[item]) 
#   print("corelation between Lab Mg and ",item," = ",corr*100,"%")

# """Heatmap plot"""

# plt.figure(figsize=(14,7))
# plt.title("Correlation of different variables")
# sns.heatmap(data=data_mg.corr(), annot=True,cmap="YlGnBu")

# plt.xlabel("Heat-Map")

# """Removing variables with low correlation value"""

# data_mg=data_mg.drop(['XRF Cu (ppm)','XRF Fe (ppm)','XRF Zn (ppm)','XRF K (%)','XRF Ca (%)'],axis=1)

# data_mg.columns

# """Standardization"""

# d=data_mg.drop(['Lab Mg (%)'],axis=1)
# cols=d.columns

# for item in cols:
#   data_mg[item]=(data_mg[item]-data_mg[item].mean())/statistics.stdev(data_mg[item])

# """Splitting dataset into traing and testing data"""

# y=data_mg['Lab Mg (%)']
# X=data_mg.drop(['Lab Mg (%)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=7)

# """Modeling & Result"""

# RF=lgb.LGBMRegressor(learning_rate=0.01, n_estimators=400)
# RF.fit(X_train,y_train)


# print("MODEL = LightGBM Regressor ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LightGBM_Mg_Train")
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LightGBM_Mg_Test")
# plt.show()

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("Mg_LightGBM")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")





# data.columns

# """# **Lab S Prediction**"""

# data_S=data[['Lab S (%)','XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)','XRF Cu (ppm)']]

# """Histogram"""

# plt.hist(data_S['Lab S (%)'])
# plt.title("Distribution of S")
# plt.xlabel("S")

# """Boxplot"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_S['Lab S (%)'])
# ax.set_title("Boxplot of Lab S")
# ax.set_xlabel('Lab S')

# """Removing Outliers"""

# Q1 = data_S['Lab S (%)'].quantile(0.25)
# Q3 = data_S['Lab S (%)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)
# data_S=data_S[(data_S['Lab S (%)']>x1) & (data_S['Lab S (%)']<x2)]

# print(data_S.shape)
# print(data.shape)

# """Correlation of Lab S with other independent variables"""

# d=data_S.drop(['Lab S (%)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_S['Lab S (%)'],d[item]) 
#   print("corelation between Lab S and ",item," = ",corr*100,"%")

# """Stnadardization"""

# data_S=data_S[['XRF K (%)','XRF Ca (%)','XRF Mg (%)','XRF S(%)','Lab S (%)','XRF Mn (ppm)']]
# d=data_S[['XRF K (%)','XRF Ca (%)','XRF Mg (%)','XRF S(%)','XRF Mn (ppm)']]
# cols=d.columns

# for item in cols:
#   data_S[item]=(data_S[item]-data_S[item].mean())/statistics.stdev(data_S[item])



# """Spliting Dataset into training and testing data"""

# y=data_S['Lab S (%)']
# X=data_S.drop(['Lab S (%)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=100)

# """Modeling"""

# RF=XGBRegressor(n_estimators=80,learning_rate=0.05)
# # (n_estimators=100,learning_rate=0.05)
# # lgb.LGBMRegressor(learning_rate=0.05,n_estimators=300)
# RF.fit(X_train,y_train)

# print("MODEL = XGBoost ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("XGBoost_S_Train")
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("XGBoost_S_Test")

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("S_XGBoost")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")



# """# **Lab Zn Prediction**"""

# data_Zn=data[['Lab Zn (ppm)', 'XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)','XRF Cu (ppm)']]

# """Histogram"""

# plt.hist(data_Zn['Lab Zn (ppm)'])
# plt.title("Distribution of Zn")
# plt.xlabel("Zn")

# """Boxplot"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_Zn['Lab Zn (ppm)'])
# ax.set_title("Boxplot of Lab Zn")
# ax.set_xlabel('Lab Zn')

# """Removing Outliers"""

# Q1 = data_Zn['Lab Zn (ppm)'].quantile(0.25)
# Q3 = data_Zn['Lab Zn (ppm)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)
# data_Zn=data_Zn[(data_Zn['Lab Zn (ppm)']>x1) & (data_Zn['Lab Zn (ppm)']<x2)]

# print(data_Zn.shape)
# print(data.shape)

# """Correlation of Lab Zn with other variables"""

# d=data_Zn.drop(['Lab Zn (ppm)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_Zn['Lab Zn (ppm)'],d[item]) 
#   print("corelation between Lab Zn and ",item," = ",corr*100,"%")

# """Removing low correlated variables"""

# data_Zn=data_Zn.drop(['XRF Cu (ppm)','XRF Mn (ppm)','XRF P (%)'],axis=1)

# data_Zn.columns

# """Standardization"""

# d=data_Zn.drop(['Lab Zn (ppm)'],axis=1)
# cols=d.columns

# for item in cols:
#   data_Zn[item]=(data_Zn[item]-data_Zn[item].mean())/statistics.stdev(data_Zn[item])

# """Spliting Dataset into training and testing data"""

# y=data_Zn['Lab Zn (ppm)']
# X=data_Zn.drop(['Lab Zn (ppm)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=100)

# """Modeling"""

# RF=SVR(kernel='rbf',C=10,gamma=0.01)
# # lgb.LGBMRegressor(learning_rate=0.01,n_estimators=200)
# RF.fit(X_train,y_train)

# print("MODEL = SVM ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("SVM_Zn_Train")
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("SVM_Zn_Test")

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("Zn_SVM")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")





# data.columns

# """# **Lab Mn Prediction**"""

# data_Mn=data[['Lab Mn (ppm)', 'XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)',
#        'XRF Cu (ppm)']]

# """Histogram"""

# plt.hist(data_Mn['Lab Mn (ppm)'])
# plt.title("Distribution of Mn")
# plt.xlabel("Mn")

# """Boxplot"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_Mn['Lab Mn (ppm)'])
# ax.set_title("Boxplot of Lab Mn")
# ax.set_xlabel('Lab Mn')

# """Removing Outliers"""

# Q1 = data_Mn['Lab Mn (ppm)'].quantile(0.25)
# Q3 = data_Mn['Lab Mn (ppm)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)
# data_Mn=data_Mn[(data_Mn['Lab Mn (ppm)']>x1) & (data_Mn['Lab Mn (ppm)']<x2)]

# print(data_Mn.shape)
# print(data.shape)

# """Correlation of Lab Mn with other independent variables"""

# d=data_Mn.drop(['Lab Mn (ppm)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_Mn['Lab Mn (ppm)'],d[item]) 
#   print("corelation between Lab Mn and ",item," = ",corr*100,"%")

# plt.figure(figsize=(14,7))
# plt.title("Correlation of different variables")
# sns.heatmap(data=data_Mn.corr(), annot=True,cmap="YlGnBu")

# plt.xlabel("Heat-Map")

# data_Mn=data_Mn[['XRF K (%)','XRF Ca (%)','XRF Mn (ppm)','XRF Fe (ppm)','Lab Mn (ppm)']]

# """Standardizing"""

# d=data_Mn.drop(['Lab Mn (ppm)'],axis=1)
# cols=d.columns

# for item in cols:
#   data_Mn[item]=(data_Mn[item]-data_Mn[item].mean())/statistics.stdev(data_Mn[item])

# """Spliting Dataset into training and testing data"""

# y=data_Mn['Lab Mn (ppm)']
# X=data_Mn.drop(['Lab Mn (ppm)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=100)

# """Modeling"""

# RF=LinearRegression()
# RF.fit(X_train,y_train)

# print("MODEL = LGBM ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LinearRegression_Mn_Train")

# plt.show()
# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LinearRegression_Mn_Test")

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("Mn_LinearRegression")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")





# data.columns



# """# **Lab Fe Prediction**"""

# data_Fe=data[['Lab Fe (ppm)','XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)',
#        'XRF Cu (ppm)']]

# """Histogram"""

# plt.hist(data_Fe['Lab Fe (ppm)'])
# plt.title("Distribution of Fe")
# plt.xlabel("Fe")

# """boxplot"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_Fe['Lab Fe (ppm)'])
# ax.set_title("Boxplot of Lab Fe")
# ax.set_xlabel('Lab Fe')

# """Removing outliers"""

# Q1 = data_Fe['Lab Fe (ppm)'].quantile(0.25)
# Q3 = data_Fe['Lab Fe (ppm)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)
# data_Fe=data_Fe[(data_Fe['Lab Fe (ppm)']>x1) & (data_Fe['Lab Fe (ppm)']<x2)]

# print(data_Fe.shape)
# print(data.shape)

# """Correlation of Lab Fe with other variables"""

# d=data_Fe.drop(['Lab Fe (ppm)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_Fe['Lab Fe (ppm)'],d[item]) 
#   print("corelation between Lab Fe and ",item," = ",corr*100,"%")

# """Taking only high correlated variables"""

# # data_Fe=data_Fe.drop(['XRF Cu (ppm)','XRF Zn (ppm)','XRF S(%)'],axis=1)
# data_Fe=data_Fe[['XRF Ca (%)','XRF Mg (%)','XRF Fe (ppm)','Lab Fe (ppm)']]

# data_Fe.columns

# """Standardizing the independent variables"""

# d=data_Fe.drop(['Lab Fe (ppm)'],axis=1)
# cols=d.columns

# for item in cols:
#   data_Fe[item]=(data_Fe[item]-data_Fe[item].mean())/statistics.stdev(data_Fe[item])

# """Spliting Dataset into training and testing data"""

# y=data_Fe['Lab Fe (ppm)']
# X=data_Fe.drop(['Lab Fe (ppm)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=100)

# """Modeling and Hyperparameter tuning"""

# params = {
#     'num_leaves': [3,5,7, 14, 21],
#     'learning_rate': [0.1, 0.03,0.05,0.01],
#     'max_depth': [-1, 3, 5],
#     'n_estimators': [20,30,50, 100, 200, 500],
# }

# grid = GridSearchCV(lgb.LGBMRegressor(random_state=100), params, scoring='r2', cv=5)
# grid.fit(X_train, y_train)

# grid.best_params_

# RF=lgb.LGBMRegressor(learning_rate=0.01,n_estimators=500)
# RF.fit(X_train,y_train)


# print("MODEL = LGBM ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LightGBM_Fe_Train")
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LightGBM_Fe_Test")
# plt.show()

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("Fe_LightGBM")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")









# """# **Lab Copper Predictions**"""

# data_Cu=data[['Lab Cu(ppm)','XRF P (%)', 'XRF K (%)', 'XRF Ca (%)', 'XRF Mg (%)',
#        'XRF S(%)', 'XRF Zn (ppm)', 'XRF Mn (ppm)', 'XRF Fe (ppm)','XRF Cu (ppm)']]

# """Histogram"""

# plt.hist(data_Cu['Lab Cu(ppm)'])
# plt.title("Distribution of Cu")
# plt.xlabel("Cu")

# """Boxplot"""

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x=data_Cu['Lab Cu(ppm)'])
# ax.set_title("Boxplot of Lab Cu")
# ax.set_xlabel('Lab Cu')

# """Removing Outliers"""

# Q1 = data_Cu['Lab Cu(ppm)'].quantile(0.25)
# Q3 = data_Cu['Lab Cu(ppm)'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# x1=Q1 - 1.5 * IQR
# x2=Q3 + 1.5 * IQR
# print(x1)
# print(x2)
# data_Cu=data_Cu[(data_Cu['Lab Cu(ppm)']>x1) & (data_Cu['Lab Cu(ppm)']<x2)]

# print(data_Cu.shape)
# print(data.shape)

# """Correlation of Lab Cu with other independent variables"""

# d=data_Cu.drop(['Lab Cu(ppm)'],axis=1)

# for item in d.columns:
#   corr, _ = pearsonr(data_Cu['Lab Cu(ppm)'],d[item]) 
#   print("corelation between Lab Cu and ",item," = ",corr*100,"%")

# """Removing low correlated variables"""

# data_Cu=data_Cu.drop(['XRF Zn (ppm)','XRF Mn (ppm)','XRF Fe (ppm)','XRF K (%)'],axis=1)

# """Standardization"""

# d=data_Cu.drop(['Lab Cu(ppm)'],axis=1)
# cols=d.columns

# for item in cols:
#   data_Cu[item]=(data_Cu[item]-data_Cu[item].mean())/statistics.stdev(data_Cu[item])

# """Spliting Dataset into training and testing data"""

# y=data_Cu['Lab Cu(ppm)']
# X=data_Cu.drop(['Lab Cu(ppm)'],axis=1)

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=100)

# """Modeling & Results"""

# RF=lgb.LGBMRegressor(learning_rate=0.1,n_estimators=20)
# RF.fit(X_train,y_train)
# # 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 20, 'num_leaves': 3}

# print("MODEL = LGBM ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# RF=SVR(kernel='rbf',C=1)
# RF.fit(X_train,y_train)
# # 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 20, 'num_leaves': 3}

# print("MODEL = SVM ")
# print("Train data :- ")
# pred_1=RF.predict(X_train)
# print("R2-Score : ",r2_score(y_train,pred_1))
# print("RMSE : ",(mean_squared_error(y_train,pred_1))**0.5)
# print("Test data :-")
# pred1=RF.predict(X_test)
# print("R2-Score : ",r2_score(y_test,pred1))
# print("RMSE : ",(mean_squared_error(y_test,pred1))**0.5)

# fig, ax = plt.subplots()
# ax.scatter(y_train, pred_1)
# ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LightGBM_Cu_Train")

# plt.show()
# fig, ax = plt.subplots()
# ax.scatter(y_test, pred1)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# ax.set_title("LightGBM_Cu_Test")

# plt.plot(y_train,pred_1,'o',color='black')
# plt.plot(y_test,pred1, 'o',color='red')
# plt.title("Cu_LightGBM")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.legend(["Train", "Test"], loc ="lower right")

# data_Cu.columns

# data.columns




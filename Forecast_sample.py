'''
This code contains initial analysis evaluating an ARIMAX 
forecasting model for Freight-spot-market prices.

For comparison with Boosted Tree method, this method 
improves RMSE by 50%, and makes a lot more intuitive sense

(Code computing the monthly forecast using the original approach 
appears in the commented section at the end of this script)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

#### 'A' contains demand information about freight
#### country-wide -- in particular, the monthly average  
#### of ratio of loads to available trucks 
A = [4.28, 3.51, 3.88,  6.28, 3.45, 3.03, 3.17, 2.1, 2.69, 2.5,
	1.85, 2.02, 1.57, 1.45, 1.77, 3.12, 2.09, 2.26, 2.4, 1.69, 
	2.26, 3.3, 2.23, 1.84, 2.89, 0.98, 1.91, 3.52, 4.4, 5.31]

df = pd.read_csv('Market_data.csv')
df = df.dropna()

df = df[df['Business Site'] == 'Spot']
df = df[df['Carrier Mode'] == 'Truckload']


#### Remove price outliers ####
df = df.query("50 < `Carrier Charge` < 5000")

#### Drop unused columns ####
df = df.drop(['Primary Reference','Actual Delivery','Carrier','Customer',
              'Carrier Linehaul','Carrier Fuel','Carrier Distance'],axis = 1)

#### convert date-time format #### 
df['Ship_Date'] = pd.to_datetime(df['Actual Ship'])
df = df.sort_values('Ship_Date').reset_index(drop = True)

#### Get rid of some outlier dates ####
df = df.iloc[3:len(df) - 3] 
df = df.reset_index(drop = True)
 
#### Break date into month and year ####
df['month'] = df['Ship_Date'].dt.month
df['year'] = df['Ship_Date'].dt.year

#### Convert Zip code into 5 digit continuous variable ####
df['ORIGIN_ZIP'] = df['ORIGIN_ZIP'].str.zfill(5)
df['DEST_ZIP'] = df['DEST_ZIP'].str.zfill(5)

#### Discard non-numeric zip codes ####
df['ORIGIN_ZIP'] = pd.to_numeric(df['ORIGIN_ZIP'],errors = 'coerce')
df['DEST_ZIP'] = pd.to_numeric(df['DEST_ZIP'],errors = 'coerce')
df = df.dropna(subset=['ORIGIN_ZIP'])
df = df.dropna(subset=['DEST_ZIP'])

#### get transaction average per month (over the three years) #### 
df = df.groupby(['year','month']).mean()


######## Save the monthly average data ##################
df.to_csv('spot_month_yr.csv')

######## This resets the indices so we can 
######## use them as inputs in the regression

df = df.reset_index(level=['month','year'])


#### Add the demand information to the dataframe ####
df['LT'] = A

#### Shift by one to use last month's demand as #### 
#### regressor for this month's cost            ####
df['LT_last'] = df['LT'].shift(1)

#### 'MA' one period lag -- use last month's average #### 
#### cost as other regressor for this month's cost   ####
df['MA'] = df['Carrier Charge'].shift(1)
# Replace the 'na' first value with the current value
df['MA'].iloc[0] = df['MA'].iloc[1]


#### Build regression/"ARIMAX" model ####
#### 'gblinear' gives standard linear regression, NOT boosted
xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', booster = 'gblinear')

y = df['Carrier Charge'].iloc[1:]
x = df.drop(['year','month','LT','Carrier Charge'],axis = 1).iloc[1:]

xg_reg.fit(x,y)

pred = xg_reg.predict(x)

#### Check metrics for the predictions compared to observations ####
rmse = np.sqrt(mean_squared_error(y,pred))
r2 = r2_score(y,pred)
print(f'RMSE: {rmse}')
print(f'r2 Score: {r2}')

#### Monthly index for the plot #### 
indices = ['04-18','05-18','06-18','07-18', '08-18', '09-18', '10-18', 
'11-18','12-18', '01-19','02-19', '03-19','04-19','05-19', '06-19','07-19', 
'08-19', '09-19', '10-19', '11-19', '12-19','01-20','02-20' ,'03-20','04-20',
'05-20', '06-20','07-20', '08-20']

#### Plot the results ####
plt.plot(indices, pred, label = 'regression')
plt.plot(y, label = 'observed')
plt.xticks(rotation = 'vertical')
plt.legend(loc = 'upper right')
plt.show()


'''
#### Previous Method (for comparison)
#### predicting Monthly average spot prices using 
#### original forecast approach: boosted regression
#### (to compare with the approach developed above)




df = pd.read_csv(''Market_data.csv')
df = df.dropna()

df = df[df['Business Site'] == 'Spot']
df = df[df['Carrier Mode'] == 'Truckload']

############################################
####### Clear outliers
############################################
df = df.query("50 < `Carrier Charge` < 5000")

# Drop off unused columns
df = df.drop(['Primary Reference','Actual Delivery','Carrier','Customer',
              'Carrier Linehaul','Carrier Fuel','Carrier Distance'],axis = 1)

df['Ship_Date'] = pd.to_datetime(df['Actual Ship'])

df['month'] = df['Ship_Date'].dt.month
df['year'] = df['Ship_Date'].dt.year

df['ORIGIN_ZIP'] = df['ORIGIN_ZIP'].str.zfill(5)
df['DEST_ZIP'] = df['DEST_ZIP'].str.zfill(5)

df['ORIGIN_ZIP'] = pd.to_numeric(df['ORIGIN_ZIP'],errors = 'coerce')
df['DEST_ZIP'] = pd.to_numeric(df['DEST_ZIP'],errors = 'coerce')
df = df.dropna(subset=['ORIGIN_ZIP'])
df = df.dropna(subset=['DEST_ZIP'])

filtered_df = df.drop(['Actual Ship','Carrier Mode','Ship_Date'], axis = 1)

merged_df = filtered_df.groupby(['ORIGIN_ZIP','DEST_ZIP','month','year']).mean()


###### Save and reload to reset indices as columns or do:
#df = df.reset_index(level=['ORIGIN_ZIP','DEST_ZIP','month','year'])

merged_df.to_csv("/monthly_comp.csv")

df = pd.read_csv("/monthly_comp.csv")

x = df.drop(['Carrier Charge'], axis=1)
y = df['Carrier Charge']

#### This is a boosted regression using random forests ####
xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror')

xg_reg.fit(x,y)


preds = xg_reg.predict(x)
rmse = np.sqrt(mean_squared_error(y,preds))
print("RMSE: %f" %(rmse))

'''
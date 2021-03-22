## 工具包导入&数据读取
### 工具包导入

'''
安装工具
# !pip install netCDF4 
'''
import pandas as pd
import numpy  as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.optimizers import Adam

import scipy 
from netCDF4 import Dataset
import netCDF4 as nc
import gc


# label_path       = './ENSO/data/SODA_label.nc'
# label_trans_path = './ENSO/data/'

# print('22')
# nc_label         = Dataset(label_path,'r')
 
# years            = np.array(nc_label['year'][:])
# months           = np.array(nc_label['month'][:])

# year_month_index = []
# vs               = []
# for i,year in enumerate(years):
#     for j,month in enumerate(months):
#         year_month_index.append('year_{}_month_{}'.format(year,month))
#         vs.append(np.array(nc_label['nino'][i,j]))

# df_SODA_label               = pd.DataFrame({'year_month':year_month_index}) 
# df_SODA_label['year_month'] = year_month_index
# df_SODA_label['label']      = vs

# df_SODA_label.to_csv(label_trans_path + 'df_SODA_label.csv',index = None)
# print('40')
# SODA_path        = './ENSO/data/SODA_train.nc'
# nc_SODA          = Dataset(SODA_path,'r')
# def trans_df(df, vals, lats, lons, years, months):
#     '''
#         (100, 36, 24, 72) -- year, month,lat,lon 
#     '''
#     for j,lat_ in enumerate(lats):
#         for i,lon_ in enumerate(lons):
#             c = 'lat_lon_{}_{}'.format(int(lat_),int(lon_))  
#             v = []
#             for y in range(len(years)):
#                 for m in range(len(months)): 
#                     v.append(vals[y,m,j,i])
#             df[c] = v
#     return df

# year_month_index = []

# years              = np.array(nc_SODA['year'][:])
# months             = np.array(nc_SODA['month'][:])
# lats             = np.array(nc_SODA['lat'][:])
# lons             = np.array(nc_SODA['lon'][:])


# for year in years:
#     for month in months:
#         year_month_index.append('year_{}_month_{}'.format(year,month))

# df_sst  = pd.DataFrame({'year_month':year_month_index}) 
# df_t300 = pd.DataFrame({'year_month':year_month_index}) 
# df_ua   = pd.DataFrame({'year_month':year_month_index}) 
# df_va   = pd.DataFrame({'year_month':year_month_index})

# print('74')
# df_sst = trans_df(df = df_sst, vals = np.array(nc_SODA['sst'][:]), lats = lats, lons = lons, years = years, months = months)
# df_t300 = trans_df(df = df_t300, vals = np.array(nc_SODA['t300'][:]), lats = lats, lons = lons, years = years, months = months)
# df_ua   = trans_df(df = df_ua, vals = np.array(nc_SODA['ua'][:]), lats = lats, lons = lons, years = years, months = months)
# df_va   = trans_df(df = df_va, vals = np.array(nc_SODA['va'][:]), lats = lats, lons = lons, years = years, months = months)

# label_trans_path = './ENSO/data/'
# df_sst.to_csv(label_trans_path  + 'df_sst_SODA.csv',index = None)
# df_t300.to_csv(label_trans_path + 'df_t300_SODA.csv',index = None)
# df_ua.to_csv(label_trans_path   + 'df_ua_SODA.csv',index = None)
# df_va.to_csv(label_trans_path   + 'df_va_SODA.csv',index = None)

# del df_sst
# gc.collect()
# del df_t300
# gc.collect()
# del df_ua
# gc.collect()
# del df_va
# gc.collect()

# label_path       = './ENSO/data/CMIP_label.nc'
# label_trans_path = './ENSO/data/'
# nc_label         = Dataset(label_path,'r')
 
# years            = np.array(nc_label['year'][:])
# months           = np.array(nc_label['month'][:])

# year_month_index = []
# vs               = []
# for i,year in enumerate(years):
#     for j,month in enumerate(months):
#         year_month_index.append('year_{}_month_{}'.format(year,month))
#         vs.append(np.array(nc_label['nino'][i,j]))

# df_CMIP_label               = pd.DataFrame({'year_month':year_month_index}) 
# df_CMIP_label['year_month'] = year_month_index
# df_CMIP_label['label']      = vs
# print('112')
# df_CMIP_label.to_csv(label_trans_path + 'df_CMIP_label.csv',index = None)
CMIP_path       = './ENSO/data/CMIP_train.nc'
CMIP_trans_path = './ENSO/data/'
nc_CMIP  = Dataset(CMIP_path,'r')
year_month_index = []

years              = np.array(nc_CMIP['year'][:])
months             = np.array(nc_CMIP['month'][:])
lats               = np.array(nc_CMIP['lat'][:])
lons               = np.array(nc_CMIP['lon'][:])

last_thre_years = 4645
for year in years:
   
    if year >= 4645 - last_thre_years:
        for month in months:
            year_month_index.append('year_{}_month_{}'.format(year,month))
print('130')
df_CMIP_sst  = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_t300 = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_ua   = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_va   = pd.DataFrame({'year_month':year_month_index})
print('136')
#因为内存限制,我们暂时取最后1000个year的数据
#改成4645试试
def trans_thre_df(df, vals, lats, lons, years, months, last_thre_years = 4645):
    '''
        (4645, 36, 24, 72) -- year, month,lat,lon 
    '''
    for j,lat_ in (enumerate(lats)):
#         print(j)
        for i,lon_ in enumerate(lons):
            c = 'lat_lon_{}_{}'.format(int(lat_),int(lon_))  
            v = []
            for y_,y in enumerate(years):
               
                if y >= 4645 - last_thre_years:
                    for m_,m in  enumerate(months): 
                        v.append(vals[y_,m_,j,i])
            df[c] = v
    return df


df_CMIP_sst  = trans_thre_df(df = df_CMIP_sst,  vals   = np.array(nc_CMIP['sst'][:]),  lats = lats, lons = lons, years = years, months = months)
df_CMIP_sst.to_csv(CMIP_trans_path + 'df_sst_CMIP.csv',index = None)
del df_CMIP_sst
gc.collect()
print('160')
df_CMIP_t300 = trans_thre_df(df = df_CMIP_t300, vals   = np.array(nc_CMIP['t300'][:]), lats = lats, lons = lons, years = years, months = months)
df_CMIP_t300.to_csv(CMIP_trans_path + 'df_t300_CMIP.csv',index = None)
del df_CMIP_t300
gc.collect()
print('166')
df_CMIP_ua   = trans_thre_df(df = df_CMIP_ua,   vals   = np.array(nc_CMIP['ua'][:]),   lats = lats, lons = lons, years = years, months = months)
df_CMIP_ua.to_csv(CMIP_trans_path + 'df_ua_CMIP.csv',index = None)
del df_CMIP_ua
gc.collect()
print('172')
df_CMIP_va   = trans_thre_df(df = df_CMIP_va,   vals   = np.array(nc_CMIP['va'][:]),   lats = lats, lons = lons, years = years, months = months)
df_CMIP_va.to_csv(CMIP_trans_path + 'df_va_CMIP.csv',index = None)
del df_CMIP_va
gc.collect()

# (36036, 1729)





import pandas as pd

# 合并
df = pd.read_csv(r'./data/product_info_simple_final_train.csv')
df_cif = pd.read_csv(r'./data/cbyieldcurve_info_final.csv')
df_tif = pd.read_csv(r'./data/time_info_final.csv')

mergedtable = pd.merge(df,df_cif,left_on='transaction_date',right_on='enddate',how='left')
mergedtable = pd.merge(mergedtable,df_tif,left_on='transaction_date',right_on='stat_date',how='left')
mergedtable=mergedtable.drop(['uv_fundown','uv_stableown','uv_fundopt','uv_fundmarket','uv_termmarket','during_days','total_net_value','enddate','stat_date'], axis=1)
mergedtable.to_csv(r'./data/product_info_simple_final_train_1.csv',index=None)


df1 = pd.read_csv(r'./data/predict_table.csv')
mergedtable1 = pd.merge(df1,df_cif,left_on='transaction_date',right_on='enddate',how='left')
mergedtable1 = pd.merge(mergedtable1,df_tif,left_on='transaction_date',right_on='stat_date',how='left')
mergedtable1=mergedtable1.drop(['enddate','stat_date'], axis=1)
mergedtable1.to_csv(r'./data/predict_input.csv',index=None)
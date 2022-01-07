import trend_discovery as td;
import pandas as pd;

target = {}
df = pd.read_csv('./dataset/itrend_dataset_sample_anon.csv')
df['time'] = pd.to_datetime(df['time'])
# Assuming df is a DataFrame, with:
# y_time is a timestamp column
# y_numeric consists of aggregations on numeric columns
# y_categoric are names of categorical columns

# AGGREGATION
target['mat'], target['boundaries'], target['constraints'], target['count_mat'] =\
    td.aggregate_as_time_series(df,
    freq='1W', y_time='time',
    y_numeric={'count': ('type', 'count')},
    y_categoric=['district', 'type'],
    trim_edges=True)

# CANDIDATE TREND DISCOVERY		
target['trends'] =\
	td.sliding_regression(target['mat'], target['boundaries'],
	target['constraints'], target['count_mat'],
	window_lengths=[26], slide_length=26,
	min_r2=0.25)
	
# COMPUTE DEVIATION FOR SEASONALITY
target['trends']['group_Half'] = target['trends']['start'].dt.month.astype('str').map({'1':1, '12':1, '6':2, '7':2})
target['trends']['deviation_season'] =\
    td.compute_deviation(target['trends'], 7, dimensions=['start', 'end'], eta_q=100, a1=0.57, a2=0.43, h=0.075)
del target['trends']['group_Half']

# COMPUTE DEVIATION FOR DISTRICT
target['trends']['deviation_district'] =\
    td.compute_deviation(target['trends'], 18, dimensions=['district'], eta_q=100, a1=0.57, a2=0.43, h=0.075)
		
print(target['trends'].sort_values('deviation_district', ascending=False)[['start', 'end', 'group_district', 'slope_norm', 'deviation_district']])
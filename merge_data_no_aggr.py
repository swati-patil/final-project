import pandas as pd

df1 = pd.read_csv('sorted_by_category.csv', low_memory = False)

df2 = pd.read_csv('review_data_no_text.csv', low_memory = False)

merged = df1.join(df2.set_index('asin'), on='asin')

print(merged.head())

merged = merged.drop(columns='Unnamed: 0', index=1)

merged.to_csv('merged_no_agg.csv')
print(merged.head())



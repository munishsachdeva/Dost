#Assignment 1
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Load the dataset
df = pd.read_csv('fo03FEB2023bhav.csv')

# a. Calculate Total PCR
total_calls = df[df['OPTION_TYP'] == 'CE']['CONTRACTS'].sum()
total_puts = df[df['OPTION_TYP'] == 'PE']['CONTRACTS'].sum()
total_pcr = total_puts / total_calls



# Filter data for the specified strikes
strikes = [17850, 17800, 17900, 18000, 17700]
filtered_data = df[df['STRIKE_PR'].isin(strikes)]

# b. Create a scatter plot for PUT CALL TOTAL OI vs CHANGE in NIFTY
fig = px.scatter(filtered_data, x='CHG_IN_OI', y='OPEN_INT', color='INSTRUMENT', hover_data=['SYMBOL'])
fig.update_layout(title='PUT CALL TOTAL OI vs CHANGE in NIFTY', xaxis_title='CHANGE in NIFTY', yaxis_title='PUT CALL TOTAL OI')



# Filter relevant columns for analysis
relevant_columns = ['SYMBOL', 'CLOSE']
df_relevant = df[relevant_columns]

# Select any 50 stocks from the 'SYMBOL' column
selected_stocks = df_relevant['SYMBOL'].unique()[:50]

# Filter the data for the selected stocks
filtered_data = df_relevant[df_relevant['SYMBOL'].isin(selected_stocks)]

# Pivot the filtered data
pivot_table = pd.pivot_table(filtered_data, values='CLOSE', index='SYMBOL')

# Calculate the percentage change in price
percentage_change = pivot_table.pct_change() * 100

# Transpose the DataFrame
percentage_change = percentage_change.T

# c. Plot the heatmap using plotly.express
fig_heatmap = px.imshow(percentage_change,
                labels=dict(x="Stocks", y="Dates", color="% Change"),
                x=percentage_change.columns,
                y=percentage_change.index)

# Update the layout of the heatmap
fig_heatmap.update_layout(title='Price Change Heatmap (in %) for 50 Stocks')



# d. Calculate Advance Decline Ratio
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df['date'] = df['TIMESTAMP'].dt.date
df_previous = df[df['date'] == df['date'].shift()].reset_index(drop=True)
df_current = df[df['date'] != df['date'].shift()].reset_index(drop=True)
# Merge the DataFrames based on symbol and date
merged_df = pd.merge(df_previous[['SYMBOL', 'CLOSE']], df_current[['SYMBOL', 'CLOSE']], on='SYMBOL', suffixes=('_prev', '_curr'))

# Calculate the advance decline ratio
advance_decline_ratio = len(merged_df[merged_df['CLOSE_curr'] > merged_df['CLOSE_prev']]) / len(merged_df[merged_df['CLOSE_curr'] < merged_df['CLOSE_prev']])

# e. Calculate top 5 stocks with highest positive and negative OI changes
top_positive_oi_change = df.groupby('SYMBOL')['CHG_IN_OI'].sum().nlargest(5)
top_negative_oi_change = df.groupby('SYMBOL')['CHG_IN_OI'].sum().nsmallest(5)

# Print the results
print('Total PCR:', total_pcr)
print("")
print('Advance Decline Ratio:', advance_decline_ratio)
print("")
print('Top 5 stocks with highest positive OI change:')
print(top_positive_oi_change)
print("")
print('Top 5 stocks with highest negative OI change:')
print(top_negative_oi_change)

# Display the dashboard
fig.show()
fig_heatmap.show()

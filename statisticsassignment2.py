#importing libraries pandas,numpy,maltplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#reading data and stored in variable df
df=pd.read_csv("C:/Users/sariy/Downloads/worldbankdata.csv")
df
#deleting the rows having NaN values
rows_to_delete = [630,631,632,633,634]
df = df.drop(rows_to_delete) 
df
#print the unique values from the series name column
print(df['Series Name'].unique())

#Filtering the data where series name == access elctricity
condition = df['Series Name'] == 'Access to electricity, rural (% of rural population)'
filtered_df = df[condition]
filtered_df
#getting all the column names of the dataframe
df.columns
#converting actual dataframe into pivot table
df_melted = pd.melt(filtered_df, id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'], 
                    var_name='Year', value_name='Value')

# Displaying the melted DataFrame
df_melted

#removing the rows which have unwanted data like (..)
df_melted = df_melted[df_melted['Value'] != '..']
df_melted
#removing the columns series code and country code 
df_melted= df_melted.drop(['Series Code', 'Country Code'], axis=1)
df_melted
df_melted['Value'] = df_melted['Value'].astype(float)
#grouping the data based on country name and year
grouped_df = df_melted.groupby(['Country Name', 'Year'])['Value'].mean().reset_index()

# reset the index after grouping
grouped_df = grouped_df.reset_index(drop=True)

grouped_df

#sort the data based on vakue column
grouped_df = grouped_df.sort_values(by='Value', ascending=False)

grouped_df

#creating line plot for access to electricity for different countries

def plot_lineplot(grouped_df):
    """
    Plot a line plot for the grouped dataframe.

    Parameters:
    grouped_df: dataframe with columns 'Year', 'Value', 'Country Name'
    """
    # Convert 'Year' column to numeric data
    grouped_df['Year'] = pd.to_numeric(grouped_df['Year'])
    grouped_df = grouped_df.sort_values(by='Year')

    # Set the figure size
    plt.figure(figsize=(8, 4))
    
    # Plot the line plot
    sns.lineplot(data=grouped_df, x='Year', y='Value', hue='Country Name', linewidth=2, linestyle='--')

    # Set plot title and labels of plot
    plt.title('Access to electricity, rural (% of rural population)')
    plt.xlabel('Year')
    plt.ylabel('Value')

    # Adjust legend position
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.show()

plot_lineplot(grouped_df)

#creating bargraph for access to electricity for different countries
def plot_bar_chart(grouped_df):
    """
    Plot a bar chart for grouped dataframe.

    Parameters:
    grouped_df: grouped dataframe with columns 'Country Name', 'Year', 'Value'
    """
    # Set the figure size
    plt.figure(figsize=(20, 8))
    
    # Code for plotting bar chart
    sns.barplot(data=grouped_df, x='Country Name', y='Value', hue='Year', edgecolor='black')
    plt.xticks(rotation=45)

    # Set plot title and labels
    plt.title('Access to electricity, rural (% of rural population)')
    plt.xlabel('Country Name')
    plt.ylabel('Value')

    # Adjust legend position
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Year')

    # Display the plot
    plt.show()

plot_bar_chart(grouped_df)

# Filter data for the years 2000 to 2021
filtered_df2 = grouped_df[(grouped_df['Year'] >= 2000) & (grouped_df['Year'] <= 2021)]

# Group by 'Country Name' and calculate the mean of 'Value'
country_mean_access = filtered_df2.groupby('Country Name')['Value'].mean().reset_index()

# Print or use country_mean_access as needed
print(country_mean_access)

grouped_df['Year'] = pd.to_numeric(grouped_df['Year'])
access_2000 = grouped_df[grouped_df['Year'] == 2000]
# Group by 'Country Name' and calculate the mean for the 'Value' column
average_mean_2000 = access_2000.groupby('Country Name')['Value'].mean().reset_index()

# Print or use average_mean_2000 as needed
print(average_mean_2000)
average_mean_2000_df = pd.DataFrame(average_mean_2000, columns=['Country Name', 'Value'])
average_mean_2000_df

# Filter data for the year 2000
access_2005 = grouped_df[grouped_df['Year'] == 2005]

# Group by 'Country Name' and calculate the mean for the 'Value' column
average_mean_2005 = access_2005.groupby('Country Name')['Value'].mean().reset_index()

# Print or use average_mean_2000 as needed
print(average_mean_2005)
average_mean_2005_df = pd.DataFrame(average_mean_2005, columns=['Country Name', 'Value'])
average_mean_2005_df

# Filter data for the year 2000
access_2021 = grouped_df[grouped_df['Year'] == 2021]

# Group by 'Country Name' and calculate the mean for the 'Value' column
average_mean_2021 = access_2021.groupby('Country Name')['Value'].mean().reset_index()

# Print average_mean_2021 
print(average_mean_2021)
average_mean_2021_df = pd.DataFrame(average_mean_2021, columns=['Country Name', 'Value'])
average_mean_2021_df

#merging the three dataframes
merged_df2 = pd.merge(average_mean_2000_df, average_mean_2005_df, on='Country Name', how='inner')
result_df = pd.merge(merged_df2, average_mean_2021_df, on='Country Name', how='inner')
result_df

#changing column names
result_df = result_df.rename(columns={'Value_x': '2000', 'Value_y': '2005','Value': '2021'})

result_df

#filtering the data based on condition
condition = df['Series Name'] == 'Arable land (% of land area)'
filtered_df = df[condition]
filtered_df

#converting dataframe into pivot table
df_melted2 = pd.melt(filtered_df, id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'], 
                    var_name='Year', value_name='Value')

# Display the melted DataFrame
df_melted2

#drop the columns 
df_melted1 = df_melted2.drop(['Series Code', 'Country Code'], axis=1)
df_melted1

#sorting the data based on value column
grouped_df1 = df_melted1.sort_values(by='Value', ascending=True)

grouped_df1
#removing unwanted data like '..'
pattern_to_remove = '..'
grouped_df1['Value'] = grouped_df1['Value'].str.replace(pattern_to_remove, '')

df5 = grouped_df1.dropna(subset=['Value'])

# reset the index after dropping rows
df5 = grouped_df1.reset_index(drop=True)

df5 = df5.loc[31:].copy()
df5 = df5.reset_index(drop=True)

df5 = df5.sort_values(by='Value', ascending=False)

df5

def plot_lineplot(df):
    """
    Plot a line plot for the specified DataFrame.

    Parameters:
    df: Pandas DataFrame with columns 'Year', 'Value', 'Country Name'
    """
    # Convert 'Value' column to float
    df['Value'] = df['Value'].astype(float)

    # Round the 'Value' column to two decimal places
    df['Value'] = df['Value'].round(2)

    # Set the figure size
    plt.figure(figsize=(8, 4))
    
    # Code for plotting line plot
    sns.lineplot(data=df, x='Year', y='Value', hue='Country Name', linewidth=2, linestyle='--')

    # Set plot title and labels
    plt.title('Arable land (% of land area) Each Country Over the period of 2000 to 2021')
    plt.xlabel('Year')
    plt.ylabel('Value')

    # Adjust legend position
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.show()

#function call with the dataframe
plot_lineplot(df5)


def plot_bar_chart(df):
    """
    Plot a bar chart for the specified DataFrame.

    Parameters:
    - df: Pandas DataFrame with columns 'Country Name', 'Year', 'Value'
    """
    # Set the figure size
    plt.figure(figsize=(20, 8))

    # Code for plotting bar chart
    sns.barplot(data=df, x='Country Name', y='Value', hue='Year', edgecolor='black')
    plt.xticks(rotation=45)

    # Set plot title and labels
    plt.title('Arable land (% of land area) Each Country Over the period of 2000 to 2021')
    plt.xlabel('Country Name')
    plt.ylabel('Value')

    # Adjust legend position
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Year')

    # Show the plot
    plt.show()
#calling the function
plot_bar_chart(df5)

#cleaning the dataframe 
final_df_cleaned = df[df.ne('..').all(axis=1)]

#filter the data based on the condition
condition = final_df_cleaned['Series Name'].isin([
    'GDP growth (annual %)',
    'Urban population (% of total population)',
    'Population growth (annual %)',
    'Population, total',
    'GDP per capita',
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'Urban population growth (annual %)'
])

# Filtering the DataFrame
filtered_df = final_df_cleaned[condition]

# Convert relevant columns to numeric
numeric_columns = ['2000','2001', '2002', '2003', '2004', '2005', '2006', '2013', '2014', '2015',
       '2016', '2017', '2018', '2019', '2020', '2021', '2022']  # Replace with your actual column names
filtered_df[numeric_columns] = filtered_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values after conversion
filtered_df = filtered_df.dropna(subset=numeric_columns)

# Grouping by 'Country Name' and 'Series Name'
grouped_df = filtered_df.groupby(['Country Name', 'Series Name'])

# Specify the columns for which you want to calculate the sum
columns_of_interest = ['2000','2001', '2002', '2003', '2004', '2005', '2006', '2013', '2014', '2015',
       '2016', '2017', '2018', '2019', '2020', '2021', '2022'] 

# Calculate the sum for each group and selected columns
sum_values = grouped_df[columns_of_interest].mean()

# Round the values to two decimal places 
sum_values_rounded = sum_values.round(2)

sum_values_formatted = sum_values_rounded.applymap('{:.2f}'.format)

# Display the results
print("Sum values (rounded to two decimal places):")
print(sum_values_formatted)


#filtering the data based on country
target_country = 'Belgium'

# Check if the target country exists in the index
if target_country in sum_values_formatted.index.get_level_values('Country Name'):
    # Filter rows for the target country using .loc
    country_df = sum_values_formatted.loc[target_country]

country_df


selected_columns = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2013']

# Melt the DataFrame
melted_df = pd.melt(country_df.reset_index(), id_vars=['Series Name'], value_vars=selected_columns, var_name='Year', value_name='Value')

# Display the reshaped DataFrame
melted_df


selected_columns = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2013']

# Melt the DataFrame
melted_df = pd.melt(country_df.reset_index(), id_vars=['Series Name'], value_vars=selected_columns, var_name='Year', value_name='Value')

# Pivot the DataFrame
pivoted_df = melted_df.pivot(index='Year', columns='Series Name', values='Value')

# Display the reshaped DataFrame
pivoted_df

#creating correlation heatmap for "Belgium"
def plot_correlation_heatmap(dataframe, title):
    """
    Plot a correlation matrix heatmap for the specified DataFrame.

    Parameters:
    - dataframe: Pandas DataFrame
    - title: Title for the heatmap
    """
    # Calculate the correlation matrix
    correlation_matrix = pivoted_df.corr()

    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    # Set title for the plot
    plt.title(title)

    # Display the plot
    plt.show()

#function call
plot_correlation_heatmap(pivoted_df, "Belgium")



target_country = 'United States'

# Checking the target country exists in the index
if target_country in sum_values_formatted.index.get_level_values('Country Name'):
    # Filter rows for the target country using .loc
    country_df_usa = sum_values_formatted.loc[target_country]

country_df_usa

selected_columns_usa = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2013']

# Melt the DataFrame
melted_df_usa = pd.melt(country_df.reset_index(), id_vars=['Series Name'], value_vars=selected_columns, var_name='Year', value_name='Value')

# Display the reshaped DataFrame
melted_df_usa

selected_columns_usa = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2013']

# Melt the DataFrame
melted_df_usa = pd.melt(country_df_usa.reset_index(), id_vars=['Series Name'], value_vars=selected_columns_usa, var_name='Year', value_name='Value')

# Pivot the DataFrame
pivoted_df_usa = melted_df_usa.pivot(index='Year', columns='Series Name', values='Value')

# Display the reshaped DataFrame
pivoted_df_usa

#creating correlation heamap for coutry "USA"
def plot_correlation_heatmap_usa(pivoted_df_usa, title="USA"):
    """
    Plot a correlation matrix heatmap for data related to the United States.

    Parameters:
    - pivoted_df_usa: Pandas DataFrame with columns for different Series Names and years as index
    - title: Title for the heatmap (default is "USA")
    """
    # Calculate the correlation matrix
    correlation_matrix_usa = pivoted_df_usa.corr()

    # Set the figure size
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(correlation_matrix_usa, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)

    # Set title for the plot
    plt.title(title)

    # Display the plot
    plt.show()

#function call with the desired arguments
plot_correlation_heatmap_usa(pivoted_df_usa, "USA")


target_country_india = 'India'

# Checking the target country exists in the index
if target_country in sum_values_formatted.index.get_level_values('Country Name'):
    # Filter rows for the target country using .loc
    country_df_india = sum_values_formatted.loc[target_country_india]

country_df_india


selected_columns_india = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2013']

# Melt the DataFrame
melted_df_india = pd.melt(country_df_india.reset_index(), id_vars=['Series Name'], value_vars=selected_columns, var_name='Year', value_name='Value')

# Display the reshaped DataFrame
melted_df_india


selected_columns_india = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2013']

# Melt the DataFrame
melted_df_india = pd.melt(country_df_india.reset_index(), id_vars=['Series Name'], value_vars=selected_columns_usa, var_name='Year', value_name='Value')

# Pivot the DataFrame
pivoted_df_india = melted_df_india.pivot(index='Year', columns='Series Name', values='Value')

# Display the reshaped DataFrame
pivoted_df_india



#creating correlation heat map for country "india"
def plot_correlation_heatmap_india(pivoted_df_india, title="India"):
    """
    Plot a correlation matrix heatmap for data related to India.

    Parameters:
    pivoted_df_india: Pandas DataFrame with columns for different Series Names and years as index
    title: Title for the heatmap (default is "India")
    """
    # Calculate the correlation matrix
    correlation_matrix_india = pivoted_df_india.corr()

    # Set the figure size
    plt.figure(figsize=(8, 6))

    # Creating heatmap
    sns.heatmap(correlation_matrix_india, annot=True, cmap='inferno', fmt=".2f", linewidths=.5)

    # Set title for the plot
    plt.title(title)

    # Display the plot
    plt.show()

#calling the function with desired arguments
plot_correlation_heatmap_india(pivoted_df_india, "India")














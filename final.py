import pandas as pd
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def ecommerce_regression(path, degree=2):
    """
    Polynomial regression on e-commerce data
    Parameters: path, degree
    Return: Polynomial Regression chart
    """
    data = pd.read_csv(path)
    
    # Convert date
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['YEAR'] = data['DATE'].dt.year
    
    # Sales per year
    sales_year = data.groupby('YEAR', as_index=False)['ECOMSA'].mean()
    
    # Actual variables
    x = sales_year[['YEAR']].values
    y = sales_year['ECOMSA'].values
    
    # Poly varibales
    poly = PolynomialFeatures(degree=degree)
    x_reg = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_reg, y)
    
    # Forecast
    x_forecast = np.arange(sales_year['YEAR'].min(), sales_year['YEAR'].max() + 1).reshape(-1, 1)
    y_forecast = model.predict(poly.transform(x_forecast))
    plt.figure(figsize=(10, 6))
    plt.scatter(sales_year['YEAR'], sales_year['ECOMSA'], color='blue', label='Actual Sales', alpha=0.7, edgecolor='black')
    plt.plot(x_forecast.flatten(), y_forecast, color='red', label='Predicted Sales (Polynomial)', linewidth=2)
    plt.title('E-Commerce Sales Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('E-Commerce Sales')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def trade_sales_regression(file_path, degree=2):
    """
    Polynomial regression on retail trade sales data
    Parameters: file_path, degree
    Returns: Polynomial Regression chart
    """
    data = pd.read_csv(file_path)
    
    # convert date
    data = pd.read_csv(file_path, parse_dates=['DATE'])
    data = data.loc[data['DATE'] >= '2000-01-01']
    data['MONTH_NUMERIC'] = (data['DATE'] - data['DATE'].min()).dt.days
    
    # Actual variables
    x = data['MONTH_NUMERIC'].values.reshape(-1, 1)
    y = data['MRTSSM44000USS'].values
    
    # Poly variables
    poly = PolynomialFeatures(degree=degree)
    x_reg = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_reg, y)
    predict = model.predict(x_reg)
    plt.figure(figsize=(10, 6))
    plt.plot(data['DATE'], y, label='Actual', color='blue', alpha=0.7)
    plt.plot(data['DATE'], predict, label='Predicted', color='red', linestyle='--')
    plt.title('Monthly Trade Sales Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('Retail Trade Sales')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def monkeypox(cases_file, retail_file):
    """
    Show correlation between monkey pox cases and retail sales data with
    linear regression
    Parameters: cases_file, retail_file
    Return: Correlation graph
    """
    data = pd.read_csv(cases_file)
    
    # convert time data
    data['Month'] = pd.to_datetime(data['Epi_date_V3']).dt.to_period('M')
    
    # total cases in month
    cases = data.groupby('Month', as_index=False)['Cases'].sum()
    
    # read file convert to date time
    retail = pd.read_csv(retail_file, parse_dates=['observation_date'])
    retail['Month'] = retail['observation_date'].dt.to_period('M')

    
    # Sum sales in needed columns
    columns = ['RSBMGESD', 'RSCCAS', 'RSDBS', 'RSEAS', 'RSFHFS', 'RSFSDP', 
                     'RSFSXMV', 'RSGASS', 'RSGCS', 'RSMVPD', 'RSNSR', 'RSXFS']
    retail['Total_Retail_Sales'] = retail[columns].sum(axis=1)
    
    # total retail by month
    monthly_sales = retail.groupby('Month')['Total_Retail_Sales'].sum().reset_index()
    
    # merge mpox and retail data
    cases_to_sales = pd.merge(cases, monthly_sales, on='Month')
    cases_to_sales['Cases_to_Sales_Ratio'] = cases_to_sales.eval('Cases / Total_Retail_Sales')

    
    # Convert data
    cases_to_sales['Month_numeric'] = cases_to_sales['Month'].dt.to_timestamp().astype('int64') / 1e9

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cases_to_sales, x='Month_numeric', y='Cases_to_Sales_Ratio', color='blue', s=100)
    sns.regplot(data=cases_to_sales, x='Month_numeric', y='Cases_to_Sales_Ratio', scatter=False, color='red', line_kws={'linewidth': 2}, ci=None)

    plt.title("Monkeypox Cases to Sales Ratio")
    plt.xlabel("Months")
    plt.ylabel("Ratio")
    plt.xticks(ticks=cases_to_sales['Month_numeric'], 
               labels=cases_to_sales['Month'].dt.strftime('%Y-%m'), 
               rotation=45)
    plt.grid()
    plt.show()

def covid(cases_file, retail_file):
    """
    Show correlation between overal covid cases and retail sales data with
    linear regression as well as zoomed in graph of greatest cases to sales ratio spike
    Parameters: cases_file, retail_file
    Return: Correlation graph
    """
    cases = pd.read_csv(cases_file)
    
    # Covert to time data
    cases['Month'] = pd.to_datetime(cases['Day']).dt.to_period('M')

    # Group in months
    monthly_cases = cases.groupby('Month')['Daily new confirmed cases of COVID-19 per million people'].sum().reset_index()
    
    # Process retail sales data
    retail = pd.read_csv(retail_file, parse_dates=['observation_date'])
    retail['Month'] = retail['observation_date'].dt.to_period('M')
    monthly_retail = retail.groupby('Month')[['RSBMGESD', 'RSCCAS', 'RSDBS', 'RSEAS', 'RSFHFS', 
                                          'RSFSDP', 'RSFSXMV', 'RSGASS', 'RSGCS', 'RSMVPD', 
                                          'RSNSR', 'RSXFS']].sum().sum(axis=1).reset_index(name='Total_Retail_Sales')

   # merge data and calc ratio
    cases_to_sales = pd.merge(monthly_cases, monthly_retail, on='Month')
    cases_to_sales['Cases_to_Sales_Ratio'] = cases_to_sales.eval('`Daily new confirmed cases of COVID-19 per million people` / Total_Retail_Sales')
    cases_to_sales['Month_numeric'] = cases_to_sales['Month'].dt.to_timestamp().astype('int64') / 1e9

    
    # Zoomed in graph
    zoomed_data = cases_to_sales[(cases_to_sales['Month'] >= '2020-11') & (cases_to_sales['Month'] <= '2021-06')]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=zoomed_data, x='Month_numeric', y='Cases_to_Sales_Ratio', color='blue', s=100)
    sns.regplot(data=zoomed_data, x='Month_numeric', y='Cases_to_Sales_Ratio', scatter=False, color='red', line_kws={'linewidth': 2}, ci=None)
    plt.title("Nov 2020 - Jun 2021 COVID to Sales Ratio Correlation")
    plt.xlabel("Month")
    plt.ylabel("Cases to Sales Ratio")
    plt.xticks(ticks=zoomed_data['Month_numeric'], labels=zoomed_data['Month'].dt.strftime('%Y-%m'), rotation=45)
    plt.grid(True)
    plt.show()

    # Overall graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cases_to_sales, x='Month_numeric', y='Cases_to_Sales_Ratio', color='blue', s=100)
    sns.regplot(data=cases_to_sales, x='Month_numeric', y='Cases_to_Sales_Ratio', scatter=False, color='red', line_kws={'linewidth': 2}, ci=None)
    plt.title("Overall COVID Cases to Sales Ratio Correlation")
    plt.xlabel("Month")
    plt.ylabel("Cases to Sales Ratio")
    plt.xticks(cases_to_sales['Month_numeric'], cases_to_sales['Month'].dt.strftime('%Y-%m'), rotation=45)
    plt.grid(True)
    plt.show()
    
def read_data(csv_file_path, start_year=None, end_year=None):
    """
    read and filter data
    parameters: csv file, end year, start year
    return: list of tup;les
    """
    data = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            try:
                # parse
                date = datetime.strptime(row[0], '%Y-%m-%d')
                year = date.year
            except ValueError:
                continue 
           
            # Filter
            if (start_year is None or year >= start_year) and (end_year is None or year <= end_year):
                gdp_value = float(row[1])
                data.append((date, gdp_value))
    return data



def main():
    ecommerce_regression('/Users/noahburra/Documents/DS2500 Project/ECOMSA.csv', degree=2)
    trade_sales_regression('/Users/noahburra/Documents/DS2500 Project/MRTSSM44000USS.csv', degree=2)
    covid('/Users/noahburra/Documents/DS2500 Project/2020_to_2023_covid.csv',
                            '/Users/noahburra/Documents/DS2500 Project/2020_to_2023_retail.csv')
    monkeypox('/Users/noahburra/Documents/DS2500 Project/2022_to_2024_mpox.csv',
                       '/Users/noahburra/Documents/DS2500 Project/2022_to_2024_retail.csv')
    

    # Unemploment file and graph
    file = '/Users/noahburra/Documents/DS2500 Project/unemployment.csv'  
    data = pd.read_csv(file)
    data['DATE'] = pd.to_datetime(data['DATE'])
    
    # moving average
    data['3_Month_MA'] = data['UNRATE'].rolling(3).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['DATE'], data['UNRATE'], label='Unemployment Rate', alpha=0.8)
    plt.plot(data['DATE'], data['3_Month_MA'], label='3-Month Moving Average', linewidth=2)
    

    plt.title('Unemployment Rate with Moving Average ')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # GDP output
    file_path = '/Users/noahburra/Documents/DS2500 Project/grossdomestic.csv'

    # Filter 
    start_year = 2010
    end_year = 2020
    gdp_data = read_data(file_path, start_year=start_year, end_year=end_year)

    # Extract
    dates, gdp_values = zip(*gdp_data)

    # Convert
    numeric_dates = np.array([date.year + date.timetuple().tm_yday / 365 for date in dates]).reshape(-1, 1)
    gdp_values = np.array(gdp_values).reshape(-1, 1)

    model = LinearRegression()
    model.fit(numeric_dates, gdp_values)

    # Predict
    predict = model.predict(numeric_dates)

    plt.figure(figsize=(12, 6))
    plt.scatter(dates, gdp_values, color='blue', label='Actual GDP', alpha=0.7)
    plt.plot(dates, predict, color='red', label='Regression Line', linewidth=2)
    plt.title('GDP Over Time')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('GDP (in billions)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

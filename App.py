#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from yellowbrick.cluster import SilhouetteVisualizer

# Set page configuration
st.set_page_config(page_title='Analysis of Greenhouse Gas Emissions', layout='wide')

# Title and Project Formulation
st.markdown("# Predicting Emissions Progress: Leveraging Data to Forecast Geographical Climate Target Success")
st.markdown("#### By Sumaia El-Kalache & Mounir Salem")
st.divider()

st.markdown("## Project Formulation")

st.markdown("### Purpose")
st.markdown("""
The purpose of the project is to analyze emissions data from 2016 and 2017 to assess progress in reducing greenhouse gas emissions. Using various machine learning models, the goal is to predict emission differences and determine whether countries, based on factors such as population and GDP, are on track to meet their reduction targets.
""")

st.markdown("### Expected Solution")
st.markdown("""
The project aims to offer a predictive model that assesses the likelihood of countries reaching their emissions reduction goals, using historical emissions data, population, and GDP as key predictors
""")

st.divider()

# Sidebar navigation
st.sidebar.title('Navigation')
pages = ['Data Collection & Cleaning', 'Data Exploration & Analysis', 'Clustering', 'Regression Analysis']
selection = st.sidebar.radio('Go to', pages)

# Function to load data with caching
@st.cache_data
def load_data():
    df1 = pd.read_csv('2016_-_Citywide_GHG_Emissions_20240207.csv')
    df2 = pd.read_csv('2017_-_Cities_Community_Wide_Emissions.csv')
    return df1, df2

df1, df2 = load_data()

# Data Cleaning and Combining
def data_cleaning_and_combining(df1, df2):
    # Cleaning df1
    df1['C40'] = df1['C40'].fillna('No')
    df1['Methodology Details'] = df1['Methodology Details'].fillna(df1['Methodology Details'].mode()[0])
    df1['Gases included'] = df1['Gases included'].fillna('Not specified')
    df1['Total City-wide Emissions (metric tonnes CO2e)'] = df1['Total City-wide Emissions (metric tonnes CO2e)'].fillna(
        df1['Total City-wide Emissions (metric tonnes CO2e)'].mean())
    df1['Total Scope 1 Emissions (metric tonnes CO2e)'] = df1['Total Scope 1 Emissions (metric tonnes CO2e)'].fillna(
        df1['Total Scope 1 Emissions (metric tonnes CO2e)'].mean())
    df1['Total Scope 2 Emissions (metric tonnes CO2e)'] = df1['Total Scope 2 Emissions (metric tonnes CO2e)'].fillna(
        df1['Total Scope 2 Emissions (metric tonnes CO2e)'].mean())
    df1['Increase/Decrease from last year'] = df1['Increase/Decrease from last year'].fillna(
        df1['Increase/Decrease from last year'].mode()[0])
    df1['Reason for increase/decrease in emissions'] = df1['Reason for increase/decrease in emissions'].fillna(
        'Not specified')
    df1['City GDP'] = df1['City GDP'].fillna(df1['City GDP'].mean())
    df1['GDP Currency'] = df1['GDP Currency'].fillna(df1['GDP Currency'].mode()[0])
    df1['Year of GDP'] = df1['Year of GDP'].fillna(df1['Year of GDP'].mode()[0])
    df1['GDP Source'] = df1['GDP Source'].fillna('Not specified')
    df1.columns = df1.columns.str.strip().str.replace('\u200b', '', regex=True)
    df1['Average annual temperature (in Celsius)'] = df1['Average annual temperature (in Celsius)'].fillna(
        df1['Average annual temperature (in Celsius)'].mean())
    df1['Average altitude (m)'] = df1['Average altitude (m)'].fillna(df1['Average altitude (m)'].mean())
    df1['Land area (in square km)'] = df1['Land area (in square km)'].fillna(df1['Land area (in square km)'].mean())
    df1['Current Population'] = df1['Current Population'].fillna(df1['Current Population'].mean())

    # Cleaning df2
    df2['C40'] = df2['C40'].fillna('No')
    df2['Protocol column'] = df2['Protocol column'].fillna(df2['Protocol column'].mode()[0])
    df2['Gases included'] = df2['Gases included'].fillna('Not specified')
    df2['Total emissions (metric tonnes CO2e)'] = df2['Total emissions (metric tonnes CO2e)'].fillna(
        df2['Total emissions (metric tonnes CO2e)'].mean())
    df2['Total Scope 1 Emissions (metric tonnes CO2e)'] = df2['Total Scope 1 Emissions (metric tonnes CO2e)'].fillna(
        df2['Total Scope 1 Emissions (metric tonnes CO2e)'].mean())
    df2['Total Scope 2 Emissions (metric tonnes CO2e)'] = df2['Total Scope 2 Emissions (metric tonnes CO2e)'].fillna(
        df2['Total Scope 2 Emissions (metric tonnes CO2e)'].mean())
    df2['Comment'] = df2['Comment'].fillna('No comment')
    df2['Increase/Decrease from last year'] = df2['Increase/Decrease from last year'].fillna(
        df2['Increase/Decrease from last year'].mode()[0])
    df2['Reason for increase/decrease in emissions'] = df2['Reason for increase/decrease in emissions'].fillna(
        'Not specified')
    df2['GDP'] = df2['GDP'].fillna(df2['GDP'].mean())
    df2['GDP Currency'] = df2['GDP Currency'].fillna(df2['GDP Currency'].mode()[0])
    df2['GDP Year'] = df2['GDP Year'].fillna(df2['GDP Year'].mode()[0])
    df2['GDP Source'] = df2['GDP Source'].fillna('Not specified')
    df2.columns = df2.columns.str.replace(r'\u200b', '', regex=True).str.strip()
    df2['Average annual temperature (in Celsius)'] = df2['Average annual temperature (in Celsius)'].fillna(
        df2['Average annual temperature (in Celsius)'].mean())
    df2['Average altitude (m)'] = df2['Average altitude (m)'].fillna(df2['Average altitude (m)'].mean())
    df2['Land area (in square km)'] = df2['Land area (in square km)'].fillna(df2['Land area (in square km)'].mean())
    df2['Population'] = df2['Population'].fillna(df2['Population'].mean())

    # Combine data
    df1_cleaned = df1[['City Name', 'Country', 'Reporting Year', 'Primary Methodology', 'C40',
                       'Total City-wide Emissions (metric tonnes CO2e)',
                       'Total Scope 1 Emissions (metric tonnes CO2e)',
                       'Total Scope 2 Emissions (metric tonnes CO2e)',
                       'Current Population', 'City GDP',
                       'Average annual temperature (in Celsius)',
                       'Land area (in square km)', 'Average altitude (m)']]

    df2_cleaned = df2[['City', 'Country', 'Reporting year', 'C40',
                       'Total emissions (metric tonnes CO2e)',
                       'Total Scope 1 Emissions (metric tonnes CO2e)',
                       'Total Scope 2 Emissions (metric tonnes CO2e)',
                       'Population', 'GDP',
                       'Average annual temperature (in Celsius)',
                       'Average altitude (m)',
                       'Land area (in square km)']]

    df1_cleaned.rename(columns={
        'City Name': 'City',
        'Reporting Year': 'Reporting year',
        'Total City-wide Emissions (metric tonnes CO2e)': 'Total emissions (metric tonnes CO2e)',
        'Current Population': 'Population',
        'City GDP': 'GDP',
    }, inplace=True)

    combined_df = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)
    return combined_df

# Call the function to get combined_df
combined_df = data_cleaning_and_combining(df1, df2)

if selection == 'Data Collection & Cleaning':
    st.title('Data Collection & Cleaning')

    st.subheader('Initial DataFrames')
    st.write('First 10 rows of df1:')
    st.dataframe(df1.head(10))

    st.write('First 10 rows of df2:')
    st.dataframe(df2.head(10))

    st.subheader('After Data Cleaning and Combining')
    st.write('Combined DataFrame:')
    st.dataframe(combined_df.head())

elif selection == 'Data Exploration & Analysis':
    st.title('Data Exploration & Analysis')

    # Total Emissions per Country
    st.subheader('Total Emissions per Country')
    country_emissions = combined_df.groupby('Country')['Total emissions (metric tonnes CO2e)'].sum().reset_index()

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Total emissions (metric tonnes CO2e)', y='Country', data=country_emissions, ax=ax1)
    ax1.set_title('Total Emissions per Country')
    ax1.set_xlabel('Total Emissions (metric tonnes CO2e)')
    ax1.set_ylabel('Country')
    st.pyplot(fig1)

    # Comment and Conclusion
    st.write("""
    **Observation:** The bar chart shows the total greenhouse gas emissions for each country in the dataset. Countries like the United States and China have significantly higher emissions compared to others.
    
    **Conclusion:** There is a substantial disparity in emissions between countries, indicating that certain nations contribute more heavily to global emissions. This suggests the need for targeted policies in high-emission countries.
    """)

    # Total Emissions vs. Population
    st.subheader('Total Emissions vs. Population')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=combined_df, x='Population', y='Total emissions (metric tonnes CO2e)', ax=ax2)
    ax2.set_title('Total Emissions vs. Population')
    ax2.set_xlabel('Population')
    ax2.set_ylabel('Total Emissions (metric tonnes CO2e)')
    st.pyplot(fig2)

    # Comment and Conclusion
    st.write("""
    **Observation:** The scatter plot illustrates the relationship between a city's population and its total emissions. There appears to be a positive correlation.
    
    **Conclusion:** Larger populations tend to be associated with higher emissions. However, the data shows significant variability, indicating that other factors such as industrial activity, energy sources, and environmental policies also play critical roles in determining emission levels.
    """)

    # Correlation Matrix
    st.subheader('Correlation Matrix')
    data_column_category = combined_df.select_dtypes(exclude=[np.number])
    label_encoder = LabelEncoder()
    for i in data_column_category:
        combined_df[i] = label_encoder.fit_transform(combined_df[i])

    correlation_matrix = combined_df.corr()
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=ax3)
    ax3.set_title('Correlation Matrix')
    st.pyplot(fig3)

    # Detailed Comment and Conclusion
    st.write("""
    **Observation:** Here we see a high positive correlation between Population and Total emissions (e.g., correlation coefficient close to +1) suggests that as the population increases, total emissions tend to increase as well. Similarly, GDP shows a positive correlation with Total emissions, indicating that wealthier cities may emit more greenhouse gases
    
    **Conclusion:** The strong positive correlation between population and total emissions confirms that more populous cities tend to emit more greenhouse gases, likely due to increased energy consumption, transportation needs, and industrial activities. The positive correlation with GDP indicates that wealthier cities may have higher emissions, possibly due to greater industrialization and consumption patterns. However, the correlations are not perfect (i.e., less than 1), implying that other factors influence emissions. This underscores the complexity of emissions dynamics and the need to consider multiple variables in predictive models.
    """)

    # Total Emissions by Primary Methodology
    st.subheader('Total Emissions by Primary Methodology')
    # Do not apply label encoding here for Primary Methodology
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=combined_df, x='Primary Methodology', y='Total emissions (metric tonnes CO2e)', ax=ax4)
    ax4.set_title('Total Emissions by Primary Methodology')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.set_ylabel('Total Emissions (metric tonnes CO2e)')
    st.pyplot(fig4)


    # Detailed Comment and Conclusion
    st.write("""
    **Observation:** The box and whisker plot illustrates the distribution of total emissions for each primary methodology used in the data reporting. Variations in the median, quartiles, and presence of outliers are evident among different methodologies. Some methodologies have a wide interquartile range, indicating significant variability in the data, while others are more tightly clustered.

    **Conclusion:** There is significant variability in emissions across different methodologies. Some methodologies show a wider spread and higher median emissions, suggesting inconsistencies in how emissions are calculated or reported. This variability can introduce challenges in comparing emissions across cities and may affect the reliability of analyses. Standardizing methodologies or adjusting for methodological differences may be necessary to ensure accurate comparisons and assessments.
    """)

    # Emissions Comparison between Scope 1 and Scope 2
    st.subheader('Emissions Comparison between Scope 1 and Scope 2')
    categories = [
        'Total Scope 1 Emissions (metric tonnes CO2e)',
        'Total Scope 2 Emissions (metric tonnes CO2e)'
    ]

    total_scope_1_emissions = combined_df['Total Scope 1 Emissions (metric tonnes CO2e)'].sum()
    total_scope_2_emissions = combined_df['Total Scope 2 Emissions (metric tonnes CO2e)'].sum()

    values = [total_scope_1_emissions, total_scope_2_emissions]

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.bar(categories, values, color=['orange', 'lightgreen'])
    ax5.set_title('Emissions Comparison')
    ax5.set_ylabel('Emissions (metric tonnes CO2e)')
    ax5.set_xlabel('Emission Categories')
    ax5.set_xticklabels(categories, rotation=15)
    st.pyplot(fig5)

    # Comment and Conclusion
    st.write("""
    **Observation:** The bar chart compares the aggregate total emissions from Scope 1 (direct emissions) and Scope 2 (indirect emissions from purchased energy).

    **Conclusion:** Scope 1 emissions are significantly higher than Scope 2 emissions, indicating that direct emissions are the primary contributor to total greenhouse gas emissions in the dataset. This emphasizes the importance of focusing on direct emission sources, such as transportation and on-site energy production, when developing strategies for emission reduction.
    """)

elif selection == 'Clustering':
    st.title('Clustering')

    st.write('Preparing data for clustering...')

    # Include 'Country' in clustering features
    columns_to_scale = ['Country', 'Total emissions (metric tonnes CO2e)', 'Population', 'GDP',
                        'Average annual temperature (in Celsius)']
    scaled_df = combined_df[columns_to_scale].copy()
    scaled_df = scaled_df.dropna()

    # Encode 'Country' using LabelEncoder
    label_encoder = LabelEncoder()
    scaled_df['Country'] = label_encoder.fit_transform(scaled_df['Country'].astype(str))

    # Scale the features
    scaler = MinMaxScaler()
    scaled_df[columns_to_scale] = scaler.fit_transform(scaled_df[columns_to_scale])

    X = scaled_df  # Use all features including 'Country'

    # KMeans clustering
    distortions = []
    K = range(2, 10)
    for k in K:
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    st.subheader('Elbow Method for Optimal K')
    fig1, ax1 = plt.subplots()
    ax1.plot(K, distortions, 'bx-')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Distortion')
    ax1.set_title('Elbow Method showing the optimal K')
    st.pyplot(fig1)

    scores = []
    for k in K:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        score = metrics.silhouette_score(X, model.labels_)
        scores.append(score)

    st.subheader('Silhouette Score Method for Optimal K')
    fig2, ax2 = plt.subplots()
    ax2.plot(K, scores, 'bx-')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score showing the optimal K')
    st.pyplot(fig2)

    st.write('Select the number of clusters based on the methods above:')
    num_clusters = st.slider('Number of clusters', min_value=2, max_value=9, value=3)

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(X)

    st.subheader('Silhouette Visualizer')
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(X)
    visualizer.finalize()
    st.pyplot(visualizer.fig)

    st.subheader('Predict Cluster for New Data')
    country_input = st.selectbox('Select Country', combined_df['Country'].unique())
    total_emissions_input = st.number_input('Total emissions (metric tonnes CO2e)', min_value=0.0)
    population_input = st.number_input('Population', min_value=0.0)
    gdp_input = st.number_input('GDP', min_value=0.0)
    temperature_input = st.number_input('Average annual temperature (in Celsius)', min_value=-50.0, max_value=50.0)

    # Encode the country input
    country_encoded = label_encoder.transform([country_input])[0]

    new_data = pd.DataFrame({
        'Country': [country_encoded],
        'Total emissions (metric tonnes CO2e)': [total_emissions_input],
        'Population': [population_input],
        'GDP': [gdp_input],
        'Average annual temperature (in Celsius)': [temperature_input]
    })

    # Scale the new data
    new_data[columns_to_scale] = scaler.transform(new_data[columns_to_scale])

    # Predict the cluster
    new_cluster = kmeans.predict(new_data)[0]
    st.write(f'The provided data belongs to cluster: {new_cluster}')

elif selection == 'Regression Analysis':
    st.title('Regression Analysis')

    regression_options = ['Linear Regression', 'Multiple Regression', 'Polynomial Regression']
    regression_choice = st.selectbox('Select Regression Type', regression_options)

    if regression_choice == 'Linear Regression':
        st.subheader('Linear Regression')

        regression_df = combined_df[['Total emissions (metric tonnes CO2e)', 'Population']].dropna()

        scaler = StandardScaler()
        regression_df[['Total emissions (metric tonnes CO2e)', 'Population']] = scaler.fit_transform(
            regression_df[['Total emissions (metric tonnes CO2e)', 'Population']])

        X = regression_df[['Population']]
        y = regression_df['Total emissions (metric tonnes CO2e)']

        test_size = st.slider('Select test size', min_value=0.1, max_value=0.9, value=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write('Model Coefficients:')
        st.write(f'Intercept: {model.intercept_:.3f}')
        st.write(f'Coefficient: {model.coef_[0]:.3f}')

        st.write('Model Performance Metrics:')
        st.write(f'MAE: {metrics.mean_absolute_error(y_test, y_pred):.3f}')
        st.write(f'MSE: {metrics.mean_squared_error(y_test, y_pred):.3f}')
        st.write(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.3f}')
        st.write(f'R²: {metrics.r2_score(y_test, y_pred):.3f}')

        st.subheader('Predicted vs. Actual Values')
        fig8, ax8 = plt.subplots()
        ax8.scatter(y_test, y_pred)
        ax8.set_xlabel('Actual Values')
        ax8.set_ylabel('Predicted Values')
        ax8.set_title('Predicted vs. Actual Values')
        st.pyplot(fig8)

        st.subheader('Predict Total Emissions Based on Population')
        population_input = st.number_input('Enter Population', min_value=0.0)
        population_scaled = scaler.transform([[0, population_input]])[0][1].reshape(-1, 1)
        predicted_emissions_scaled = model.predict(population_scaled)
        predicted_emissions = scaler.inverse_transform([[predicted_emissions_scaled[0], 0]])[0][0]
        st.write(f'Predicted Total Emissions: {predicted_emissions:.2f}')

    elif regression_choice == 'Multiple Regression':
        st.subheader('Multiple Regression')

        # Create scaled_df by scaling combined_df
        scaler = StandardScaler()
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
        scaled_df = pd.DataFrame(scaler.fit_transform(combined_df[numeric_columns]), columns=numeric_columns)

        # Shuffle the data
        df_shuffled = shuffle(scaled_df, random_state=42)

        # 'Population' and 'GDP' as independent variables
        #regression_df = scaled_df[['Total emissions (metric tonnes CO2e)', 'Population', 'GDP']].dropna()

        X = df_shuffled[['Population', 'GDP']]
        y = df_shuffled['Total emissions (metric tonnes CO2e)']

        # Handle missing values (though scaling should have taken care of NaNs)
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        test_size = st.slider('Select test size', min_value=0.1, max_value=0.9, value=0.2)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_predictions = model.predict(X_test_scaled)

        feature_names = ['Population', 'GDP']
        equation = ' + '.join(f'({coef:.2f} × {name})' for coef, name in zip(model.coef_, feature_names))

        st.write('Model Coefficients:')
        st.write(f'Intercept: {model.intercept_:.2f}')
        coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
        st.dataframe(coeff_df)

        st.write('Equation:')
        st.latex(f'y = {model.intercept_:.2f} + {equation}')

        # Display metrics
        metrics_dict = {
            'MAE': metrics.mean_absolute_error(y_test, y_predictions),
            'MSE': metrics.mean_squared_error(y_test, y_predictions),
            'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_predictions)),
            'R-Squared': metrics.r2_score(y_test, y_predictions)
        }
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value']).round(3)
        st.dataframe(metrics_df)

        st.subheader('Predicted vs. Actual Values')
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_predictions)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs. Actual Values')
        st.pyplot(fig)

        st.subheader('Predict Total Emissions Based on Inputs')
        population_input = st.number_input('Enter Population (scaled value)', min_value=-3.0, max_value=3.0, value=0.0)
        gdp_input = st.number_input('Enter GDP (scaled value)', min_value=-3.0, max_value=3.0, value=0.0)

        input_data = np.array([[population_input, gdp_input]])
        predicted_emissions = model.predict(input_data)[0]
        st.write(f'Predicted Total Emissions (scaled): {predicted_emissions:.2f}')

    elif regression_choice == 'Polynomial Regression':
        st.subheader('Polynomial Regression')

        # Create scaled_df by scaling combined_df
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(combined_df.select_dtypes(include=[np.number])), columns=combined_df.select_dtypes(include=[np.number]).columns)

        # Shuffle the data
        scaled_df = shuffle(scaled_df, random_state=42)

        regression_df = scaled_df[['Total emissions (metric tonnes CO2e)', 'Population']].dropna()

        X = regression_df[['Population']].values
        y = regression_df['Total emissions (metric tonnes CO2e)'].values

        # No need to scale X and y here as they are already scaled
        test_size = st.slider('Select test size', min_value=0.1, max_value=0.9, value=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        degree = 2  # Fixed degree as per your example
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        pol_reg = LinearRegression()
        pol_reg.fit(X_train_poly, y_train)

        y_predict = pol_reg.predict(X_test_poly)

        # Generate formula string
        terms = []
        for i, coef in enumerate(pol_reg.coef_):
            if coef != 0:
                term = f"{coef:.2f}"
                if i > 0:
                    powers = poly.powers_[i]
                    variables = []
                    for power, feature in zip(powers, ['Population']):
                        if power == 1:
                            variables.append(f"{feature}")
                        elif power > 1:
                            variables.append(f"{feature}^{power}")
                    term += "×" + "×".join(variables)
                terms.append(term)
        intercept = pol_reg.intercept_
        formula = "y = " + f"{intercept:.2f} + " + " + ".join(terms[1:])  # Skip the intercept term in coefficients

        st.write('Equation:')
        st.latex(formula)

        # Display metrics
        metrics_dict = {
            'MAE': metrics.mean_absolute_error(y_test, y_predict),
            'MSE': metrics.mean_squared_error(y_test, y_predict),
            'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_predict)),
            'R-Squared': metrics.r2_score(y_test, y_predict)
        }
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value']).round(3)
        st.dataframe(metrics_df)

        st.subheader('Predicted vs. Actual Values')
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='red', label='Actual Data')
        # Generate a sequence of values for plotting the polynomial regression line
        X_sequence = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
        X_sequence_poly = poly.transform(X_sequence)
        y_sequence_pred = pol_reg.predict(X_sequence_poly)
        ax.plot(X_sequence, y_sequence_pred, color='blue', label='Polynomial Regression')
        ax.set_xlabel('Population')
        ax.set_ylabel('Total Emissions')
        ax.set_title('Polynomial Regression')
        ax.legend()
        st.pyplot(fig)

        st.subheader('Predict Total Emissions Based on Population')
        population_input = st.number_input('Enter Population (scaled value)', min_value=-3.0, max_value=3.0, value=0.0)
        input_data = np.array([[population_input]])
        input_data_poly = poly.transform(input_data)
        predicted_emissions = pol_reg.predict(input_data_poly)[0]
        st.write(f'Predicted Total Emissions (scaled): {predicted_emissions:.2f}')

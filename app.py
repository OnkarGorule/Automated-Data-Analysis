from flask import Flask, request, jsonify, render_template, redirect, url_for
from itertools import combinations
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

app = Flask(__name__)

# Store the DataFrame globally (for demonstration purposes; avoid in production)
df = None

# Helper functions to check if a variable is continuous or discrete
def is_continuous(series):
    return series.dtype in [np.float64, np.int64] and series.dropna().nunique() > 10

def is_discrete(series):
    return (series.dtype in [np.int64, object] or series.nunique() <= 10) and not is_continuous(series)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/machine')
def machine():
    return render_template('machine.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global df
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        csv_file = request.files['csv_file']
        df = pd.read_csv(csv_file)

        # Classify columns as continuous or discrete
        all_vars = [col for col in df.columns]
        continuous_vars = [col for col in df.columns if is_continuous(df[col])]
        discrete_vars = [col for col in df.columns if is_discrete(df[col])]

        return jsonify({
            'all_vars': all_vars,
            'continuous_vars': continuous_vars,
            'discrete_vars': discrete_vars
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 400

@app.route('/perform_operation', methods=['POST'])
def perform_operation():
    global df
    if df is None:
        return jsonify({'error': 'No CSV file uploaded'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    variables = data.get('variables', [])
    descriptive = data.get('descriptive', [])

    if not variables or not descriptive:
        return jsonify({'error': 'Please select at least one variable and one operation'}), 400

    results = {}
    for variable in variables:
        if variable in df.columns:
            variable_results = {}
            for operation in descriptive:
                try:
                    if operation == 'mean':
                        variable_results['mean'] = df[variable].mean()
                    elif operation == 'median':
                        variable_results['median'] = df[variable].median()
                    elif operation == 'mode':
                        variable_results['mode'] = df[variable].mode()[0] if not df[variable].mode().empty else None
                    elif operation == 'sum':
                        variable_results['sum'] = df[variable].sum()
                    elif operation == 'stddev':
                        variable_results['stddev'] = df[variable].std()
                    elif operation == 'variance':
                        variable_results['variance'] = df[variable].var()
                    elif operation == 'min':
                        variable_results['min'] = df[variable].min()
                    elif operation == 'max':
                        variable_results['max'] = df[variable].max()
                    elif operation == 'range':
                        variable_results['range'] = df[variable].max() - df[variable].min()
                    elif operation == 'q1':
                        variable_results['q1'] = df[variable].quantile(0.25)
                    elif operation == 'q3':
                        variable_results['q3'] = df[variable].quantile(0.75)
                    elif operation == 'iqr':
                        variable_results['iqr'] = df[variable].quantile(0.75) - df[variable].quantile(0.25)
                    elif operation == 'skew':
                        variable_results['skew'] = stats.skew(df[variable].dropna())
                    elif operation == 'kurtosis':
                        variable_results['kurtosis'] = stats.kurtosis(df[variable].dropna())
                    else:
                        variable_results[operation] = 'Invalid operation'
                except Exception as e:
                    variable_results[operation] = f"Error: {str(e)}"
            results[variable] = variable_results

    return jsonify({'result': results})

@app.route('/generate_visualization', methods=['POST'])
def generate_visualization():
    global df
    if df is None:
        return jsonify({'error': 'No CSV file uploaded'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    variables = data.get('variables', [])
    graph_type = data.get('visualization', None)

    if not variables or not graph_type:
        return jsonify({'error': 'Please select at least one variable and a graph type'}), 400

    results = {}
    output_dir = "static/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        if graph_type == 'heatmap':
            if len(variables) > 1:
                corr_matrix = df[variables].corr()  # Compute correlation matrix
                
                fig = px.imshow(    
                    corr_matrix, 
                    labels=dict(color="Correlation"), 
                    x=variables, 
                    y=variables, 
                    color_continuous_scale="greens", 
                    text_auto=".2f"
                )

                fig.update_layout(title="Heatmap of Selected Variables")

                # Save as interactive HTML
                output_path = f"static/heatmap_{'_'.join(variables)}.html"
                fig.write_html(output_path)
                results['heatmap'] = f"/{output_path}"
            else:
                results['heatmap'] = {'error': 'Heatmap requires multiple variables'}

        elif graph_type == 'pairplot':
            if len(variables) > 1:
                fig = px.scatter_matrix(
                    df[variables], 
                    dimensions=variables, 
                    title="Pairplot of Selected Variables", 
                   
                )

                # Save as interactive HTML
                output_path = f"static/pairplot_{'_'.join(variables)}.html"
                fig.write_html(output_path)
                results['pairplot'] = f"/{output_path}"
            else:
                results['pairplot'] = {'error': 'Pair plot requires multiple variables'}

        elif graph_type == 'scatter':
            if len(variables) == 2:
                var1, var2 = variables

                fig = px.scatter(df, x=var1, y=var2, title=f'Scatter Plot: {var1} vs {var2}',
                                labels={var1: var1, var2: var2}, opacity=0.7)

                # Save as interactive HTML
                output_path = f"static/scatter_{var1}_vs_{var2}.html"
                fig.write_html(output_path)
                results['scatter'] = f"/{output_path}"
            else:
                results['scatter'] = {'error': 'Scatter plot requires exactly two variables'}


        else:
            for variable in variables:
                if variable not in df.columns:
                    results[variable] = {'error': 'Variable not found in the dataset'}
                    continue

                output_filename = f"{variable}_{graph_type}.html"
                output_path = os.path.join(output_dir, output_filename)

                if graph_type == 'pie':
                    count = df[variable].value_counts()

                    fig = px.pie(
                        names=count.index, 
                        values=count.values, 
                        title=f'Pie Chart: {variable}', 
                        color_discrete_sequence=px.colors.sequential.Greens, 
                        hole=0  # Set to 0 for a full pie chart, adjust for a donut chart
                    )
                    # Save Plotly figure as an interactive HTML file
                    output_path = f"static/{variable}_pie_chart.html"
                    fig.write_html(output_path)

                elif graph_type == 'histogram':
                    fig = px.histogram(df, x=variable, nbins=10, title=f'Histogram: {variable}', 
                                    labels={variable: variable, 'count': 'Frequency'},
                                    opacity=0.7, color_discrete_sequence=['blue'])
                    # Save Plotly figure as an HTML file instead of an image
                    output_path = f"static/{variable}_histogram.html"
                    fig.write_html(output_path)
                    
                elif graph_type == 'boxplot':
                    fig = px.box(df, y=variable, title=f'Boxplot: {variable}')

                    # Save as interactive HTML
                    output_path = f"static/boxplot_{variable}.html"
                    fig.write_html(output_path)
                    results['boxplot'] = f"/{output_path}"

                elif graph_type == 'line':
                    fig = px.line(df, y=variable, title=f'Line Chart: {variable}', labels={'index': 'Index', variable: variable})

                    # Save as interactive HTML
                    output_path = f"static/line_chart_{variable}.html"
                    fig.write_html(output_path)
                    results['line'] = f"/{output_path}"

                results[variable] = f"/{output_path}"
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': results})

@app.route('/generate_hypothesis', methods=['POST'])
def generate_hypothesis():
    global df
    if df is None:
        return jsonify({'error': 'No CSV file uploaded'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    variables = data.get('variables', [])
    tests = data.get('test', [])
    population_mean = data.get('population_mean', None)
    distribution = data.get('distribution', 'norm')  # Required for KS one-sample test

    # Validate input
    if not variables:
        return jsonify({'error': 'Please select at least one variable'}), 400
    if not tests:
        return jsonify({'error': 'Please select at least one test'}), 400

    # Check if variables exist in DataFrame
    invalid_variables = [var for var in variables if var not in df.columns]
    if invalid_variables:
        return jsonify({'error': f"Invalid variables: {', '.join(invalid_variables)}"}), 400

    results = []

    for test in tests:
        try:
            # Parametric tests
            if test == 'anova':
                if len(variables) < 3:
                    return jsonify({'error': 'ANOVA requires at least three variables.'}), 400
                groups = [df[var].dropna() for var in variables]
                if any(len(group.unique()) < 2 for group in groups):  # Ensure variance
                    results.append({'test': 'ANOVA', 'error': 'All groups must have more than one unique value.'})
                else:
                    f_stat, p_value = stats.f_oneway(*groups)
                    results.append({
                        'variables': variables,
                        'test': 'ANOVA',
                        'statistic': f_stat,
                        'p_value': p_value,
                        'conclusion' : "Conclusion"
                    })        

            elif test == 'bartlett':
                if len(variables) < 2:
                    return jsonify({'error': 'Bartlett Test requires at least two variables.'}), 400
                groups = [df[var].dropna() for var in variables]
                if any(len(group) < 2 for group in groups):
                    results.append({
                        'test': 'Bartlett Test',
                        'error': 'All groups must have at least two values.'
                    })
                else:
                    bart_stat, p_value = stats.bartlett(*groups)
                    results.append({
                        'variables': variables,
                        'test': 'Bartlett Test',
                        'statistic': bart_stat,
                        'p_value': p_value,
                        'conclusion' : "Conclusion"
                    })

            elif test == 't-test-ind':
                if len(variables) != 2:
                    return jsonify({'error': 'T-Test Independent requires exactly two variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                results.append({
                    'variable1': variables[0],
                    'variable2': variables[1],
                    'test': 'T-Test Independent',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            elif test == 't-test-pair':
                if len(variables) != 2:
                    return jsonify({'error': 'Paired T-Test requires exactly two variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                if len(group1) != len(group2):
                    results.append({
                        'test': 'Paired T-Test',
                        'error': 'Groups must have the same length.'
                    })
                else:
                    t_stat, p_value = stats.ttest_rel(group1, group2)
                    results.append({
                        'variable1': variables[0],
                        'variable2': variables[1],
                        'test': 'Paired T-Test',
                        'statistic': t_stat,
                        'p_value': p_value,
                        'conclusion' : "Conclusion"
                    })

            elif test == 't-test-one':
                if len(variables) != 1:
                    return jsonify({'error': 'One-Sample T-Test requires exactly one variable.'}), 400
                if population_mean is None:
                    return jsonify({'error': 'One-Sample T-Test requires a population mean.'}), 400
                sample = df[variables[0]].dropna()
                t_stat, p_value = stats.ttest_1samp(sample, popmean=population_mean)
                results.append({
                    'variable': variables[0],
                    'test': 'One-Sample T-Test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            elif test == 'z-test':
                if len(variables) != 2:
                    return jsonify({'error': 'Z-Test requires exactly two variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                z_stat, p_value = stats.ztest(group1, group2, alternative='two-sided')
                results.append({
                    'variable1': variables[0],
                    'variable2': variables[1],
                    'test': 'Z-Test',
                    'statistic': z_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            elif test == 'f-test':
                if len(variables) != 2:
                    return jsonify({'error': 'F-Test requires exactly two variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
                f_stat = var1 / var2
                p_value = 1 - min(f_stat, 1/f_stat)  # Approximation
                results.append({
                    'variable1': variables[0],
                    'variable2': variables[1],
                    'test': 'F-Test',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            elif test == 'chi-test':
                if len(variables) != 2:
                    return jsonify({'error': 'Chi-Square Test requires exactly two categorical variables.'}), 400
                if not (df[variables[0]].dtype == 'object' or pd.api.types.is_categorical_dtype(df[variables[0]])):
                    results.append({
                        'test': 'Chi-Square Test',
                        'error': 'Both variables must be categorical.'
                    })
                else:
                    contingency_table = pd.crosstab(df[variables[0]], df[variables[1]])
                    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    results.append({
                        'variable1': variables[0],
                        'variable2': variables[1],
                        'test': 'Chi-Square Test',
                        'statistic': chi2_stat,
                        'p_value': p_value,
                        'conclusion' : "Conclusion"
                    })

            # elif test == 'anderson':
            #     if len(variables) != 1:
            #         return jsonify({'error': 'Select one variable at a time.'}), 400
            #     var = df[variables[0]].dropna()
            #     result = stats.anderson(var)
            #     stat = result.statistic
            #     crit = result.critical_values[2]
            #     results.append({
            #         'variable': variables[0],
            #         'test': 'Anderson Darling Test',
            #         'statistic': stat,
            #         'conclusion' : "Conclusion"
            #     })

            elif test == 'shapiro':
                if len(variables) != 1:
                    return jsonify({'error': 'Select one variable at a time.'}), 400
                var = df[variables[0]].dropna()
                s_stat, p_value = stats.shapiro(var)
                results.append({
                    'variable': variables[0],
                    'test': 'Shapiro Wilk Test',
                    'statistic': s_stat,
                    'p_value': p_value,
                    'conclusion' : "The data follows normal distrbution" if p_value < 0.05 else "The data does not follows normal distrbution"
                })

            # Non-Parametric tests

            # elif test == 'ksto':
            #     if len(variables) != 1:
            #         return jsonify({'error': 'One-Sample KS Test requires exactly one variable.'}), 400
            #     sample = df[variables[0]].dropna()
            #     ks_stat, p_value = ks_1samp(sample, distribution)
            #     results.append({
            #         'variable': variables[0],
            #         'test': 'One-Sample KS Test',
            #         'statistic': ks_stat,
            #         'p_value': p_value
            #     })

            # Kolmogorov-Smirnov Two-Sample Test
            elif test == 'kstt':
                if len(variables) != 2:
                    return jsonify({'error': 'Two-Sample KS Test requires exactly two variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                ks_stat, p_value = stats.ks_2samp(group1, group2)
                results.append({
                    'variable1': variables[0],
                    'variable2': variables[1],
                    'test': 'Two-Sample KS Test',
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            # Wilcoxon Signed-Rank Test
            elif test == 'wilcoxon':
                if len(variables) != 2:
                    return jsonify({'error': 'Wilcoxon Signed-Rank Test requires exactly two paired variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                if len(group1) != len(group2):
                    results.append({
                        'test': 'Wilcoxon Signed-Rank Test',
                        'error': 'Groups must have the same length.'
                    })
                else:
                    wilcoxon_stat, p_value = stats.wilcoxon(group1, group2)
                    results.append({
                        'variable1': variables[0],
                        'variable2': variables[1],
                        'test': 'Wilcoxon Signed-Rank Test',
                        'statistic': wilcoxon_stat,
                        'p_value': p_value,
                        'conclusion' : "Conclusion"
                    })

            elif test == 'median':
                if len(variables) < 2:
                    return jsonify({'error': 'Median Test requires at least two variables.'}), 400
                groups = [df[var].dropna() for var in variables]
                stat, p_value, _, _ = stats.median_test(*groups)
                results.append({
                    'variables': variables,
                    'test': 'Median Test',
                    'statistic': stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            # Run Test (Wald-Wolfowitz Runs Test for Randomness)
            elif test == 'wald-test':
                if len(variables) != 1:
                    return jsonify({'error': 'Run Test requires exactly one variable.'}), 400
                sample = df[variables[0]].dropna()
                z_stat, p_value = stats.runstest_1samp(sample, correction=False)
                results.append({
                    'variable': variables[0],
                    'test': 'Run Test (Wald-Wolfowitz)',
                    'statistic': z_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })
            elif test == 'mann-whitney':
                if len(variables) != 2:
                    return jsonify({'error': 'Mann-Whitney U Test requires exactly two variables.'}), 400
                group1, group2 = df[variables[0]].dropna(), df[variables[1]].dropna()
                u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                results.append({
                    'variable1': variables[0],
                    'variable2': variables[1],
                    'test': 'Mann-Whitney U Test',
                    'statistic': u_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            # Cochran's Q Test (For related binary data)
            # elif test == 'cochran':
            #     if len(variables) < 2:
            #         return jsonify({'error': 'Cochran\'s Q Test requires at least two variables.'}), 400
            #     q_stat, p_value = cochranqtest(df[variables].dropna())
            #     results.append({
            #         'variables': variables,
            #         'test': 'Cochran\'s Q Test',
            #         'statistic': q_stat,
            #         'p_value': p_value
            #     })

            # Kruskal-Wallis Test (Non-parametric ANOVA for independent groups)
            elif test == 'kruskal-wallis':
                if len(variables) < 2:
                    return jsonify({'error': 'Kruskal-Wallis Test requires at least two variables.'}), 400
                groups = [df[var].dropna() for var in variables]
                h_stat, p_value = stats.kruskal(*groups)
                results.append({
                    'variables': variables,
                    'test': 'Kruskal-Wallis Test',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })

            # Friedman’s Test (For repeated measures comparison)
            elif test == 'friedman':
                if len(variables) < 2:
                    return jsonify({'error': 'Friedman\'s Test requires at least two variables.'}), 400
                groups = [df[var].dropna() for var in variables]
                f_stat, p_value = stats.friedmanchisquare(*groups)
                results.append({
                    'variables': variables,
                    'test': 'Friedman\'s Test',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'conclusion' : "Conclusion"
                })
            else:
                results.append({
                    'test': test,
                    'error': 'Invalid test'
                })

        except Exception as e:
            results.append({'test': test, 'error': str(e)})

    return jsonify({'result': results})

@app.route('/generate_correlation', methods=['POST'])
def generate_correlation():
    global df
    if df is None:
        return jsonify({'error': 'No CSV file uploaded'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    variables = data.get('variables', [])
    correlation = data.get('correlation', None)

    if not variables or not correlation:
        return jsonify({'error': 'Please select at least two variables and a correlation method'}), 400

    if len(variables) < 2:
        return jsonify({'error': 'Please select at least two variables for correlation analysis'}), 400

    results = {}

    # Compute pairwise correlations
    for var1, var2 in combinations(variables, 2):
        if var1 not in df.columns or var2 not in df.columns:
            results[f"{var1}  {var2}"] = 'Invalid variables'
            continue

        try:
            if correlation == 'pearson':
                corr_value, p_value = stats.pearsonr(df[var1], df[var2])
            elif correlation == 'spearman':
                corr_value, p_value = stats.spearmanr(df[var1], df[var2])
            elif correlation == 'kendall':
                corr_value, p_value = stats.kendalltau(df[var1], df[var2])
            else:
                results[f"{var1} {var2}"] = 'Invalid correlation method'
                continue

            results[f"{var1} {var2}"] = {
                'correlation_value': corr_value,
                'p_value': p_value
            }
        except Exception as e:
            results[f"{var1} {var2}"] = f"Error: {str(e)}"

    return jsonify({'result': results})

@app.route('/generate_regression', methods=['POST'])
def generate_regression():
    global df
    if df is None:
        return jsonify({'error': 'No CSV file uploaded'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    variables = data.get('variables', [])
    correlation = data.get('correlation', None)

    if not variables or not correlation:
        return jsonify({'error': 'Please select at least two variables and a correlation method'}), 400

    if len(variables) < 2:
        return jsonify({'error': 'Please select at least two variables for correlation analysis'}), 400

    results = {}

     # Compute pairwise correlations
    for var1, var2 in combinations(variables, 2):
        if var1 not in df.columns or var2 not in df.columns:
            results[f"{var1}  {var2}"] = 'Invalid variables'
            continue

        try:
            if correlation == 'pearson':
                corr_value, p_value = stats.pearsonr(df[var1], df[var2])
            elif correlation == 'spearman':
                corr_value, p_value = stats.spearmanr(df[var1], df[var2])
            elif correlation == 'kendall':
                corr_value, p_value = stats.kendalltau(df[var1], df[var2])
            else:
                results[f"{var1} {var2}"] = 'Invalid correlation method'
                continue

            results[f"{var1} {var2}"] = {
                'correlation_value': corr_value,
                'p_value': p_value
            }
        except Exception as e:
            results[f"{var1} {var2}"] = f"Error: {str(e)}"

    return jsonify({'result': results})

if __name__ == '__main__':
    app.run(debug=True)

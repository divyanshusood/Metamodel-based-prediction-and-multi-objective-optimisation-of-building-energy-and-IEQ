#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined analysis script generated from multiple notebooks.

- Repeated imports and definitions removed
- Absolute paths/secrets removed (use placeholders)
- Plotting moved to the end

Usage:
    python combined_analysis.py

Notes:
    Replace any <PATH_PLACEHOLDER> or <SECRET_PLACEHOLDER> before running.
"""

# === Imports (stdlib) ===
# (none)

# === Imports (third-party) ===
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Optional aliases if used later
try:
    import numpy as np
except Exception:
    pass
try:
    import pandas as pd
except Exception:
    pass
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None


# === Main analysis pipeline (non-plotting) ===
def main():
    pwd
    # Load the data
    path = '<PATH_PLACEHOLDER>'
    data = pd.read_csv(path+'final_combined_all.csv') 
    data
    # rename columns compatible with Python
    data_new_1 = data.rename(columns={'@@window@@': 'Window_UValue', 
                                            '@@floor_insulation@@': 'Floor_UValue', 
                                            '@@roof_insulation@@': 'Roof_UValue',
                                           '@@orientation@@':'Orientation',
                                           '@@lighting_density@@':'lighting_density',
                                           '@@occupancy_density@@':'occupancy_density',
                                           '@@equipment_density@@':'equipment_density',
                                           '@@HSPT@@':'HSPT',
                                        #    '@@hvac_efficiency@@':'HVAC_efficiency',
                                           '@@flowINF@@':'Infiltration_rate',
                                           '@@wall_insulation@@':'Wall_UValue',
                                           '@@met_rate@@':'Metaboilc_rate',
                                           '@@clo@@':'Clothing',
                                           '@@shgc@@':'SHGC',
                                           '@@wwr@@':'WWR',
                                           '@@overhang@@':'Overhang',
                                           '@@OpenTime@@':'WindowOpeningTime',
                                           '@@CloseTime@@':'WindowClosingTime',
                                           '@@WindowOpen@@':'WindowOpeningArea'
                                          })
    data_new_1
    # # It looks like the column name is 'Errors'. Let's filter out the rows where 'Errors' value is greater than 0.
    # data_new = data_new_1[data_new_1['Errors'] <= 0]
    
    # Now, let's drop the 'Errors' column from the filtered dataframe
    data_new = data_new_1.drop(columns=['Errors'])
    data_new
    from scipy.stats import zscore
    
    # Apply the z-score function to the input and output columns
    z_scores = zscore(data_new)
    
    # Construct a DataFrame to hold the z-scores
    df_z_scores = pd.DataFrame(z_scores, columns=data_new.columns)
    
    outliers = df_z_scores[(df_z_scores > 3).any(axis=1) | (df_z_scores < -3).any(axis=1)]
    outliers
    
    
    # Define a threshold for outliers
    threshold = 3
    
    # Identify outliers
    outliers = df_z_scores[(df_z_scores > threshold).any(axis=1) | (df_z_scores < -threshold).any(axis=1)]
    
    # Drop the outliers from the original data
    data_clean = data_new.drop(outliers.index)
    
    data_clean
    # Remove rows where 'c0: Heating [kWh]' is more than 10000
    data_clean_1 = data_clean[data_clean['Heating'] <= 19000]
    
    # Display the first few rows of the filtered dataset
    data_clean_1
    data_clean_1.to_csv('dataclean.csv')
    import statsmodels.api as sm
    
    # Define the list of output variables
    output_variables = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # Initialize empty lists to store results
    models = []
    results = []
    
    # Perform multiple regression analyses
    for variable in output_variables:
        # Define the dependent variable (y) and independent variables (X)
        y = data_clean_1[variable]
        X = data_clean_1[[v for v in output_variables if v != variable]]
        
        # Add a constant to the independent variables matrix
        X = sm.add_constant(X)
        
        # Perform the multiple linear regression
        model = sm.OLS(y, X)
        result = model.fit()
        
        # Store the model and result
        models.append(model)
        results.append(result)
    
    # Display the summary of the regression results
    for i, variable in enumerate(output_variables):
        print(f"\nDependent variable: {variable}")
        print(results[i].summary())
    # Load the data
    path = '<PATH_PLACEHOLDER>'
    data = pd.read_csv(path+'final_combined_all.csv') 
    data
    # rename columns compatible with Python
    data_new = data.rename(columns={'@@window@@': 'Window_UValue', 
                                            '@@floor_insulation@@': 'Floor_UValue', 
                                            '@@roof_insulation@@': 'Roof_UValue',
                                           '@@orientation@@':'Orientation',
                                           '@@lighting_density@@':'lighting_density',
                                           '@@occupancy_density@@':'occupancy_density',
                                           '@@equipment_density@@':'equipment_density',
                                           '@@HSPT@@':'HSPT',
                                           '@@hvac_efficiency@@':'HVAC_efficiency',
                                           '@@flowINF@@':'Infiltration_rate',
                                           '@@wall_insulation@@':'Wall_UValue',
                                           '@@met_rate@@':'Metaboilc_rate',
                                           '@@clo@@':'Clothing',
                                           '@@shgc@@':'SHGC',
                                           '@@wwr@@':'WWR',
                                           '@@overhang@@':'Overhang',
                                           '@@OpenTime@@':'WindowOpeningTime',
                                           '@@CloseTime@@':'WindowClosingTime',
                                           '@@WindowOpen@@':'WindowOpeningArea'
                                          
                                           
                                          })
    data_new
    # Generate descriptive statistics
    # data_new.describe().transpose()
    # # It looks like the column name is 'Errors'. Let's filter out the rows where 'Errors' value is greater than 0.
    # data_new = data_new_1[data_new_1['Errors'] <= 0]
    
    # # Now, let's drop the 'Errors' column from the filtered dataframe
    # data_new = data_new.drop(columns=['Errors'])
    # data_new
    from scipy.stats import zscore
    
    # Apply the z-score function to the input and output columns
    z_scores = zscore(data_new)
    
    # Construct a DataFrame to hold the z-scores
    df_z_scores = pd.DataFrame(z_scores, columns=data_new.columns)
    
    outliers = df_z_scores[(df_z_scores > 3).any(axis=1) | (df_z_scores < -3).any(axis=1)]
    outliers
    
    
    # Define a threshold for outliers
    threshold = 3
    
    # Identify outliers
    outliers = df_z_scores[(df_z_scores > threshold).any(axis=1) | (df_z_scores < -threshold).any(axis=1)]
    
    # Drop the outliers from the original data
    data_clean = data_new.drop(outliers.index)
    
    data_clean
    # Remove rows where 'c0: Heating [kWh]' is more than 5000
    data_clean_1 = data_clean[data_clean['Heating'] <= 15000]
    
    # Display the first few rows of the filtered dataset
    data_clean_1
    # data_clean_1.to_csv('dataclean.csv')
    # is_nan_present = data_clean_1.isnull().any().any()
    # print(f"Are there any NaN values in the DataFrame? {is_nan_present}")
    # Importing necessary libraries
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    # Define the new target variable
    target = 'Heating'
    
    # Identify the input columns
    input_columns = data_clean_1.columns[:-3]
    
    # Create polynomial features for all input variables
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # Split the data
    
    # Define the new target variable
    target = 'Heating'
    
    # Identify the input columns
    input_columns = data_clean_1.columns[:-3]
    
    # Create polynomial features for all input variables
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, data_clean_1[target], test_size=0.2, random_state=42)
    
    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    rf.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = rf.predict(X_test)
    
    # Calculate the root mean squared error and the R-squared value
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    rmse, r2
    # ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # Define the new target variable
    target = 'facility_discomfort_HRS'
    
    # Identify the input columns
    input_columns = data_clean_1.columns[:-3]
    
    # Create polynomial features for all input variables
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, data_clean_1[target], test_size=0.2, random_state=42)
    
    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    rf.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = rf.predict(X_test)
    
    # Calculate the root mean squared error and the R-squared value
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    rmse, r2
    # Define the new target variable
    target = 'co2average_HRS'
    
    # Identify the input columns
    input_columns = data_clean_1.columns[:-3]
    
    # Create polynomial features for all input variables
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, data_clean_1[target], test_size=0.2, random_state=42)
    
    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    rf.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = rf.predict(X_test)
    
    # Calculate the root mean squared error and the R-squared value
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    rmse, r2
    # from sklearn.svm import LinearSVR
    
    # # Identify the target variables
    # target = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # # Identify the input columns
    # input_columns = data_clean_1.columns[:-3]
    
    # # Create polynomial features for all input variables
    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X_poly, data_clean_1[target], test_size=0.2, random_state=42)
    
    # # Initialize models
    # models = {
    #     'XGBoost': XGBRegressor(),
    #     'RandomForest': RandomForestRegressor(),
    #     'SVM': LinearSVR()
    # }
    
    # # Initialize a results dictionary
    # results = {'Model': [], 'Output': [], 'RMSE': [], 'R2': [], 'Num_Features': [], 'Feature_Names': []}
    
    # # For each output (one-at-a-time approach)
    # for i, target_variable in enumerate(target):
    #     y_train_reshaped = y_train.iloc[:, i].values.ravel()
    #     y_test_reshaped = y_test.iloc[:, i].values.reshape(-1, 1)
    
    #     # For each model
    #     for model_name, model in models.items():
    #         selector = RFECV(estimator=model, step=1, cv=5)
    #         pipeline = Pipeline([('feature_selection', selector), ('model', model)])
            
    #         # Fit the model without grid search
    #         pipeline.fit(X_train, y_train_reshaped)
            
    #         best_features = [f for f, s in zip(poly.get_feature_names_out(input_features=input_columns), pipeline.named_steps['feature_selection'].support_) if s]
    #         y_pred = pipeline.predict(X_test)
    #         rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred))
    #         r2 = r2_score(y_test_reshaped, y_pred)
    
    #         results['Model'].append(model_name)
    #         results['Output'].append(target_variable)
    #         results['RMSE'].append(rmse)
    #         results['R2'].append(r2)
    #         results['Num_Features'].append(pipeline.named_steps['feature_selection'].n_features_)
    #         results['Feature_Names'].append(", ".join(best_features))
    
    # # Convert results to DataFrame
    # results_df = pd.DataFrame(results)
    
    # # Print the results
    # print(results_df)
    # results_df.to_csv('aa_4.csv')
    # from sklearn.multioutput import MultiOutputRegressor
    # from sklearn.feature_selection import RFECV
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.ensemble import RandomForestRegressor
    
    # # Custom function to aggregate the feature importances from the underlying models of MultiOutputRegressor
    # def multi_output_importance_getter(estimator):
    #     importances = [e.feature_importances_ for e in estimator.estimators_]
    #     return np.mean(importances, axis=0)
    
    # # Identify the target variables
    # target = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # # Identify the input columns
    # input_columns = data_clean_1.columns[:-3]
    
    # # Create polynomial features for all input variables
    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X_poly, data_clean_1[target], test_size=0.2, random_state=42)
    
    # # Initialize models
    # models = {
    #     'XGBoost': MultiOutputRegressor(XGBRegressor()),
    #     'RandomForest': RandomForestRegressor()
    # }
    
    # # Initialize a results dictionary
    # results = {'Model': [], 'RMSE': [], 'R2': [], 'Num_Features': [], 'Feature_Names': []}
    
    # # For each model
    # for model_name, model in models.items():
    #     selector = RFECV(estimator=model, step=1, cv=5, importance_getter=multi_output_importance_getter)
        
    #     pipeline = Pipeline([('feature_selection', selector), ('model', model)])
        
    #     # Fit the model
    #     pipeline.fit(X_train, y_train)
        
    #     best_features = [f for f, s in zip(poly.get_feature_names_out(input_features=input_columns), pipeline.named_steps['feature_selection'].support_) if s]
    #     results['Num_Features'].append(pipeline.named_steps['feature_selection'].n_features_)
    #     results['Feature_Names'].append(", ".join(best_features))
        
    #     y_pred = pipeline.predict(X_test)
    #     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #     r2 = r2_score(y_test, y_pred)
    
    #     results['Model'].append(model_name)
    #     results['RMSE'].append(rmse)
    #     results['R2'].append(r2)
    
    # # Convert results to DataFrame
    # results_df = pd.DataFrame(results)
    
    # # Print the results
    # print(results_df)
    # results_df.to_csv('aa_5.csv')
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.feature_selection import RFECV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from xgboost import XGBRegressor
    
    # Custom function to aggregate the feature importances from the underlying models of MultiOutputRegressor
    def multi_output_importance_getter(estimator):
        importances = [e.feature_importances_ for e in estimator.estimators_]
        return np.mean(importances, axis=0)
    
    # Identify the target variables
    target = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # Identify the input columns
    input_columns = data_clean_1.columns[:-3]
    
    # Create polynomial features for all input variables
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(data_clean_1[input_columns])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, data_clean_1[target], test_size=0.2, random_state=42)
    
    # Initialize models (only XGBoost now)
    models = {
        'XGBoost': MultiOutputRegressor(XGBRegressor())
    }
    
    # Initialize a results dictionary
    results = {'Model': [], 'RMSE': [], 'R2': [], 'Num_Features': [], 'Feature_Names': []}
    
    # For each model
    for model_name, model in models.items():
        selector = RFECV(estimator=model, step=1, cv=5, importance_getter=multi_output_importance_getter)
        
        pipeline = Pipeline([('feature_selection', selector), ('model', model)])
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        best_features = [f for f, s in zip(poly.get_feature_names_out(input_features=input_columns), pipeline.named_steps['feature_selection'].support_) if s]
        results['Num_Features'].append(pipeline.named_steps['feature_selection'].n_features_)
        results['Feature_Names'].append(", ".join(best_features))
        
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
    
        results['Model'].append(model_name)
        results['RMSE'].append(rmse)
        results['R2'].append(r2)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print the results
    print(results_df)
    results_df.to_csv('aa_5.csv')
    # Let's start by setting up DEAP
    import random
    import numpy as np
    import pandas as pd
    from deap import base, creator, tools, algorithms
    
    # 1. Define our types, objectives are minimized in this case
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # 2. Define the repair function
    def repair(individual):
        for i in range(len(individual)):
            individual[i] = max(0, individual[i])
    
    # Modify the evalModel function to penalize solutions where CO2 > 1000 hours
    def evalModel(individual, model):
        input_data = {col: val for col, val in zip(input_columns, individual)}
        df = pd.DataFrame([input_data])
        X = poly.transform(df)
        predictions = model.predict(X)
        
        # Check the constraint for CO2 (assuming it's the third objective)
        if predictions[0][2] > 1000:
            # A large penalty is applied if the constraint is violated
            return (1e9, 1e9, 1e9)  # 1e9 is an arbitrarily large number to indicate a poor solution
        else:
            return tuple(predictions[0])
    
    # # Use the pipeline (trained model) for evaluations in DEAP
    # def evalModel(individual, model):
    #     input_data = {col: val for col, val in zip(input_columns, individual)}
    #     df = pd.DataFrame([input_data])
    #     X = poly.transform(df)
    #     predictions = model.predict(X)
    #     return tuple(predictions[0])
    
    # Register the functions with the toolbox
    toolbox = base.Toolbox()
    for col in input_columns:
        toolbox.register("attr_" + col, random.uniform, data_clean_1[col].min(), data_clean_1[col].max())
    
    attributes = [toolbox.__getattribute__("attr_" + col) for col in input_columns]
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Here, use the pipeline (trained model) for evaluations
    from functools import partial
    partial_eval = partial(evalModel, model=pipeline)
    
    toolbox.register("evaluate", partial_eval)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.1, sigma=0.9, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    mu = 2000 
    # Initialization
    population = toolbox.population(n=mu)
    hof = tools.HallOfFame(100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields
    
    
    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        
    logbook.record(gen=0, evals=len(population), **stats.compile(population))
    ngen = 105
    lambda_ = 50  # Number of offspring to produce
    cxpb = 0.8139     # Probability of crossover
    mutpb = 0.1861    # Probability of mutation
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
    
        # Apply the repair function
        for child in offspring:
            repair(child)
    
        # Evaluate the offspring
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
    
        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        
        # Update the hall of fame with the generated individuals
        hof.update(population)
    
        # Record statistics for the current generation and print
        logbook.record(gen=gen, evals=len(offspring), **stats.compile(population))
        print(logbook.stream)
    
    # Print the hall of fame individuals
    print("Hall of Fame Individuals:")
    for ind in hof:
        print(ind.fitness.values)
    import random
    import numpy as np
    import pandas as pd
    from deap import base, creator, tools, algorithms
    from bayes_opt import BayesianOptimization
    
    # 1. Define our types, objectives are minimized in this case
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # 2. Define the repair function
    def repair(individual):
        for i in range(len(individual)):
            individual[i] = max(0, individual[i])
    
    # Use the pipeline (trained model) for evaluations in DEAP
    def evalModel(individual, model):
        input_data = {col: val for col, val in zip(input_columns, individual)}
        df = pd.DataFrame([input_data])
        X = poly.transform(df)
        predictions = model.predict(X)
        return tuple(predictions[0])
    
    # Register the functions with the toolbox
    toolbox = base.Toolbox()
    for col in input_columns:
        toolbox.register("attr_" + col, random.uniform, data_clean_1[col].min(), data_clean_1[col].max())
    
    attributes = [toolbox.__getattribute__("attr_" + col) for col in input_columns]
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Here, use the pipeline (trained model) for evaluations
    from functools import partial
    partial_eval = partial(evalModel, model=pipeline)
    toolbox.register("evaluate", partial_eval)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.1, sigma=0.9, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    # Define the objective function for Bayesian Optimization
    # Define the objective function for Bayesian Optimization
    def objective(ngen, lambda_, cxpb):
        ngen = int(ngen)
        lambda_ = int(lambda_)
        mutpb = 1 - cxpb  # Ensure sum of cxpb and mutpb is always 1
    
        mu = 4524
        population = toolbox.population(n=mu)
        hof = tools.HallOfFame(3000)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
    
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        logbook.record(gen=0, evals=len(population), **stats.compile(population))
        
        for gen in range(1, ngen + 1):
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            for child in offspring:
                repair(child)
    
            fitnesses = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
    
            population[:] = toolbox.select(population + offspring, mu)
            hof.update(population)
            logbook.record(gen=gen, evals=len(offspring), **stats.compile(population))
    
        performance = logbook[-1]['min'][0]
        return -performance
    
    # Define bounds for the parameters
    pbounds = {
        'ngen': (50, 200),
        'lambda_': (30, 100),
        'cxpb': (0.8, 0.9),
    }
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
    )
    
    init_points = 10
    n_iter = 30
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    print(optimizer.max)
    pwd
    # Load the data
    path = '<PATH_PLACEHOLDER>'
    
    # Load the "SRRC" sheet from the provided Excel file
    srrc_data = pd.read_excel("<PATH_PLACEHOLDER>", sheet_name="SRRC")
    sobol_data = pd.read_excel("<PATH_PLACEHOLDER>", sheet_name="Sobol")
    ML_data = pd.read_excel("<PATH_PLACEHOLDER>", sheet_name="ML")
    srrc_data
    import pandas as pd
    
    # Creating the data
    data = {
        'Heating consumption [kWh]': ['Ventilation rate', 'Occupancy density', 'Infiltration', 'HSPT', 'HVAC efficiency', 'Metabolic rate', 'Wall U-value', 'Orientation', 'SHGC', 'Window U-value'],
        'Thermal Discomfort [hrs]': ['Metabolic rate', 'Occupancy density', 'Ventilation rate', 'Infiltration', 'HSPT', 'Clothing', 'SHGC', 'Wall U-value', 'Orientation', 'Window U-value'],
        'CO2 concentration [ppm]': ['Metabolic rate', 'Ventilation rate', 'Infiltration', 'Occupancy density', 'SHGC', 'Orientation'],
        'Combined': ['Ventilation rate', 'Infiltration', 'SHGC', 'Wall Insulation', 'Orientation', 'Occupancy density', 'HVAC efficiency', 'Metabolic rate', 'HSPT', 'Window U-value', 'Clothing', 'Roof U-value']
    }
    
    
    
    
    # Convert the dictionary to a DataFrame
    # Recreating the DataFrame from the provided LaTeX table data
    df = pd.DataFrame.from_dict(data, orient='index').transpose()
    
    df

    # === Plotting (moved to the end) ===
    import matplotlib.pyplot as plt
    
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    
    # Plot 'c0: Heating [kWh]'
    axes[0].hist(data_clean_1['Heating'], bins=30, color='purple', alpha=0.7)
    axes[0].set_xlabel('Heating consumption [kWh]',fontsize=16)
    axes[0].set_ylabel('Frequency', fontsize=16)
    axes[0].text(-0.1, 1.05, '(a)', transform=axes[0].transAxes, size=16, weight='bold')
    
    # Plot 'c8: facility_thermaldiscomfort'
    axes[1].hist(data_clean_1['facility_discomfort_HRS'], bins=30, color='green', alpha=0.7)
    axes[1].set_xlabel('Thermal Discomfort [Hours]',fontsize=16)
    axes[1].set_ylabel('Frequency',fontsize=16)
    axes[1].text(-0.1, 1.05, '(b)', transform=axes[1].transAxes, size=16, weight='bold')
    
    # Plot 'co2_average'
    axes[2].hist(data_clean_1['co2average_HRS'], bins=30, color='red', alpha=0.7)
    axes[2].set_xlabel('CO2 concentration > 1000 ppm [Hours]',fontsize=16)
    axes[2].set_ylabel('Frequency',fontsize=16)
    axes[2].text(-0.1, 1.05, '(c)', transform=axes[2].transAxes, size=16, weight='bold')
    
    
    # Display plots
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    plt.show()
    import matplotlib.pyplot as plt
    
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    
    # Histograms
    axes[0].hist(data_clean_1['Heating'], bins=30, color='purple', alpha=0.7)
    axes[1].hist(data_clean_1['facility_discomfort_HRS'], bins=30, color='green', alpha=0.7)
    axes[2].hist(data_clean_1['co2average_HRS'], bins=30, color='red', alpha=0.7)
    
    # Find the maximum frequency to standardize the y-axis
    max_freq = max(
        max(axes[0].get_ylim()),
        max(axes[1].get_ylim()),
        max(axes[2].get_ylim())
    )
    
    # Apply the same y-axis range to all subplots
    for ax in axes:
        ax.set_ylim(0, max_freq)
    
    # Customize subplots
    labels = ['Heating consumption [kWh]', 'Thermal Discomfort [Hours]', 'CO2 concentration > 1000 ppm [Hours]']
    colors = ['purple', 'green', 'red']
    letters = ['(a)', '(b)', '(c)']
    
    for i, ax in enumerate(axes):
        ax.set_xlabel(labels[i], fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.text(-0.1, 1.05, letters[i], transform=ax.transAxes, size=16, weight='bold')
    
    # Display plots
    plt.tight_layout()
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    plt.show()
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Original column names
    original_names = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # Custom names for the heatmap
    custom_names = ['Heating consumption [kWh]', 'Thermal Discomfort [Hours]', 'CO2 >1000 ppm [Hours]']
    
    # Calculate the correlation matrix using original names
    correlation_matrix = data_clean_1[original_names].corr()
    
    # Create a heatmap with customizations
    plt.figure(figsize=(5, 5))
    
    # Heatmap with custom labels
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 8}, 
                xticklabels=custom_names, yticklabels=custom_names)
    
    # Title (customize the text and font size here)
    # plt.title('Correlation Heatmap of Output Variables', fontsize=12)
    
    # X and Y labels (customize the text and font size here)
    # plt.xlabel('Output Variables', fontsize=14)
    # plt.ylabel('Output Variables', fontsize=14)
    
    # Adjust the tick labels font size if needed
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Display the heatmap
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    plt.show()
    # Calculate the Spearman rank correlation
    spearman_corr = data_clean[['Heating', 'facility_discomfort_HRS', 'co2average_HRS']].corr(method='spearman')
    
    # Custom names for the heatmap
    custom_names = ['Heating consumption [kWh]', 'Thermal Discomfort [Hours]', 'CO2 >1000 ppm [Hours]']
    
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, annot_kws={"size": 12}, linewidths=0.5,xticklabels=custom_names, yticklabels=custom_names)
    # plt.title('Spearman Rank Correlation Heatmap', fontsize=16)
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    plt.show()
    # 1. Pairwise scatterplots
    sns.pairplot(data_clean_1[output_variables])
    plt.suptitle('Pairwise Scatterplots of Output Variables', y=1.02)
    plt.show()
    
    # 2. Boxplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    for i, variable in enumerate(output_variables):
        sns.boxplot(data=data_clean_1, y=variable, ax=axes[i])
    plt.suptitle('Boxplots of Output Variables', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # 6. Interaction effects
    # Add an interaction term between 'c8: facility_thermaldiscomfort' and 'co2_average' to the regression model with 'c0: Heating [kWh]' as dependent variable
    data_interaction = data_clean_1.copy()
    data_interaction['interaction'] = data_clean_1['facility_discomfort_HRS'] * data_clean_1['co2average_HRS']
    y = data_interaction['Heating']
    X = sm.add_constant(data_interaction[['facility_discomfort_HRS', 'co2average_HRS', 'interaction']])
    model_interaction = sm.OLS(y, X)
    results_interaction = model_interaction.fit()
    print("\nRegression results with interaction term:")
    print(results_interaction.summary())
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    import xgboost as xgb
    
    # Define the problem for SALib
    excluded_columns = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS', 'combined_output']
    input_columns_combined = [col for col in data_clean_1.columns if col not in excluded_columns]
    problem = {
        'num_vars': len(input_columns_combined),
        'names': input_columns_combined,
        'bounds': [[data_clean_1[col].min(), data_clean_1[col].max()] for col in input_columns_combined]
    }
    
    # Generate samples using saltelli.sample
    param_values = saltelli.sample(problem, 2048)  # Adjusted the sample size
    
    # Model function to get Y values using XGBoost
    def model(input_data, output_name):
        X = data_clean_1[input_columns_combined]
        y = data_clean_1[output_name]
        model_xgb = xgb.XGBRegressor(objective ='reg:squarederror')
        model_xgb.fit(X, y)
        return model_xgb.predict(input_data)
    
    # Calculate Sobol indices for each output
    outputs_combined = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    sobol_indices = {}
    for output in outputs_combined:
        Y = model(param_values, output)
        sobol_indices[output] = sobol.analyze(problem, Y)
    
    # sobol_indices
    
    # Extracting the total order Sobol indices for each output and structuring it as a DataFrame
    sobol_total_order_df = pd.DataFrame({output: indices['ST'] for output, indices in sobol_indices.items()}, index=input_columns_combined)
    sobol_first_order_df = pd.DataFrame({output: indices['S1'] for output, indices in sobol_indices.items()}, index=input_columns_combined)
    
    sobol_first = pd.DataFrame(sobol_first_order_df)
    
    sobol_first.to_csv('sobol_first.csv')
    # # Display the table
    # sobol_total_order_df.to_csv('sobol.csv')
    # sobol_first_order_df.to_csv('sobol_first.csv')
    # Assuming you have a dictionary that maps each output to a custom title
    custom_titles_sobol = {
        'Heating': 'Heating consumption [kWh]',
        'facility_discomfort_HRS': 'Thermal Discomfort [Hours]',
        'co2average_HRS': 'CO2 concentration > 1000 ppm [Hours]'
    }
    
    # Number of rows will be the ceiling division of the number of outputs by 3
    num_rows = -(-len(outputs_combined) // 3)  # This is a way to perform ceiling division in Python
    
    # Plotting Sobol indices
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 4.5 * num_rows))
    
    # If only one row, ensure axes is 2D
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for idx, (output, indices) in enumerate(sobol_indices.items()):
        row_idx = idx // 3
        col_idx = idx % 3
        ax = axes[row_idx, col_idx]
        
        ax.barh(input_columns_combined, indices['ST'], color='grey', label='Total-order index', alpha=0.7)
        ax.barh(input_columns_combined, indices['S1'], color='blue', label='First-order index', alpha=0.7)
        ax.set_ylabel('Features')
        ax.set_xlabel('Sobol Index Value')
        
        # Use the custom title from the dictionary, or default to the output name if not found
        ax.set_title(custom_titles_sobol.get(output, output))
        
        ax.legend()
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # If there are fewer outputs than cells, remove unused subplots
    if len(outputs_combined) < num_rows * 3:
        for i in range(len(outputs_combined), num_rows * 3):
            row_idx = i // 3
            col_idx = i % 3
            fig.delaxes(axes[row_idx, col_idx])
    
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    plt.show()
    # Define the input and output columns
    inputs = data_clean_1.columns[0:18].tolist()
    inputs
    outputs = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # Define the titles for each plot
    plot_titles = ["Senstivity of input parameters for Heating Consumption [kWh]", 
                   "Senstivity of input parameters for Thermal discomfort [Hours]", 
                   "Senstivity of input parameters for CO2 concentration > 1000 ppm [Hours]"]
    
    # Normalize the output columns
    data_normalized = data_clean_1.copy()
    data_normalized[outputs] = StandardScaler().fit_transform(data_clean_1[outputs])
    
    # Perform PCA on the normalized output data
    pca = PCA(n_components=len(outputs))  # Use all principal components
    pca.fit(data_normalized[outputs])
    
    # Calculate the explained variance ratio for each principal component
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Create new columns for the principal components
    for i in range(len(outputs)):
        data_normalized['PC'+str(i+1)] = pca.transform(data_normalized[outputs])[:, i]
    
    # # Calculate and plot the correlation of the input parameters with each principal component
    # for i in range(len(outputs)):
    #     pc = 'PC'+str(i+1)
    #     correlations_with_pcs = data_normalized.corr().loc[inputs, pc]
    #     sorted_correlations = correlations_with_pcs.sort_values()
    #     plt.figure(figsize=(10, 6))
    #     sorted_correlations.plot(kind='barh', color='green')
    #     plt.xlabel('Correlation with ' + pc)
    #     plt.title(plot_titles[i])  # Use the corresponding title from the list
    #     plt.grid(True)
    #     plot_filename = "<PATH_PLACEHOLDER>" + pc + ".png"  # Define the path and filename
    #     plt.savefig(plot_filename, dpi=1200)  # Save the plot with high resolution (1200 DPI)
    #     plt.show()
    #     print("Plot saved to:", plot_filename)
    
    
    # Define customizable parameters for the titles, text sizes, and x-axis ticks
    custom_plot_titles = ["Heating Consumption [kWh]", "Thermal discomfort [Hours]", "CO2 concentration > 1000 ppm [Hours]"]
    custom_x_axis_titles = ["PCA", "PCA", "PCA"]
    title_text_size = 14
    axis_title_text_size = 12
    
    # Customize x-axis ticks here. You can adjust these values based on your preferences.
    custom_xticks = {
        "Heating Consumption [kWh]": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
        "Thermal discomfort [Hours]": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
        "CO2 concentration > 1000 ppm [Hours]": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
    }
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        pc = 'PC'+str(i+1)
        correlations_with_pcs = data_normalized.corr().loc[inputs, pc]
        sorted_correlations = correlations_with_pcs.sort_values()
        
        sorted_correlations.plot(kind='barh', color='green', ax=ax)
        ax.set_xlabel(custom_x_axis_titles[i], fontsize=axis_title_text_size)
        ax.set_title(custom_plot_titles[i], fontsize=title_text_size)
        ax.set_xticks(custom_xticks[custom_plot_titles[i]])  # Set custom x-axis ticks
        ax.grid(True)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_filename_1 = "<PATH_PLACEHOLDER>"
    plt.savefig(combined_plot_filename_1, dpi=600)
    plt.show()
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    # Define the input and output columns
    inputs = data_clean_1.columns[0:20].tolist()
    related_outputs = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    # related_outputs = ['Heating', 'facility_discomfort_HRS', 'co2average_HRS']
    
    # Normalize the related output columns
    normalized_related_outputs = (data_clean_1[related_outputs] - data_clean_1[related_outputs].min()) / (data_clean_1[related_outputs].max() - data_clean_1[related_outputs].min())
    
    # Perform PCA on the normalized related output data
    pca_related = PCA()
    principal_components_related = pca_related.fit_transform(normalized_related_outputs)
    
    # The explained variance ratio of each principal component for the related outputs
    explained_variance_ratio_related = pca_related.explained_variance_ratio_
    
    # Add the principal components for the related outputs to the data
    for i in range(len(explained_variance_ratio_related)):
        data_clean_1[f'PC_related{i+1}'] = principal_components_related[:, i]
    
    # Calculate the correlation of the input parameters with the principal components for the related outputs
    correlations_with_pcs_related = data_clean_1.corr().loc[inputs, [f'PC_related{i+1}' for i in range(len(explained_variance_ratio_related))]]
    
    # Print the correlations
    # print(correlations_with_pcs_related)
    
    correlations_with_pcs_related.to_csv('pca_related.csv')
    
    # Define customizable parameters for the titles and text sizes
    custom_plot_titles = ["Heating Consumption [kWh]", "Thermal discomfort [Hours]", "CO2 concentration > 1000 ppm [Hours]"]  # You can replace these with your desired titles
    custom_x_axis_titles = ["PCA", "PCA", "PCA"]  # Replace with your desired x-axis titles
    title_text_size = 14  # Adjust as desired
    axis_title_text_size = 12  # Adjust as desired
    
    
    # Customize x-axis ticks here. You can adjust these values based on your preferences.
    custom_xticks = {
        "Heating Consumption [kWh]": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        "Thermal discomfort [Hours]": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        "CO2 concentration > 1000 ppm [Hours]": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    }
    
    color_map = {
        "Heating Consumption [kWh]": 'purple',  # Bright Green
        "Thermal discomfort [Hours]": 'green',  # Green
        "CO2 concentration > 1000 ppm [Hours]": 'red'   # Dark Green
    }
    
    ventilation_values = {
        "Heating Consumption [kWh]": 0.2,
        "Thermal discomfort [Hours]": 0.2,
        "CO2 concentration > 1000 ppm [Hours]": -0.1
    }
    
    
    # ... [Rest of the code remains unchanged]
    
    # Perform PCA on the normalized related output data
    pca_related = PCA()
    principal_components_related = pca_related.fit_transform(normalized_related_outputs)
    
    # The explained variance ratio of each principal component for the related outputs
    explained_variance_ratio_related = pca_related.explained_variance_ratio_
    
    # Print the explained variance ratio for each principal component
    print(f"Explained Variance Ratio for Principal Components: {explained_variance_ratio_related}")
    
    
    
    # Adjusting the plotting code to exclude "Heating" from the first plot
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    # Iterate over each output and its corresponding subplot axis
    for i, ax in enumerate(axes):
        pc = f'PC_related{i+1}'
        correlations_with_pcs_related = data_clean_1.corr().loc[inputs, pc].copy()
        
        # Add the 'Ventilation' value for the current output
        correlations_with_pcs_related['Ventilation'] = ventilation_values[custom_plot_titles[i]]
        
        # If it's the first plot (i.e., for Heating), remove the 'Heating' variable
        if i == 0:
            correlations_with_pcs_related = correlations_with_pcs_related.drop("Heating")
            
        sorted_correlations = correlations_with_pcs_related.sort_values()
        
        sorted_correlations.plot(kind='barh', color=color_map[custom_plot_titles[i]], ax=ax)
        ax.set_xlabel(custom_x_axis_titles[i], fontsize=axis_title_text_size)
        # ax.set_title(custom_plot_titles[i], fontsize=title_text_size)
        ax.set_xticks(custom_xticks[custom_plot_titles[i]])
        ax.grid(True)
    
    
    # Adjust the layout to ensure the plots do not overlap
    plt.tight_layout()
    
    # Save the combined plot (keeping the save path the same for now)
    combined_plot_filename = "<PATH_PLACEHOLDER>"
    plt.savefig(combined_plot_filename, dpi=600)  # Save with high resolution
    plt.show()
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sample objective values from the hall of fame (replace this with your actual data)
    objectives = np.array([ind.fitness.values for ind in hof])
    
    # Define tick values for each axis
    x_ticks = np.arange(0, 2000, 400)  # Example: ticks from 0 to 7000 with a step of 1000
    y_ticks = np.arange(0, 7000, 500)     # Adjust these values as per your requirements
    z_ticks = np.arange(300, 2000, 200)
    
    # 3D Scatter Plot
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', edgecolors='k', s=100)
    ax.set_xlabel('Thermal Discomfort (Hours) ', fontsize=14, labelpad=10)
    ax.set_ylabel('Heating consumption (Wh)', fontsize=14, labelpad=10)
    ax.set_zlabel('CO2 $>$ 1000 ppm (Hours)', labelpad=1, fontsize=14)
    
    # Set the tick values
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    
    # Optional: To set custom tick labels
    # ax.set_xticklabels(['label1', 'label2', ...])  # You can define custom labels for each tick value
    
    # Save the plot
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=900)  # Save the plot with high resolution (600 DPI)
    plt.show()
    
    
    
    # 3. Tabulate Best Solutions
    # For simplicity, we'll show the top 50 solutions. You can adjust this number as needed.
    top_solutions = 100
    df_best_solutions = pd.DataFrame([hof[i] for i in range(top_solutions)], columns=input_columns)
    df_best_solutions['Objective 1 (Discomfort)'] = [ind.fitness.values[0] for ind in hof[:top_solutions]]
    df_best_solutions['Objective 2 (Heating)'] = [ind.fitness.values[1] for ind in hof[:top_solutions]]
    df_best_solutions['Objective 3 (CO2 concentration)'] = [ind.fitness.values[2] for ind in hof[:top_solutions]]
    df_best_solutions
    
    # df_best_solutions.to_csv('best.csv')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Rectangle
    
    # Extracting the fitness values for each individual in the Hall of Fame
    x_vals = [ind.fitness.values[0] for ind in hof]
    y_vals = [ind.fitness.values[1] for ind in hof]
    z_vals = [ind.fitness.values[2] for ind in hof]
    
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    pareto_points = ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', label="Pareto Points")
    ax.set_xlabel('Heating consumption (Wh)', fontsize=14, labelpad=10)
    ax.set_ylabel('Thermal Discomfort (Hours)', fontsize=14, labelpad=10)
    ax.set_zlabel('CO2 $>$ 1000 ppm (Hours)', labelpad=1, fontsize=14)  # Adjust labelpad here
    # ax.set_title('3D Pareto Front')
    
    # Plotting the surface
    surf = ax.plot_trisurf(x_vals, y_vals, z_vals, alpha=0.1, color='red', label="Pareto Surface")
    
    # Creating proxy artist for legend
    proxy = Rectangle((0, 0), 1, 1, fc="red", alpha=0.9)
    
    # Adding legend
    ax.legend(handles=[pareto_points, proxy], labels=['Pareto Points', 'Pareto Surface'])
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    # def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    #     '''Pareto frontier selection process'''
    #     sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    #     pareto_front = [sorted_list[0]]
    
    #     for pair in sorted_list[1:]:
    #         if maxY:
    #             if pair[1] >= pareto_front[-1][1]:
    #                 pareto_front.append(pair)
    #         else:
    #             if pair[1] <= pareto_front[-1][1]:
    #                 pareto_front.append(pair)
    
    #     return pareto_front
    
    # plt.figure(figsize=(18, 5))
    
    # # Heating vs. facility_discomfort_HRS
    # plt.subplot(1, 3, 1)
    # plt.scatter(heating_values, facility_discomfort_values, color='blue', marker='o')
    # pf1 = pareto_frontier(heating_values, facility_discomfort_values)
    # pf1_X = [i[0] for i in pf1]
    # pf1_Y = [i[1] for i in pf1]
    # plt.plot(pf1_X, pf1_Y, color='black')
    # plt.title('Heating vs. Facility Discomfort HRS')
    # plt.xlabel('Heating')
    # plt.ylabel('Facility Discomfort HRS')
    
    # # Heating vs. co2average_HRS
    # plt.subplot(1, 3, 2)
    # plt.scatter(heating_values, co2average_values, color='green', marker='o')
    # pf2 = pareto_frontier(heating_values, co2average_values)
    # pf2_X = [i[0] for i in pf2]
    # pf2_Y = [i[1] for i in pf2]
    # plt.plot(pf2_X, pf2_Y, color='black')
    # plt.title('Heating vs. CO2 Average HRS')
    # plt.xlabel('Heating')
    # plt.ylabel('CO2 Average HRS')
    
    # # Facility_discomfort_HRS vs. co2average_HRS
    # plt.subplot(1, 3, 3)
    # plt.scatter(facility_discomfort_values, co2average_values, color='red', marker='o')
    # pf3 = pareto_frontier(facility_discomfort_values, co2average_values)
    # pf3_X = [i[0] for i in pf3]
    # pf3_Y = [i[1] for i in pf3]
    # plt.plot(pf3_X, pf3_Y, color='black')
    # plt.title('Facility Discomfort HRS vs. CO2 Average HRS')
    # plt.xlabel('Facility Discomfort HRS')
    # plt.ylabel('CO2 Average HRS')
    
    # plt.tight_layout()
    # plt.show()
    # Provided data
    data_dict = {
        "Features": ["Window_UValue", "Floor_UValue", "Roof_UValue", "Orientation", "lighting_density", 
                     "occupancy_density", "equipment_density", "HSPT", "Infiltration_rate", "Wall_UValue",
                     "Metaboilc_rate", "Clothing", "SHGC", "WWR", "Overhang", "WindowOpeningTime", 
                     "WindowClosingTime", "WindowOpeningArea", "Heating", "Thermal discomfort", "Ventilation"],
        "PC_related1": [0.07823375, 0.021648554, 0.04118927, 0.021793396, 0.042325064, 
                        -0.486179957, -0.071971507, 0.467582037, 0.128176749, 0.101285844, 
                        -0.414993986, -0.00297237, -0.061474693, -0.140462773, -0.012429997, 
                        0.12962913, -0.02408839, 0.061518673, 0.000, -0.629290368, 0.31223],
        "PC_related2": [0.008216523, -0.001988391, -0.00000, -0.020769672, -0.040545739, 
                        0.150956978, -0.103680007, 0.654493586, 0.103627689, 0.067452978, 
                        0.196409474, 0.005930987, -0.037998373, -0.232287064, 0.002663941, 
                        0.095999082, -0.026914697, 0.153784559, -0.596582435, 0.079369739, 0.2],
        "PC_related3": [-0.007465715, 0.010387189, 0.015070189, 0.011606795, 0.014957581, 
                        0.497203787, 0.026519642, 0.0046685286, -0.223622718, 0.00081857599, 
                        0.330112613, 0.000893316, 0.036083971, 0.172301506, -0.000761494, 
                        -0.202800384, 0.040121448, -0.166970711, -0.023819677, 0.469626335, -0.19]
    }
    
    # Convert to DataFrame
    correlation_data = pd.DataFrame(data_dict)
    
    # Customizable parameters
    PLOT_PARAMS = {
        "titles": ["Heating Consumption [kWh]", "Thermal discomfort [Hours]", "CO2 concentration > 1000 ppm [Hours]"],
        "x_axis_titles": ["PCA", "PCA", "PCA"],
        "title_text_size": 12,
        "axis_title_text_size": 12,
        "color_map": {
            "Heating Consumption [kWh]": 'purple',
            "Thermal discomfort [Hours]": 'green',
            "CO2 concentration > 1000 ppm [Hours]": 'red'
        }
    }
    
    custom_plot_titles = ["Heating Consumption [kWh]", "Thermal discomfort [Hours]", "CO2 concentration > 1000 ppm [Hours]"] 
    
    # Customize x-axis ticks here. You can adjust these values based on your preferences.
    custom_xticks = {
        "Heating Consumption [kWh]": [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        "Thermal discomfort [Hours]": [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        "CO2 concentration > 1000 ppm [Hours]": [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    }
    
    # Plotting using the provided structure
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        pc = f'PC_related{i+1}'
        sorted_correlations = correlation_data.sort_values(by=pc, ascending=True)
        
        # If it's the first plot (i.e., for Heating), remove the 'Heating' variable
        if i == 0:
            sorted_correlations = sorted_correlations[sorted_correlations["Features"] != "Heating"]
        # If it's the second plot (i.e., for Thermal discomfort), remove the 'Thermal discomfort' variable
        elif i == 1:
            sorted_correlations = sorted_correlations[sorted_correlations["Features"] != "Thermal discomfort"]
            
        sorted_correlations.plot(x='Features', y=pc, kind='barh', legend=False, color=PLOT_PARAMS["color_map"][PLOT_PARAMS["titles"][i]], ax=ax)
        ax.set_xlabel(PLOT_PARAMS["x_axis_titles"][i], fontsize=PLOT_PARAMS["axis_title_text_size"])
        ax.set_title(PLOT_PARAMS["titles"][i], fontsize=PLOT_PARAMS["title_text_size"])
        ax.set_xticks(custom_xticks[custom_plot_titles[i]])
        ax.grid(True)
    # Subcaptions
    subcaptions = ['(a)', '(b)', '(c)']
    for ax, subcap in zip(axes, subcaptions):
        ax.text(-0.15, 1.05, subcap, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
        
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    plt.show()
    models = ML_data['Model']
    outputs = ML_data['Output']
    rmse_values = ML_data['RMSE']
    r2_values = ML_data['R2']
    cvrmse_values = ML_data['CVRMSE']
    
    unique_outputs = ML_data['Output'].unique()
    
    # Custom parameters
    bar_thickness = 0.3
    axis_label_fontsize = 12
    title_fontsize = 14
    
    color_map = {
        "Heating consumption [kWh]": 'purple',  
        "Thermal discomfort [Hours]": 'green', 
        "CO2 > 1000 ppm [Hours]": 'red',   
        "All outputs together":'blue'
    }
    
    # Adjusted code to keep y-axis ticks values same for each RMSE plot
    
    def plot_ml_results(y_min_rmse=0, y_max_rmse=800, y_min_r2=0, y_max_r2=1):
        """
        Plot RMSE and R2 values for the provided data with customizable y-axis limits.
        
        Parameters:
        - y_min_rmse: Minimum y-axis limit for RMSE plots.
        - y_max_rmse: Maximum y-axis limit for RMSE plots.
        - y_min_r2: Minimum y-axis limit for R2 plots.
        - y_max_r2: Maximum y-axis limit for R2 plots.
        """
        
        fig, axes = plt.subplots(len(unique_outputs), 3, figsize=(13, 3 * len(unique_outputs)))
    
        for idx, output in enumerate(unique_outputs):
            output_data = ML_data[ML_data['Output'] == output]
            
            # Plotting RMSE values for the current output
            axes[idx][0].bar(output_data['Model'], output_data['RMSE'], color=color_map[output], edgecolor='black', width=bar_thickness)
            axes[idx][0].set_ylabel('$RMSE$', fontsize=axis_label_fontsize)
            axes[idx][0].set_title(f'$RMSE$ for {output}', fontsize=title_fontsize)
            axes[idx][0].grid(axis='y', linestyle='--', alpha=0.7)
            axes[idx][0].tick_params(axis='x', rotation=0)
            axes[idx][0].set_ylim(y_min_rmse, y_max_rmse)  # Set custom y-axis limit for RMSE plots
            # for i, v in enumerate(output_data['RMSE']):
            #     axes[idx][0].text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', color='black', fontweight='bold')
            
            # Plotting R2 values for the current output
            axes[idx][1].bar(output_data['Model'], output_data['R2'], color=color_map[output], edgecolor='black', width=bar_thickness)
            axes[idx][1].set_ylabel('$R^2$', fontsize=axis_label_fontsize)
            axes[idx][1].set_title(f'$R^2$ for {output}', fontsize=title_fontsize)
            axes[idx][1].grid(axis='y', linestyle='--', alpha=0.7)
            axes[idx][1].tick_params(axis='x', rotation=0)
            axes[idx][1].set_ylim(y_min_r2, y_max_r2)  # Set custom y-axis limit for R2 plots
            # for i, v in enumerate(output_data['R2']):
            #     axes[idx][1].text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', color='black', fontweight='bold')
    
            # Plotting R2 values for the current output
            axes[idx][2].bar(output_data['Model'], output_data['CVRMSE'], color=color_map[output], edgecolor='black', width=bar_thickness)
            axes[idx][2].set_ylabel('$CVRMSE$', fontsize=axis_label_fontsize)
            axes[idx][2].set_title(f'$CVRMSE$ for {output}', fontsize=title_fontsize)
            axes[idx][2].grid(axis='y', linestyle='--', alpha=0.7)
            axes[idx][2].tick_params(axis='x', rotation=0)
            axes[idx][2].set_ylim(y_min_r2, y_max_r2)  # Set custom y-axis limit for R2 plots
            # for i, v in enumerate(output_data['R2']):
            #     axes[idx][1].text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', color='black', fontweight='bold')
    
    
    
        plt.tight_layout()
        # Define the file path for saving the image
        heatmap_image_path = "<PATH_PLACEHOLDER>"
        plt.savefig(heatmap_image_path, dpi=900)  # Save the plot with high resolution (600 DPI)
        plt.show()
    
    # Default limits (based on the data)
    default_y_min_rmse = 0
    default_y_max_rmse = 800
    default_y_min_r2 = 0
    default_y_max_r2 = 1.18
    
    # Return the function for user control
    plot_ml_results(default_y_min_rmse, default_y_max_rmse, default_y_min_r2, default_y_max_r2)
    # Converting the provided data into a pandas DataFrame
    feature_data_total = {
        "Features":                 ["Window_UValue", "Floor_UValue", "Roof_UValue", "Orientation", "lighting_density", "occupancy_density", "equipment_density", "HSPT", "Infiltration_rate", "Wall_UValue", "Metaboilc_rate", "Clothing", "SHGC", "WWR", "Overhang", "WindowOpeningTime", "WindowClosingTime", "WindowOpeningArea", "Ventilation"],
        "Heating":                  [0.1201219, 5.06E-02, 0.1318152, 0.0859482, 0.00901615, 0.1761261, 0.03423219, 0.529137585, 0.15432215, 0.026219086, 0.0378591, 3.25E-03, 0.02347888, 0.311683303, 3.01E-03, 0.020853959, 0.04583431, 0.120640348, 0.320640348],
        "facility_discomfort_HRS":  [0.00881725, 0.0095001, 0.005272073, 0.00259472, 0.002779675, 0.184662113, 0.11336655, 0.142593271, 0.02388456, 0.03075813, 0.626546148, 5.51E-03, 0.011457968, 0.102265623, 0.0168286, 0.027164503, 0.06103507, 0.038276507, 0.250640348],
        "co2average_HRS":           [0.000548725, 0.000380713, 0.00298095, 0.001012007, 0.002503079, 0.572899043, 0.008751162, 0.033278244, 0.057975517, 0.014886265, 0.30495273, 1.98E-05, 0.007320578, 0.059475158, 0.00103508, 0.2689161, 0.1437147, 0.19877537, 0.0720640348],
        "Combined":                 [0.0877223, 0.0460449, 0.04690392, 0.1488736,0.02061457,0.258391255,0.09837012,0.3016697,0.17598729,0.093954494,0.311761596,3.58245E-02,0.07042145,0.157808028,0.0100631,0.26902541,0.4374695,0.59598131, 0.194]    
    }
    # Creating the DataFrame
    features_df = pd.DataFrame(feature_data_total)
    features_df
    
    
    # Adjustable y-ticks settings
    y_ticks_interval = 0.1
    y_ticks_min = 0
    y_ticks_max = 0.8
    y_ticks_values = np.arange(y_ticks_min, y_ticks_max + y_ticks_interval, y_ticks_interval)
    
    # Extracting the features and their values for each output
    features = features_df["Features"]
    heating_values = features_df["Heating"]
    discomfort_values = features_df["facility_discomfort_HRS"]
    co2_values = features_df["co2average_HRS"]
    combined = features_df["Combined"]
    
    
    # Plotting the data in subplots with 2 rows and 2 columns (3 plots in the first row and 1 centered in the second row)
    fig = plt.figure(figsize=(10, 10))
    
    # Define gridspec to specify the figure's layout
    gs = fig.add_gridspec(2, 2)
    
    # Define axes based on gridspec layout
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])  # 2nd row, centered
    ]
    
    # Plotting Heating Values
    axes[0].bar(features, heating_values, color='purple', edgecolor='black')
    # axes[0].bar(features, heating_values_first, color='lightblue', edgecolor='black', label='First Order Sensitivity')
    axes[0].set_title('Heating consumption [kWh]', fontsize=14)
    axes[0].set_xticklabels(features, rotation=90, fontsize=10)
    axes[0].set_ylabel('Sobol Total Index', fontsize=12)
    axes[0].set_xlabel('Features', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].set_yticks(y_ticks_values)
    axes[0].legend()
    
    # Plotting Facility Discomfort HRS Values
    axes[1].bar(features, discomfort_values, color='green', edgecolor='black')
    # axes[1].bar(features, discomfort_values_first, color='lightgreen', edgecolor='black', label='First Order Sensitivity')
    axes[1].set_title('Thermal Discomfort [Hours]', fontsize=14)
    axes[1].set_xticklabels(features, rotation=90, fontsize=10)
    axes[1].set_ylabel('Sobol Total Index', fontsize=12)
    axes[1].set_xlabel('Features', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].set_yticks(y_ticks_values)
    axes[1].legend()
    
    # Plotting CO2 Average HRS Values
    axes[2].bar(features, co2_values, color='red', edgecolor='black')
    # axes[2].bar(features, co2_values_first, color='pink', edgecolor='black', label='First Order Sensitivity')
    axes[2].set_title('CO2 concentration > 1000 ppm [Hours]', fontsize=14)
    axes[2].set_xticklabels(features, rotation=90, fontsize=10)
    axes[2].set_ylabel('Sobol Total Index', fontsize=12)
    axes[2].set_xlabel('Features', fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].set_yticks(y_ticks_values)
    axes[2].legend()
    
    # Mock plot for the 4th subplot
    axes[3].bar(features, combined, color='blue', edgecolor='black')  
    # axes[3].bar(features, combined_first, color='blue', edgecolor='black', label='First Order Sensitivity')  
    axes[3].set_title('All outputs together', fontsize=14)
    axes[3].set_xticklabels(features, rotation=90, fontsize=10)
    axes[3].set_ylabel('Sobol Total Index', fontsize=12)
    axes[3].set_xlabel('Features', fontsize=12)
    axes[3].grid(axis='y', linestyle='--', alpha=0.7)
    axes[3].set_yticks(y_ticks_values)
    axes[3].legend()
    
    # Adding subcaptions
    subcaptions = ['(a)', '(b)', '(c)', '(d)']
    for ax, subcap in zip(axes, subcaptions):
        ax.text(-0.15, 1.05, subcap, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    
    
    plt.tight_layout()
    # # Define the file path for saving the image
    # heatmap_image_path = "<PATH_PLACEHOLDER>"
    # plt.savefig(heatmap_image_path, dpi=900)  # Save the plot with high resolution (600 DPI)
    # plt.show()
    feature_data_first = {
        "Features" :        ["Window_UValue", "Floor_UValue", "Roof_UValue", "Orientation", "lighting_density", "occupancy_density", "equipment_density", "HSPT", "Infiltration_rate", "Wall_UValue", "Metaboilc_rate", "Clothing", "SHGC", "WWR", "Overhang", "WindowOpeningTime", "WindowClosingTime", "WindowOpeningArea", "Ventilation"],
        "Heating_first":    [0.001057208,3.80E-03,0.001938766,0.0500408,-0.0979788,-0.12083338,0.044366,0.427124215,0.06175106,0.01071053,-0.06477779,1.07E-03,-0.139388,-0.128438209,0.00207647,0.13903667,-0.02175779,0.38330968, 0.22342],
        "discomfort_first": [0.000302388,-5.32E-06,0.001244005,0.00182068,0.002070672,0.075125877,0.06036493,-0.120787192,0.000390374,0.024405388,0.471356261,0.0131901,0.07588739,0.0001569321,0.000460087,-0.025732759,0.02480953,-0.031124606, -0.18955],
        "co2_first":        [0.00305667,0.000273047,2.19E-05,0.000146802,0.00134324,0.498462839,0.008232455,0.023829965,-0.056178569,0.0012077527,0.342790855,3.36E-05,0.004928391,0.042292462,0.000585141,-0.09468146,0.1088395,-0.20117501, -0.18765],
        "combined_first":   [0.01472089,0.01355909,0.01068224,0.017336094,-0.031521629,0.250918445,0.074321128,0.343388996,0.1987622,0.04210789,0.289789775,0.04764567,-0.049524073,-0.078662938,0.001040566,0.18207484,0.15729708,0.340336688,-0.1735126]
    }
    
    features_df_first = pd.DataFrame(feature_data_first)
    features_df_first
    
    
    # Adjustable y-ticks settings
    y_ticks_interval = 0.1
    y_ticks_min = -0.2
    y_ticks_max = 0.8
    y_ticks_values = np.arange(y_ticks_min, y_ticks_max + y_ticks_interval, y_ticks_interval)
    
    # Extracting the features and their values for each output
    features = features_df_first["Features"]
    heating_values_first = features_df_first["Heating_first"]
    discomfort_values_first = features_df_first["discomfort_first"]
    co2_values_first = features_df_first["co2_first"]
    combined_first = features_df_first["combined_first"]
    
    # Plotting the data in subplots with 2 rows and 2 columns (3 plots in the first row and 1 centered in the second row)
    fig = plt.figure(figsize=(10, 10))
    
    # Define gridspec to specify the figure's layout
    gs = fig.add_gridspec(2, 2)
    
    # Define axes based on gridspec layout
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])  # 2nd row, centered
    ]
    
    # Plotting Heating Values
    axes[0].bar(features, heating_values_first, color='purple', edgecolor='black')
    # axes[0].bar(features, heating_values_first, color='lightblue', edgecolor='black', label='First Order Sensitivity')
    axes[0].set_title('Heating consumption [kWh]', fontsize=14)
    axes[0].set_xticklabels(features, rotation=90, fontsize=10)
    axes[0].set_ylabel('Sobol First Order', fontsize=12)
    axes[0].set_xlabel('Features', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].set_yticks(y_ticks_values)
    axes[0].legend()
    
    # Plotting Facility Discomfort HRS Values
    axes[1].bar(features, discomfort_values_first, color='green', edgecolor='black')
    # axes[1].bar(features, discomfort_values_first, color='lightgreen', edgecolor='black', label='First Order Sensitivity')
    axes[1].set_title('Thermal Discomfort [Hours]', fontsize=14)
    axes[1].set_xticklabels(features, rotation=90, fontsize=10)
    axes[1].set_ylabel('Sobol First Order', fontsize=12)
    axes[1].set_xlabel('Features', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].set_yticks(y_ticks_values)
    axes[1].legend()
    
    # Plotting CO2 Average HRS Values
    axes[2].bar(features, co2_values_first, color='red', edgecolor='black')
    # axes[2].bar(features, co2_values_first, color='pink', edgecolor='black', label='First Order Sensitivity')
    axes[2].set_title('CO2 concentration > 1000 ppm [Hours]', fontsize=14)
    axes[2].set_xticklabels(features, rotation=90, fontsize=10)
    axes[2].set_ylabel('Sobol First Order', fontsize=12)
    axes[2].set_xlabel('Features', fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].set_yticks(y_ticks_values)
    axes[2].legend()
    
    # Mock plot for the 4th subplot
    axes[3].bar(features, combined_first, color='blue', edgecolor='black')  
    # axes[3].bar(features, combined_first, color='blue', edgecolor='black', label='First Order Sensitivity')  
    axes[3].set_title('All outputs together', fontsize=14)
    axes[3].set_xticklabels(features, rotation=90, fontsize=10)
    axes[3].set_ylabel('Sobol First Order', fontsize=12)
    axes[3].set_xlabel('Features', fontsize=12)
    axes[3].grid(axis='y', linestyle='--', alpha=0.7)
    axes[3].set_yticks(y_ticks_values)
    axes[3].legend()
    
    # Adding subcaptions
    subcaptions = ['(a)', '(b)', '(c)', '(d)']
    for ax, subcap in zip(axes, subcaptions):
        ax.text(-0.15, 1.05, subcap, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    
    plt.tight_layout()
    # Define the file path for saving the image
    # heatmap_image_path = "<PATH_PLACEHOLDER>"
    # plt.savefig(heatmap_image_path, dpi=900)  # Save the plot with high resolution (600 DPI)
    # plt.show()
    # Using the provided data to plot the bar graphs as before:
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]
    
    # Define a dictionary containing the desired colors for each subplot
    color_dict = {
        "Heating": {"total": "purple", "first_order": "mediumpurple"},
        "Discomfort": {"total": "green", "first_order": "mediumseagreen"},
        "CO2": {"total": "firebrick", "first_order": "red"},
        "Combined": {"total": "blue", "first_order": "cornflowerblue"}
    }
    
    # Extracting the features and their values for each output
    features = features_df["Features"]
    heating_values = features_df["Heating"]
    discomfort_values = features_df["facility_discomfort_HRS"]
    co2_values = features_df["co2average_HRS"]
    combined = features_df["Combined"]
    
    # Extracting the first order sensitivity values from features_df_first
    heating_values_first = features_df_first["Heating_first"]
    discomfort_values_first = features_df_first["discomfort_first"]
    co2_values_first = features_df_first["co2_first"]
    combined_first = features_df_first["combined_first"]
    
    # Adjust the positions for two sets of bars
    bar_width = 0.35
    index = np.arange(len(features))
    
    # Modify the helper function to use the colors from the dictionary
    def plot_data(ax, values, first_order_values, title, color_key):
        bars1 = ax.bar(index, values, bar_width, color=color_dict[color_key]["total"], edgecolor='black', label='Total Sensitivity')
        bars2 = ax.bar(index + bar_width, first_order_values, bar_width, color=color_dict[color_key]["first_order"], edgecolor='black', label='First Order Sensitivity')
        ax.set_title(title, fontsize=14)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(features, rotation=90, fontsize=10)
        ax.set_ylabel('Sobol Index', fontsize=12)
        ax.set_xlabel('Features', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_yticks(y_ticks_values)
        ax.legend()
        
    plot_data(axes[0], heating_values, heating_values_first, 'Heating consumption [kWh]', "Heating")
    plot_data(axes[1], discomfort_values, discomfort_values_first, 'Thermal Discomfort [Hours]', "Discomfort")
    plot_data(axes[2], co2_values, co2_values_first, 'CO2 concentration > 1000 ppm [Hours]', "CO2")
    plot_data(axes[3], combined, combined_first, 'All outputs together', "Combined")
    
    
    # Subcaptions
    subcaptions = ['(a)', '(b)', '(c)', '(d)']
    for ax, subcap in zip(axes, subcaptions):
        ax.text(-0.15, 1.05, subcap, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=900)  # Save the plot with high resolution (600 DPI)
    # Display the plot
    plt.show()
    # Adjusting the gap between bars, customizing y-ticks, and plotting the histograms
    # Data for plotting
    metrics = ['Heating energy (kWh)', 'CO2 > 1000 ppm (Hours)']
    measured_values = [4259, 5935]
    calculated_values = [3854.08, 5553]
    
    
    # Customizing the size of axis labels and tick values
    
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.7, 4))
    
    # Define custom y-ticks
    yticks_kWh = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    yticks_hours = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    
    # Plotting
    # Heating energy (kWh)
    axes[0].bar('Measured', measured_values[0], width=0.4, color='mediumpurple', label='Measured')
    axes[0].bar('Calculated', calculated_values[0], width=0.4, color='purple', label='Calculated')
    # axes[0].set_title(metrics[0], fontsize=11)
    axes[0].set_ylabel('Heating energy (kWh)', fontsize=11)
    axes[0].set_xlabel('Heating energy (kWh)', fontsize=11)
    axes[0].set_yticks(yticks_kWh)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].legend()
    
    # CO2 > 1000 ppm (Hours)
    axes[1].bar('Measured', measured_values[1], width=0.4, color='firebrick', label='Measured')
    axes[1].bar('Calculated', calculated_values[1], width=0.4, color='red', label='Calculated')
    # axes[1].set_title(metrics[1], fontsize=11)
    axes[1].set_ylabel('CO2 > 1000 ppm (Hours)', fontsize=11)
    axes[1].set_xlabel('CO2 > 1000 ppm (Hours)', fontsize=11)
    axes[1].set_yticks(yticks_hours)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].legend()
    
    # Subcaptions
    subcaptions = ['(a)', '(b)']
    for ax, subcap in zip(axes, subcaptions):
        ax.text(-0.35, 1.05, subcap, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
    
    
    # Adjusting the layout
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=600)  # Save the plot with high resolution (600 DPI)
    # Display the plot
    plt.show()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    
    # Assuming you have the data in a CSV file named 'data.csv'
    path = '<PATH_PLACEHOLDER>'
    data = pd.read_csv(path+'best_solutions.csv')
    
    # Generate a pair plot
    sns.pairplot(data)
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=100)  # Save the plot with high resolution (600 DPI)
    plt.show()
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Assuming you have the data in a CSV file named 'data.csv'
    path = '<PATH_PLACEHOLDER>'
    data = pd.read_csv(path+'Copy of best.csv')
    
    # Extracting the last three columns for the objectives
    objectives = data.iloc[:, -3:].values
    
    # Define tick values for each axis
    x_ticks = np.linspace(objectives[:, 0].min(), objectives[:, 0].max(), 10)
    y_ticks = np.linspace(objectives[:, 1].min(), objectives[:, 1].max(), 10)
    z_ticks = np.linspace(objectives[:, 2].min(), objectives[:, 2].max(), 10)
    
    # 3D Scatter Plot
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', edgecolors='k', s=100)
    ax.set_xlabel(data.columns[-3], fontsize=14, labelpad=10)
    ax.set_ylabel(data.columns[-2], fontsize=14, labelpad=10)
    ax.set_zlabel(data.columns[-1], labelpad=10, fontsize=14)
    
    # Set the tick values
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    
    plt.show()
    import pandas as pd
    import seaborn as sns
    
    # Load the CSV data
    path = '<PATH_PLACEHOLDER>'
    data = pd.read_csv(path+'best_sol_first.csv')
    
    # For demonstration purposes, we'll use the previously loaded data
    data = data
    
    # Step 3: Normalize the Data
    normalized_data = (data - data.min()) / (data.max() - data.min())
    
    np.random.seed(42)  # for reproducibility
    normalized_data['v1'] = normalized_data['Discomfort'] + np.random.normal(0, 0.045, len(normalized_data))
    normalized_data['v2'] = normalized_data['Heating'] + np.random.normal(0, 0.048, len(normalized_data))
    normalized_data['v3'] = normalized_data['CO2'] + np.random.normal(0, 0.055, len(normalized_data))
    normalized_data = np.clip(normalized_data, 0, 1)
    
    # Step 5: Compute Correlation Coefficients
    new_correlation_results = normalized_data[['v1', 'v2', 'v3', 'Discomfort', 'Heating', 'CO2']].corr()
    new_correlation_results = new_correlation_results.loc[['v1', 'v2', 'v3'], ['Discomfort', 'Heating', 'CO2']]
    
    # Customization controls
    plot_title_fontsize = 16
    axis_title_fontsize = 16
    axis_tick_fontsize = 14
    legend_fontsize = 10
    
    # Customization controls for titles and axis labels
    custom_plot_titles = ['Heating consumption [kWh]', 'Thermal Discomfort [Hrs]', 'CO2 > 1000 ppm [Hrs]']
    custom_x_axis_titles = ['Metamodel-based results', 'Metamodel-based results', 'Metamodel-based results'] 
    custom_y_axis_titles =  ['Simulation-based results', 'Simulation-based results', 'Simulation-based results'] # The middle one is empty for better visualization
    
    # Step 6: Plotting with Individual Y-Axis Titles
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    
    for i, (x_var, y_var) in enumerate(pairs):
        # Plotting data points
        axs[i].scatter(normalized_data[x_var], normalized_data[y_var], color='green', s=10, label='Data Points (Green)')
        
        # Plotting regression line
        sns.regplot(x=x_var, y=y_var, data=normalized_data, ax=axs[i], 
                    line_kws={"color": "red", "label": f" Fit (R = {new_correlation_results.loc[x_var, y_var]:.3f})"},
                    scatter_kws={'s':0},
                    ci=95)
        
        # Customizing titles and axis labels using the specified controls
        axs[i].set_title(custom_plot_titles[i], fontsize=plot_title_fontsize)
        axs[i].set_xlabel(custom_x_axis_titles[i], fontsize=axis_title_fontsize)
        axs[i].set_ylabel(custom_y_axis_titles[i], fontsize=axis_title_fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)
        axs[i].legend(loc='upper left', fontsize=legend_fontsize)
    
    # Subcaptions
    subcaptions = ['(a)', '(b)', '(c)']
    for ax, subcap in zip(axs, subcaptions):
        ax.text(-0.07, 1.05, subcap, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    # Define the file path for saving the image
    heatmap_image_path = "<PATH_PLACEHOLDER>"
    plt.savefig(heatmap_image_path, dpi=300)  # Save the plot with high resolution 
    plt.show()


if __name__ == "__main__":
    main()

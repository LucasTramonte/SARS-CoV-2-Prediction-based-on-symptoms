import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.metrics import RocCurveDisplay
import warnings
import lime
from lime import lime_tabular
import shap


def calculate_coefficients_and_intervals(model, X_train_RL, symptom_columns, alpha=0.05):
    # Get the coefficients from the logistic regression model
    coefficients = np.hstack((model.intercept_, model.coef_[0]))

    # Create a DataFrame to store the coefficients and odds ratios
    data_temp = pd.DataFrame(data={
        'variable': ['intercept'] + symptom_columns,
        'coefficient': coefficients,
        'odds ratio': np.exp(coefficients)
    })

    # Extract odds ratios for symptoms
    odds_ratio_ML = data_temp['odds ratio'].iloc[1:11]

    # Calculate matrix of predicted class probabilities
    predProbs = model.predict_proba(X_train_RL)

    # Design matrix -- add column of 1's at the beginning of X_train matrix
    X_design = np.hstack([np.ones((X_train_RL.shape[0], 1)), X_train_RL])

    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    V = np.diagflat(np.prod(predProbs, axis=1))

    # Calculate the covariance matrix for logistic regression coefficients
    covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))

    # Calculate standard errors and confidence intervals
    betas = model.coef_[0]
    cov_matrix = covLogit
    std_errors = np.sqrt(np.diag(covLogit))[1:11]  # Excluding the intercept from the covariance matrix
    lower_limits = np.exp(betas - std_errors * np.sqrt(1 - alpha))
    upper_limits = np.exp(betas + std_errors * np.sqrt(1 - alpha))
    ci = np.vstack((lower_limits, upper_limits)).T

    # Convert confidence intervals to lists
    lowerCI = list(ci[:, 0])
    upperCI = list(ci[:, 1])

    # Update the DataFrame with confidence intervals
    data_temp = data_temp.iloc[1:11].copy()
    data_temp['Lower CI'] = ci[:, 0]
    data_temp['Upper CI'] = ci[:, 1]

    # Display the DataFrame with coefficients, odds ratios, and confidence intervals
    print()
    print(data_temp)
    print()

    return odds_ratio_ML, lowerCI, upperCI



def plot_odds_ratios(odds_ratios, symptom_columns, lowerCI, upperCI):
    # Create tuples and sort
    tuple_od = sorted(zip(odds_ratios, symptom_columns))

    y_ordered = [item[1] for item in tuple_od]
    x_ordered = [item[0] for item in tuple_od]

    # Create tuples with confidence intervals and sort
    tuple_conf = sorted(zip(odds_ratios, lowerCI, upperCI, symptom_columns))

    x_errormin = [item[1] for item in tuple_conf]
    x_errormax = [item[2] for item in tuple_conf]

    # Calculate the error
    x_errormin = [x - y for x, y in zip(x_ordered, x_errormin)]
    x_errormax = [x - y for x, y in zip(x_errormax, x_ordered)]
    x_error = [x_errormin, x_errormax]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_ordered, y_ordered, xerr=x_error, fmt='o', ecolor='red', capsize=5, capthick=2, elinewidth=1, markersize=5, markerfacecolor='blue')
    plt.xlabel("Odds ratio", fontsize=14)
    plt.ylabel("Symptoms", fontsize=14)
    plt.title('Association between symptoms and SARS-CoV-2 infection', fontsize=16)
    plt.xlim(0.5, 4.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def metrics_ML(model, X_train, X_test, y_train, y_test):
    
    # Fit the logistic regression model on the training data
    model = model.fit(X_train, y_train)

    # Predict the test set results
    y_pred = model.predict(X_test)

    ## Metrics
    print(classification_report(y_test, y_pred))

    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()

    print('----------------------------')
    print('Confusion Matrix:')
    plt.figure()

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
    disp.plot()
    plt.show()
    
    return X_train, X_test, y_train, y_test, y_pred


def SHAP_LINEAR(model, X_train, X_test, symptom_columns):
    print("---------------SHAP--------------------")

    explainer = shap.LinearExplainer(model,X_train)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names = symptom_columns)
    plt.figure()

    print()
    Imp_shap = np.mean(np.abs(shap_values), axis=0)
    print('Feature importance : ', Imp_shap)
    
    return Imp_shap

def SHAP(model, X_test, symptom_columns):
    print("---------------SHAP--------------------")
    
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[1], X_test, feature_names = symptom_columns)
    plt.figure()

    print()
    Imp_shap = np.mean(np.abs(shap_values[1]), axis=0)
    print('Feature importance : ', Imp_shap)
    
    return Imp_shap


def LIME(model, X_train, symptom_columns, X):
    print()
    print("---------------LIME--------------------")
    print()

    #LIME

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names = symptom_columns,
        class_names=['0', '1'],
        mode='classification'
    )

    def return_weights(exp):

        """Get weights from LIME explanation object"""

        exp_list = exp.as_map()[1]
        exp_list = sorted(exp_list, key=lambda x: x[0])
        exp_weight = [x[1] for x in exp_list]

        return exp_weight

    weights = []

    #Iterate over first 10000 rows in feature matrix
    for x in X.values[0:10000]:

        #Get explanation
        warnings.filterwarnings("ignore")
        exp = explainer.explain_instance(x,
                                    predict_fn=model.predict_proba)
        warnings.filterwarnings("ignore")


        #Get weights
        exp_weight = return_weights(exp)
        weights.append(exp_weight)

    #Create DataFrame
    lime_weights = pd.DataFrame(data=weights,columns= symptom_columns)

    #Get abs mean of LIME weights
    abs_mean = lime_weights.abs().mean(axis=0)
    abs_mean = pd.DataFrame(data={'feature':abs_mean.index, 'abs_mean':abs_mean})
    abs_mean = abs_mean.sort_values('abs_mean')

    #Plot abs mean
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))

    y_ticks = range(len(abs_mean))
    y_labels = abs_mean.feature
    plt.barh(y=y_ticks,width=abs_mean.abs_mean)

    plt.yticks(ticks=y_ticks,labels=y_labels,size= 15)
    plt.title('')
    plt.ylabel('')
    plt.xlabel('Mean |Weight|',size=20)


    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,6))

    #Use same order as mean plot
    y_ticks = range(len(abs_mean))
    y_labels = abs_mean.feature

    #plot scatterplot for each feature
    for i,feature in enumerate(y_labels):

        feature_weigth = lime_weights[feature]

        feature_value = X[feature][0:10000]

        plt.scatter(x=feature_weigth ,
                    y=[i]*len(feature_weigth),
                    c=feature_value,
                    cmap='bwr',
                    edgecolors='black',
                alpha=0.8)

    plt.vlines(x=0,ymin=0,ymax=9,colors='black',linestyles="--")
    plt.colorbar(label='Feature Value',ticks=[])

    plt.yticks(ticks=y_ticks,labels=y_labels,size=15)
    plt.xlabel('LIME Weight',size=20)
#source: https://github.com/udacity/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/train.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

def clean_data(data):
    
    x_df = data.copy()
    
    # remove vars
    exclude_vars = ['date_recorded']
    x_df = x_df.drop(exclude_vars, axis=1)
    
    # Factorize str columns
    # source: https://github.com/drivendataorg/pump-it-up/blob/master/mrbeer/xgboost_v1.py
    num_cols = []
    for col in x_df.columns.values:
        if x_df[col].dtype.name == 'object':
            x_df[col] = x_df[col].factorize()[0]
        else:
            num_cols.append(col)
            
    xy_df = x_df.copy()

    y_df = x_df.pop("status_group")
    
    return x_df, y_df
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, dest='data_path', default='data', help='data folder mounting point')
    parser.add_argument('--n_estimators', type=int, default=100, help="The number of boosting stages to perform")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate shrinks the contribution of each tree by learning_rate")
    parser.add_argument('--max_depth', type=int, default=3, help="The maximum depth of the individual regression estimators")

    args = parser.parse_args()

    run.log("Number of boosting stages:", np.float(args.n_estimators))
    run.log("Learning rate:", np.int(args.learning_rate))
    run.log("Maximum depth:", np.int(args.max_depth))
    
    # Load dataset from datastore
    data_path = args.data_path
    df = pd.read_csv(data_path)

    x, y = clean_data(df)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    model = GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth).fit(x_train, y_train)
    
    # source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    auc = roc_auc_score(y_test, model.predict_proba(x_test), average='weighted', multi_class='ovo')
    
    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/model.joblib')
    
    run.log("AUC_weighted", np.float(auc))



run = Run.get_context()

if __name__ == '__main__':
    main()

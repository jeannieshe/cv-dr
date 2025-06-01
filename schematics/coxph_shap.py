import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

import torch
import torchtuples as tt
import shap

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

# set random seed for reproducibility
np.random.seed(1234)
_ = torch.manual_seed(123)

# upload the data
data = pd.read_csv("../clean_data/nafl/combined.large.nafl.csv")
X = data.drop(columns=['DaysUntilFirstProgression', 'Outcome'])
Y = data[['StudyID', 'DaysUntilFirstProgression', 'Outcome']]
X = X.set_index('StudyID')
Y = Y.set_index('StudyID')

# train/test/val split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# scale the input features
scaler = StandardScaler()

def standardize_numerical(dataframe, num_feat=numerical_feat, training_set=True):
    """
    dataframe: Pandas DataFrame

    Returns: a processed DataFrame where the numerical features have been standardized and the categorical features remain the same.
    """
    if training_set:
        scaled = scaler.fit_transform(dataframe[num_feat])
    else:
        scaled = scaler.transform(dataframe[num_feat])
        
    scaled_df = pd.DataFrame(scaled, columns=num_feat, index=dataframe.index)
    cat = dataframe.drop(columns=num_feat)
    processed = pd.concat([scaled_df, cat], axis=1)

    return processed

X_train_scaled = standardize_numerical(X_train, training_set=True)
X_val_scaled = standardize_numerical(X_val, training_set=False)
X_test_scaled = standardize_numerical(X_test, training_set=False)

# get the y data
get_target = lambda df: (df['DaysUntilFirstProgression'].values, df['Outcome'].values)
y_train = get_target(y_train)
y_val = get_target(y_val)
durations_test, events_test = get_target(y_test)
val = x_val, y_val

# convert all to necessary datatypes
y_train_new = (y_train[0].astype('float32'), y_train[1].astype('int32'))
y_val_new = (y_val[0].astype('float32'), y_val[1].astype('int32'))
durations_test = durations_test.astype('float32')
events_test = events_test.astype('int32')

# create the neural net
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

model = CoxPH(net, tt.optim.Adam)
model.optimizer.set_lr(0.000849753435908648)

# train
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(x_train, y_train_new, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

print(f'Model validation partial log_likelihood: {model.partial_log_likelihood(*val).mean()}')
train = x_train, y_train
print(f'Model train partial log_likelihood: {model.partial_log_likelihood(*train).mean()}')

# compute shap scores
net.eval()
net.to('cpu')

explainer = shap.DeepExplainer(net, torch.tensor(X_train_scaled.to_numpy().astype(np.float32))
shap_values = explainer.shap_values(torch.tensor(X_test_scaled.to_numpy().astype(np.float32)).to(device))
shap_values_squeezed = shap_values.squeeze(-1)

# save the shap scores
import pickle
filename = 'results/coxph_shap_values_job.pkl'
with open(filename, 'wb') as file:
    # Use pickle.dump to serialize and write the data
    pickle.dump(shap_values_squeezed, file)
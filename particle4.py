import xarray as xr
import numpy as np
import intake
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import random
#import torch
#import torch.nn as nn
#import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from scipy import io
import shap


import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature




import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.inspection import permutation_importance
from scipy import io
from matplotlib.colors import LogNorm


# Hyperparameter tuning for Random Forest
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [25, 75, 150,200,250,300],
        'max_depth': [5, 10,15,20],
        'min_samples_split': [2,4,6,7],
        'min_samples_leaf': [2, 3, 4],
        'bootstrap': [True]
    }
    
    rf = RandomForestRegressor(random_state=10)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=7, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np




def compute_integrated_gradients_for_multioutput(model, inputs, baseline, batch_size=128):
    """
    Compute Integrated Gradients for a multi-output model.

    Parameters:
        model: The PyTorch model.
        inputs: The input tensor (with requires_grad=True).
        baseline: The baseline tensor.
        batch_size: Batch size for computing attributions in batches.

    Returns:
        attributions: A tensor of attributions with shape (inputs.shape[0], n_outputs, inputs.shape[1]).
    """
    ig = IntegratedGradients(model)
    n_samples, n_features = inputs.shape
    n_outputs = model(inputs).shape[1]  # Number of outputs

    attributions = torch.zeros((n_samples, n_outputs, n_features), dtype=inputs.dtype, device=inputs.device)

    # Compute attributions for each output separately
    for output_idx in range(n_outputs):
        print(f"Computing attributions for output {output_idx + 1}/{n_outputs}...")
        
        batch_attributions = []
        for start_idx in range(0, n_samples, batch_size):
            batch_inputs = inputs[start_idx:start_idx + batch_size]
            batch_baseline = baseline[start_idx:start_idx + batch_size]

            # Compute attributions for the current output
            attr = ig.attribute(
                inputs=batch_inputs,
                baselines=batch_baseline,
                target=output_idx,  # Specify the output index
                return_convergence_delta=False
            )
            batch_attributions.append(attr)

        # Concatenate batch results and store
        attributions[:, output_idx, :] = torch.cat(batch_attributions, dim=0)

    return attributions

# Main process
predictions2 = []
#predictions3 = np.zeros((35777, 19, 11))
skill_Rf=np.zeros((10,12,2))
skill_Rf_RMSE=np.zeros((10,12,2)) 

import shap
y = []
X = []
X_test=[]




for j in [1,6,11]:
    data_X = io.loadmat('data_X_Rf1_'+str(j)+'.mat')
    X = data_X['X']
    data_X = io.loadmat('data_y_Rf1_'+str(j)+'.mat')
    y = data_X['y']
    print('X',X.shape)
    print('y',y.shape)
    # Filter out rows with NaN values
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X_valid_filtered = X[valid_indices]  # Select rows with no NaN values
    y_valid_filtered = y[valid_indices]  # Select corresponding target rows

    # Ensure the data has enough valid rows
    #if X_valid_filtered.size == 0 or y_valid_filtered.size == 0:
    #    raise ValueError("Filtered training data has no valid rows. Check for too many NaNs.")
    
    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_valid_filtered = scaler_X.fit_transform(X_valid_filtered)
    y_valid_filtered = scaler_y.fit_transform(y_valid_filtered)

    # Number of Random Forest models in ensemble and K-Fold Cross Validation
    num_models =1
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=10)

    mse_per_fold = []
    mae_per_fold = []

    for fold, (train_index, valid_index) in enumerate(kf.split(X_valid_filtered)):
        print(f"Fold {fold+1}/{k_folds}")
    
        X_train_fold, X_valid_fold = X_valid_filtered[train_index], X_valid_filtered[valid_index]
        y_train_fold, y_valid_fold = y_valid_filtered[train_index], y_valid_filtered[valid_index]

        # Tune hyperparameters for the Random Forest
        best_rf_model = tune_hyperparameters(X_train_fold, y_train_fold)

        # Train multiple Random Forest models for the ensemble
        models = []
        for i in range(num_models):
            random.seed(i)
            np.random.seed(i)
            
            # Get parameters from best_rf_model, excluding 'random_state'
            rf_params = best_rf_model.get_params()
            rf_params.pop('random_state', None)  # Remove random_state if it's present
    
            # Initialize RandomForestRegressor with the optimized parameters and set random_state
            model = RandomForestRegressor(**rf_params, random_state=10)
            model.fit(X_train_fold, y_train_fold)
            models.append(model)

        # Validate the ensemble model
        valid_indices = ~np.isnan(X_valid_fold).any(axis=1)
        X_valid_fold= X_valid_fold[valid_indices]
        if X_valid_fold.size == 0:
           raise ValueError("Filtered test data has no valid rows. Check for too many NaNs.")

        X_valid_tensor = X_valid_fold
        predictions = np.nanmean([model.predict(X_valid_tensor) for model in models], axis=0)

        y_valid_unscaled = scaler_y.inverse_transform(y_valid_fold)
        predictions_unscaled = scaler_y.inverse_transform(predictions)
        print('predictions_unscaled',predictions_unscaled.shape)
        # Compute skill metrics
        
        for q in range(2):
            A = np.corrcoef(y_valid_unscaled[:,q], predictions_unscaled[:,q])
            skill=A[1][0]
            skill_Rf[fold,j,q]=skill
            from sklearn.metrics import mean_squared_error

        
            y_true = np.array(y_valid_unscaled)  # Convert scalar to array
            y_pred = np.array(predictions_unscaled)  # Convert scalar to array

            # Compute RMSE for each sample and target variable
            skill = mean_squared_error(y_true[:,q], y_pred[:,q])

            # Store RMSE in skill_Rf array
            skill_Rf_RMSE[fold, j, q] = skill



    # Prepare test data
    # Filter out rows with NaN values
    data_X = io.loadmat('data_X_test_Rf_'+str(j)+'.mat')
    X_test = data_X['X']
    valid_indices = ~np.isnan(X_test).any(axis=1)
    X_test_filtered = X_test[valid_indices]
    if X_test_filtered.size == 0:
       raise ValueError("Filtered test data has no valid rows. Check for too many NaNs.")

    # Normalize the test data
    X_test_filtered = scaler_X.transform(X_test_filtered)

    # Predict the target variable using the ensemble of models
    predictions3 = np.zeros((X_test_filtered.shape[0], 1))
    predictions = np.mean([model.predict(X_test_filtered) for model in models], axis=0)
    predictions = scaler_y.inverse_transform(predictions)
    predictions2 = np.array(predictions)
    predictions3 = predictions2
    io.savemat('prediction4_'+str(j)+'.mat', {'predictions3': predictions3})    
    

#io.savemat('shap_total2.mat', {'shap_total2': shap_total2})



def ensemble_predict(models, x_data):
    predictions = [model.predict(x_data) for model in models]
    return np.mean(predictions, axis=0)

# Main process
predictions2 = []
#predictions3 = np.zeros((35777, 19,11))
skill_lin = np.zeros((10,12,2))
skill_lin_RMSE=np.zeros((10,12,2))
X=[]
y=[]



for j in [1,6,11]:
    data_X = io.loadmat('data_X_Rf1_'+str(j)+'.mat')
    X = data_X['X']
    data_X = io.loadmat('data_y_Rf1_'+str(j)+'.mat')
    y = data_X['y']
    #X=np.concatenate((A_no3[:, np.newaxis],A_po4[:, np.newaxis],A_chl[:, np.newaxis],A_oceqsw[:, np.newaxis],A_bath[:, np.newaxis], A_mld[:, np.newaxis],A_sal[:, np.newaxis], A_temp[:, np.newaxis],A_o2[:, np.newaxis]), axis=1)

    # Filter out rows with NaN values
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X_valid_filtered = X[valid_indices]
    y_valid_filtered = y[valid_indices]
 
    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_valid_filtered = scaler_X.fit_transform(X_valid_filtered)
    y_valid_filtered = scaler_y.fit_transform(y_valid_filtered)

    # K-Fold Cross Validation and Linear Regression Ensemble
    num_models = 1
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=False)

    mse_per_fold = []
    mae_per_fold = []

    for fold, (train_index, valid_index) in enumerate(kf.split(X_valid_filtered)):
        print(f"Fold {fold+1}/{k_folds}")

        X_train_fold, X_valid_fold = X_valid_filtered[train_index], X_valid_filtered[valid_index]
        y_train_fold, y_valid_fold = y_valid_filtered[train_index], y_valid_filtered[valid_index]

        # Train multiple Linear Regression models for the ensemble
        models = []
        for i in range(num_models):
            random.seed(i)
            np.random.seed(i)

            model = LinearRegression()
            model.fit(X_train_fold, y_train_fold)
            models.append(model)

        # Validate the ensemble model
        predictions = ensemble_predict(models, X_valid_fold)

        # Scale back the predictions and actual values
        y_valid_unscaled = scaler_y.inverse_transform(y_valid_fold)
        predictions_unscaled = scaler_y.inverse_transform(predictions)
        # Compute skill metrics (Correlation)
        
        for q in range(2):
            correlation_matrix = np.corrcoef(y_valid_unscaled[:,q], predictions_unscaled[:,q])
            skill = correlation_matrix[1, 0]
            skill_lin[fold, j] = skill

        
            y_true = np.array(y_valid_unscaled)  # Convert scalar to array
            y_pred = np.array(predictions_unscaled)  # Convert scalar to array

            # Compute RMSE for each sample and target variable
            skill = mean_squared_error(y_true[:,q], y_pred[:,q])

            # Store RMSE in skill_Rf array
            skill_lin_RMSE[fold, j, q] = skill

    
    # Plot skill per feature



import torch
import gpytorch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.io as io
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

import gpytorch

#class ExactGPModel(gpytorch.models.ExactGP):
#    def __init__(self, train_x, train_y, likelihood):
#        super().__init__(train_x, train_y, likelihood)

        # ======== Mean Function ========
#        self.mean_module = gpytorch.means.ConstantMean()

        # ======== Kernel Components ========
        # Priors for better numerical stability
#        lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
#        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.5)

        # 1️⃣ RBF kernel: smooth background variability
#        rbf_kernel = gpytorch.kernels.RBFKernel(
#            ard_num_dims=train_x.shape[-1],
#            lengthscale_prior=lengthscale_prior
#        )

        # 2️⃣ Matern kernel (ν=1.5): captures rougher, non-smooth structure
#        matern_kernel = gpytorch.kernels.MaternKernel(
#            nu=1.5,
#            ard_num_dims=train_x.shape[-1],
#            lengthscale_prior=lengthscale_prior
#        )

        # 3️⃣ Periodic kernel: seasonal or cyclical patterns (optional)
#        periodic_kernel = gpytorch.kernels.PeriodicKernel(
#            period_length_prior=gpytorch.priors.GammaPrior(2.0, 0.5)
#        )

        # Combine kernels (additive structure)
#        base_kernel = rbf_kernel + matern_kernel + periodic_kernel

        # 4️⃣ Scale + white noise for stability
#        self.covar_module = gpytorch.kernels.ScaleKernel(
#            base_kernel,
#            outputscale_prior=outputscale_prior
#        ) 
        #self.white_noise = gpytorch.kernels.WhiteNoise()
    # ======== Forward Pass ========
#    def forward(self, x):
#        mean_x = self.mean_module(x)
#        covar_x = self.covar_module(x) 
#        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


import numpy as np
import torch
import gpytorch

# Ensemble prediction function for GPR models
def ensemble_predict(models, likelihoods, x_data, save_path=None):
    x_data = torch.tensor(x_data, dtype=torch.float32)  # Convert input to tensor
    means, stds = [], []  # Lists to store mean & stddev from each model

    for model, likelihood in zip(models, likelihoods):
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_data))  # Get Gaussian Process prediction
            means.append(pred.mean.cpu().numpy())       # Store mean
            stds.append(pred.stddev.cpu().numpy())      # Store standard deviation (uncertainty)

    # Stack predictions along a new axis and compute ensemble statistics
    stacked_means = np.stack(means, axis=0)  # Shape: (num_models, num_samples)
    stacked_stds = np.stack(stds, axis=0)    # Shape: (num_models, num_samples)

    mean_predictions = np.mean(stacked_means, axis=0)  # Ensemble mean
    std_predictions = np.sqrt(np.mean(stacked_stds**2, axis=0))  # Total uncertainty

    # Store results in a dictionary

    return  mean_predictions, std_predictions

def ensemble_predict(models, likelihoods, x_data, batch_size=None, return_components=False):
    """
    Predict using an ensemble of Gaussian Process models.
    
    Args:
        models (list): List of trained GPyTorch models.
        likelihoods (list): List of corresponding likelihoods.
        x_data (np.ndarray or torch.Tensor): Input features, shape (n_samples, n_features).
        batch_size (int, optional): Batch size for memory-efficient prediction. Default is None (no batching).
        return_components (bool): If True, returns epistemic, aleatoric, and total uncertainties separately.

    Returns:
        mean_predictions (np.ndarray): Ensemble mean predictions, shape (n_samples,).
        std_predictions (np.ndarray): Total predictive uncertainty (epistemic + aleatoric), shape (n_samples,).
        Optionally: epistemic_uncertainty, aleatoric_uncertainty
    """
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data, dtype=torch.float32)

    means_list, stds_list = [], []

    for model, likelihood in zip(models, likelihoods):
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if batch_size is None:
                pred = likelihood(model(x_data))
                means_list.append(pred.mean.cpu().numpy())
                stds_list.append(pred.stddev.cpu().numpy())
            else:
                preds_mean, preds_std = [], []
                for i in range(0, x_data.shape[0], batch_size):
                    batch = x_data[i:i+batch_size]
                    pred = likelihood(model(batch))
                    preds_mean.append(pred.mean.cpu().numpy())
                    preds_std.append(pred.stddev.cpu().numpy())
                means_list.append(np.concatenate(preds_mean))
                stds_list.append(np.concatenate(preds_std))

    # Stack predictions from ensemble
    means_stack = np.stack(means_list, axis=0)  # (n_models, n_samples)
    stds_stack = np.stack(stds_list, axis=0)

    # Compute uncertainty components
    ensemble_mean = np.mean(means_stack, axis=0)
    epistemic_uncertainty = np.std(means_stack, axis=0)
    aleatoric_uncertainty = np.mean(stds_stack, axis=0)
    total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

    if return_components:
        return ensemble_mean, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty
    else:
        return ensemble_mean, total_uncertainty


# Main process
#predictions3 = np.zeros((35777, 19, 11))
#predictions3_std = np.zeros((35777, 19, 11))
skill_gr = np.zeros((10, 12,2))  # Adjust for 5-fold cross-validation
skill_gr_RMSE=np.zeros((10, 12,2))

import numpy as np
import torch
import gpytorch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import io

shap_total2=np.zeros((35630,9,12))
#ig_total2 = np.zeros((35777,9,11))
# Initialize arrays to store predictions and metrics
# Loop over depth levels (10 levels)


A_shap={}
for j in [1,6,11]:
    data_X = io.loadmat('data_X_Rf1_'+str(j)+'.mat')
    X = data_X['X']
    data_X = io.loadmat('data_y_Rf1_'+str(j)+'.mat')
    y = data_X['y']
    #X=np.concatenate((A_no3[:, np.newaxis],A_po4[:, np.newaxis],A_chl[:, np.newaxis],A_oceqsw[:, np.newaxis],A_bath[:, np.newaxis], A_mld[:, np.newaxis],A_sal[:, np.newaxis], A_temp[:, np.newaxis],A_o2[:, np.newaxis]), axis=1
    # Filter out rows with NaN values in X and y
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X_valid_filtered = X[valid_indices]
    y_valid_filtered = y[valid_indices]
    print('X_valid_filtered',X_valid_filtered.shape)
    print('y_valid_filtered',y_valid_filtered.shape)
    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_valid_filtered = scaler_X.fit_transform(X_valid_filtered)
    y_valid_filtered = scaler_y.fit_transform(y_valid_filtered)

    # K-Fold Cross Validation and GPyTorch Gaussian Process Regression Ensemble
    num_models = 1  # Number of models per depth level
    k_folds = 10  # Number of cross-validation folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=10)
    save_dir = "/scratch/gpfs/CDEUTSCH/gian/particle_size_article"
    scatter_dir = os.path.join(save_dir, "scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)
    for fold, (train_index, valid_index) in enumerate(kf.split(X_valid_filtered)):
        print(f"  Fold {fold + 1}/{k_folds}")

        X_train_fold, X_valid_fold = X_valid_filtered[train_index], X_valid_filtered[valid_index]
        y_train_fold, y_valid_fold = y_valid_filtered[train_index], y_valid_filtered[valid_index]

        models = []  # List to store models
        likelihoods = []  # List to store likelihoods

        # Train 15 models with different random seeds
        for model_idx in range(num_models):
            torch.manual_seed(model_idx)  # Set a unique random seed for each model
            model_predictions = []  # Store predictions for each target
            model_predictions_std = []  # Store prediction uncertainties

            for k in range(y_valid_filtered.shape[1]):  # Train separate models for each target
                X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_fold[:, k], dtype=torch.float32)
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPModel(X_train_tensor, y_train_tensor, likelihood)
                model.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=X_train_tensor.shape[1])
                )
                model.covar_module.base_kernel.lengthscale_constraint = gpytorch.constraints.Interval(0.1, 10.0)
                model.covar_module.outputscale_constraint = gpytorch.constraints.Interval(0.1, 10.0)
                likelihood.noise_constraint = gpytorch.constraints.Interval(1e-4, 1.0)
                model.train()
                likelihood.train()
                optimizer = torch.optim.AdamW([
                {'params': model.parameters(), 'weight_decay': 1e-4}  # L2 Regularization
                ], lr=0.005)
                #optimizer = torch.optim.LBFGS(
                #model.parameters(),
                #lr=0.5,
                #max_iter=20,
                #line_search_fn="strong_wolfe"
                #)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                #def closure():
                #    optimizer.zero_grad()
                    #loss = loss_fn(model(x), y)
                #    loss = -mll(output, y_train_tensor)
                #    loss.backward()
                #    return loss
                for epoch in range(150):  # Number of epochs
                    optimizer.zero_grad()
                    output = model(X_train_tensor)
                    loss = -mll(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()

                models.append(model)
                likelihoods.append(likelihood)

        
        predictions_fold = []
        predictions_fold_std = []
        for k in range(y_valid_filtered.shape[1]):  # Validate for each target
                start = k * num_models
                end   = (k + 1) * num_models
                models_k = models[start:end]
                likelihoods_k = likelihoods[start:end]
                X_valid_tensor = torch.tensor(X_valid_fold, dtype=torch.float32)
                predictions, predictions_std = ensemble_predict(models_k, likelihoods_k, X_valid_tensor)
                predictions_fold.append(predictions)
                predictions_fold_std.append(predictions_std)

        # Stack predictions for all targets
        predictions_fold = np.stack(predictions_fold, axis=1)  # Shape: (num_samples, num_targets)
        print('predictions_fold',predictions_fold.shape)
        predictions_fold_std = np.stack(predictions_fold_std, axis=1)
        print('predictions_fold_std',predictions_fold_std.shape)
        # Scale back the predictions and actual values
        y_valid_unscaled = scaler_y.inverse_transform(y_valid_fold)
        predictions_unscaled = scaler_y.inverse_transform(predictions_fold)
        predictions_unscaled_std = (predictions_fold_std)

        # Compute skill metrics (Correlation and RMSE)
        #for k in range(19):  # Loop over targets
        
        for q in range(2):
            correlation_matrix = np.corrcoef(y_valid_unscaled[:,q], predictions_unscaled[:,q])
            skill = correlation_matrix[0, 1]
            skill_gr[fold, j,q] = skill

            rmse = mean_squared_error(y_valid_unscaled[:,q], predictions_unscaled[:,q])
            skill_gr_RMSE[fold, j, q] = rmse

            #y_valid_unscaled2[fold,:,q]=y_valid_unscaled[:,q]
            #predictions_unscaled2[fold,:,q]=predictions_unscaled[:,q]
            
            plt.figure(figsize=(6, 6))
            plt.scatter(
            y_valid_unscaled[:, q],
            predictions_unscaled[:, q],
            s=20, alpha=0.6, edgecolors="k"
            )

            # 1:1 line
            min_val = min(y_valid_unscaled[:, q].min(), predictions_unscaled[:, q].min())
            max_val = max(y_valid_unscaled[:, q].max(), predictions_unscaled[:, q].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            plt.xlabel("Observed (y_valid)", fontsize=14)
            plt.ylabel("Predicted", fontsize=14)
            plt.title(f"Fold {fold} | Var {q} | r={skill:.2f}, RMSE={rmse:.3f}", fontsize=14)
            plt.grid(True, alpha=0.3)

            # Save figure
            outfile = os.path.join(scatter_dir, f"scatter_level{j}_fold{fold}_var{q}.png")
            plt.savefig(outfile, dpi=200, bbox_inches="tight")
            plt.close()
            #scatter_1[fold,:,q]=y_valid_unscaled[:, q]
    

    data_X = io.loadmat('data_X_test_Rf_'+str(j)+'.mat')
    X_test = data_X['X']
    # --- Build target matrix (2 variables) ---
    valid_indices = ~np.isnan(X_test).any(axis=1)
    X_test_filtered = X_test[valid_indices]

    if X_test_filtered.size == 0:
        raise ValueError("Filtered test data has no valid rows. Check for too many NaNs.")

    # Normalize the test data
    X_test_filtered = scaler_X.transform(X_test_filtered)
    X_test_filtered = torch.tensor(X_test_filtered, dtype=torch.float32)
    
    predictions_list = []
    predictions_std_list = []
    # Predict the target variable using the ensemble of models
    for k in range(y_valid_filtered.shape[1]):  # loop over targets (2)
        # get ensemble models for this target
        start_idx = k * num_models
        end_idx = (k + 1) * num_models
        models_k = models[start_idx:end_idx]
        likelihoods_k = likelihoods[start_idx:end_idx]
        # predict
        pred_k, pred_std_k = ensemble_predict(models_k, likelihoods_k, X_test_filtered)

        # reshape to (n_samples, 1)
        predictions_list.append(pred_k.reshape(-1, 1))
        predictions_std_list.append(pred_std_k.reshape(-1, 1))

    # Store predictions for the current depth level
    predictions = np.concatenate(predictions_list, axis=1)
    predictions_std = np.concatenate(predictions_std_list, axis=1)

    # invert scaling
    predictions = scaler_y.inverse_transform(predictions)
    predictions_std = predictions_std * scaler_y.scale_ 
    io.savemat('prediction4_gr_'+str(j)+'.mat', {'predictions3': predictions})
    io.savemat('prediction4_gr_std_'+str(j)+'.mat', {'predictions3_std': predictions_std})

    # Select a trained Gaussian Process model for SHAP analysis
    def model_predict(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_tensor))
            return pred.mean.cpu().numpy()

    def predict_std(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
           pred = likelihood(model(x_tensor))
           return pred.stddev.cpu().numpy()

    # Use a subset of test data as reference
    if j in [1]:
       X_test_np = X_test_filtered.cpu().numpy()
       summary_ref = shap.kmeans(X_test_np, 200)
       #reference_data = X_test_np[np.random.choice(X_test_np.shape[0], 50, replace=False)]
       explainer = shap.KernelExplainer(model_predict, summary_ref)  
       #explainer = shap.KernelExplainer(predict_std, summary_ref) 
       # Create SHAP KernelExplainer
       #explainer = shap.KernelExplainer(model_predict, reference_data)
       #explainer = shap.KernelExplainer(predict_std, reference_data)
       # Compute SHAP values for all test samples
       shap_values = explainer.shap_values(X_test_np)
       # Reshape SHAP values to match (samples, features, targets)
       shap_values = np.array(shap_values)

       # Store SHAP values
       #A_shap[j] = shap_values  # Shape: (35763, 9, 19, 10)
       io.savemat('shap_total2_'+str(j)+'.mat', {'shap_total2':  shap_values})



io.savemat('skill_gr.mat', {'skill_gr': skill_gr})

io.savemat('skill_gr_RMSE.mat', {'skill_gr_RMSE': skill_gr_RMSE})

io.savemat('skill_lin.mat', {'skill_lin': skill_lin})

io.savemat('skill_lin_RMSE.mat', {'skill_lin_RMSE':skill_lin_RMSE})

io.savemat('skill_Rf.mat', {'skill_Rf': skill_Rf})  

io.savemat('skill_Rf_RMSE.mat', {'skill_Rf_RMSE': skill_Rf_RMSE})  

io.savemat('shap_total2.mat', {'shap_total2': shap_total2})

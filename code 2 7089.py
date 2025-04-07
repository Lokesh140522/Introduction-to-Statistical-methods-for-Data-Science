import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from sklearn.model_selection import train_test_split

# Setup
sns.set_theme(style="darkgrid")
sns.set_palette("flare")
os.makedirs("figures", exist_ok=True)

feature_map = {
    "ModelA": lambda x1, x2, n: np.c_[x1**3, x1**5, x2, np.ones(n)],
    "ModelB": lambda x1, x2, n: np.c_[x1, x2, np.ones(n)],
    "ModelC": lambda x1, x2, n: np.c_[x1, x1**2, x1**4, x2, np.ones(n)],
    "ModelD": lambda x1, x2, n: np.c_[x1, x1**2, x1**3, x1**5, x2, np.ones(n)],
    "ModelE": lambda x1, x2, n: np.c_[x1, x1**3, x1**4, x2, np.ones(n)],
}

def fit_polynomial(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    print("\n[Task 2.1] Estimated Parameters (theta):")
    print(theta)
    return theta

def residual_analysis(y_true, y_pred):
    res = y_true - y_pred
    rss = np.sum(res**2)
    print("\n[Task 2.2] Residual Sum of Squares (RSS):")
    print(rss)
    return res, rss

def compute_likelihood_metrics(rss, n):
    sigma_sq = rss / (n - 1)
    log_likelihood = - (n / 2) * np.log(2 * np.pi) - (n / 2) * np.log(sigma_sq) - rss / (2 * sigma_sq)
    print("\n[Task 2.3] Log-Likelihood and Estimated Variance:")
    print("Log-Likelihood:", log_likelihood)
    print("Estimated Variance (sigma^2):", sigma_sq)
    return log_likelihood, sigma_sq

def calc_aic_bic(log_like, k, n):
    AIC = 2 * k - 2 * log_like
    BIC = k * np.log(n) - 2 * log_like
    print("\n[Task 2.4] AIC and BIC Values:")
    print("AIC:", AIC)
    print("BIC:", BIC)
    return AIC, BIC

def show_diagnostics(residuals, title):
    plt.figure()
    sns.histplot(residuals, kde=True, color='mediumvioletred', edgecolor='black')
    plt.title(f"Residual Distribution - {title}")
    plt.savefig(f"figures/{title}_residual_hist.png")
    plt.close()

    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot - {title}")
    plt.savefig(f"figures/{title}_qq_plot.png")
    plt.close()

def run_all_models(x1, x2, y):
    n = len(y)
    results = []

    for name, builder in feature_map.items():
        print("\n============================")
        print(f"Evaluating {name}")
        print("============================")
        X = builder(x1, x2, n)
        theta = fit_polynomial(X, y)
        y_hat = X @ theta
        resids, rss = residual_analysis(y, y_hat)
        logL, var = compute_likelihood_metrics(rss, n)
        aic, bic = calc_aic_bic(logL, X.shape[1], n)
        show_diagnostics(resids, name)

        results.append({
            "Model": name,
            "Parameters": theta,
            "RSS": rss,
            "LogLikelihood": logL,
            "Sigma^2": var,
            "AIC": aic,
            "BIC": bic
        })

    return pd.DataFrame(results)

def task_2_7_best_model(x1, x2, y, model_name):
    print("\n[Task 2.7] Training/Test Evaluation for:", model_name)
    X_full = feature_map[model_name](x1, x2, len(y))
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42)

    theta = fit_polynomial(X_train, y_train)
    y_pred = X_test @ theta
    resids, rss = residual_analysis(y_test, y_pred)
    _, sigma_sq = compute_likelihood_metrics(rss, len(y_test))

    # 95% CI of predictions
    y_std_err = np.sqrt(np.sum((X_test @ np.linalg.pinv(X_train.T @ X_train))**2, axis=1) * sigma_sq)
    ci = 1.96 * y_std_err

    # Plot predictions vs actual with CI
    plt.figure(figsize=(10, 6))
    plt.errorbar(np.arange(len(y_test)), y_pred, yerr=ci, fmt='o', label='Prediction ±95% CI')
    plt.plot(np.arange(len(y_test)), y_test, 'rx', label='Actual')
    plt.legend()
    plt.title(f"Test Predictions with 95% CI - {model_name}")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Output")
    plt.grid(True)
    plt.savefig(f"figures/{model_name}_test_predictions_CI.png")
    plt.close()

# Example Usage:
X_df = pd.read_csv("../data/X.csv")
y_df = pd.read_csv("../data/y.csv")
data = pd.concat([X_df, y_df], axis=1)
summary_df = run_all_models(data['x1'].values, data['x2'].values, data['y'].values)
best_model_name = summary_df.sort_values('AIC').iloc[0]['Model']
task_2_7_best_model(data['x1'].values, data['x2'].values, data['y'].values, best_model_name)

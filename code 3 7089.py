import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid")
sns.set_palette("flare")
os.makedirs("figures", exist_ok=True)

# Load signal data
def load_signal_data():
    X = pd.read_csv("../data/X.csv")
    y = pd.read_csv("../data/y.csv")
    return pd.concat([X, y], axis=1)

# Define design matrix
def define_design_matrix(label, x1, x2, n):
    matrices = {
        "A": np.c_[x1**3, x1**5, x2, np.ones(n)],
        "B": np.c_[x1, x2, np.ones(n)],
        "C": np.c_[x1, x1**2, x1**4, x2, np.ones(n)],
        "D": np.c_[x1, x1**2, x1**3, x1**5, x2, np.ones(n)],
        "E": np.c_[x1, x1**3, x1**4, x2, np.ones(n)],
    }
    return matrices[label]

# Uniform sampling
def uniform_sample(center, scale, size):
    spread = scale * abs(center)
    return np.random.uniform(center - spread, center + spread, size)

# Rejection ABC
def rejection_abc(y, X, theta, idx1, idx2, sample_size=10000, accept_count=500, threshold=5e4):
    fixed = theta.copy()
    s1 = uniform_sample(theta[idx1], 0.2, sample_size)
    s2 = uniform_sample(theta[idx2], 0.2, sample_size)

    acc_1, acc_2 = [], []
    for i in range(sample_size):
        trial = fixed.copy()
        trial[idx1] = s1[i]
        trial[idx2] = s2[i]
        y_sim = X @ trial
        err = np.sum((y - y_sim) ** 2)
        if err < threshold:
            acc_1.append(s1[i])
            acc_2.append(s2[i])
        if len(acc_1) >= accept_count:
            break
    print(f"[Rejection ABC] Accepted Samples: {len(acc_1)}")
    return np.array(acc_1), np.array(acc_2)

# Plotting posterior distributions
def plot_marginals_joint(a1, a2, i1, i2):
    plt.figure()
    sns.histplot(a1, kde=True, color='deeppink', edgecolor='black')
    plt.title(f"Posterior of Theta[{i1}]")
    plt.xlabel(f"Theta[{i1}]")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(f"figures/posterior_theta_{i1}.png")
    plt.close()

    plt.figure()
    sns.histplot(a2, kde=True, color='mediumturquoise', edgecolor='black')
    plt.title(f"Posterior of Theta[{i2}]")
    plt.xlabel(f"Theta[{i2}]")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(f"figures/posterior_theta_{i2}.png")
    plt.close()

    plt.figure()
    sns.kdeplot(x=a1, y=a2, fill=True, cmap="flare")
    plt.title("Joint Posterior")
    plt.xlabel(f"Theta[{i1}]")
    plt.ylabel(f"Theta[{i2}]")
    plt.grid(True)
    plt.savefig("figures/joint_posterior.png")
    plt.close()

# Execution pipeline
data = load_signal_data()
x1, x2, y = data['x1'].values, data['x2'].values, data['y'].values
n = len(y)

label = "C"
theta = np.array([8.55216135, 6.24691708, -0.28296309, 4.15988326, 10.17369609])  # From Task 2.1
idxs = np.argsort(np.abs(theta))[-2:]  # Indices of 2 largest magnitude parameters

X = define_design_matrix(label, x1, x2, n)
a1, a2 = rejection_abc(y, X, theta, idxs[1], idxs[0])  # idxs[1] is larger one

plot_marginals_joint(a1, a2, idxs[1], idxs[0])

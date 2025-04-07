import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_palette("flare")

class MEGEDA:
    def __init__(self, time_path, x_path, y_path):
        self.df = self._load_and_prepare(time_path, x_path, y_path)

    def _load_and_prepare(self, time_path, x_path, y_path):
        time = pd.read_csv(time_path)
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        df = pd.concat([time, X, y], axis=1)
        df['x2'] = df['x2'].map({0: 'Neutral', 1: 'Emotional'})
        return df

    def plot_time_series(self):
        melted = pd.melt(self.df, id_vars=['time'], value_vars=['x1', 'y'], var_name='Signal', value_name='Value')
        sns.relplot(data=melted, x='time', y='Value', col='Signal', kind='line', height=4, aspect=2)
        plt.suptitle("Time Series of x1 and y", y=1.05)
        plt.savefig("../figure/time_series.png")

        plt.show()

    def plot_kde(self):
        sns.kdeplot(self.df['x1'], fill=True, label='x1', color='mediumorchid')
        sns.kdeplot(self.df['y'], fill=True, label='y', color='darkcyan')
        plt.legend()
        plt.title("KDE Plot for x1 and y")
        plt.grid(True)
        plt.savefig("../figure/kde.png")
        plt.show()

    def plot_pairplot(self):
        sns.pairplot(self.df[['x1', 'y']], diag_kind='kde', plot_kws={'color': 'sienna'})
        plt.suptitle("Pairplot: x1 vs y", y=1.02)
        plt.savefig("../figure/pairplot.png")
        plt.show()

    def show_correlation(self):
        corr = self.df['x1'].corr(self.df['y'])
        print("Pearson Correlation (x1 vs y):", round(corr, 4))

    def plot_boxplot_by_category(self):
        sns.boxplot(x='x2', y='y', data=self.df, palette='flare')
        plt.title("Brain Response by Audio Category")
        plt.xlabel("Audio Type")
        plt.ylabel("MEG Signal (y)")
        plt.grid(True)
        plt.savefig("../figure/boxplot.png")

        plt.show()

    def analyze_by_category(self):
        for name, subdf in self.df.groupby('x2'):
            sns.scatterplot(data=subdf, x='x1', y='y', color='sienna', alpha=0.6)
            plt.title(f'Scatter of x1 vs y - {name} Voice')
            plt.xlabel("x1")
            plt.ylabel("y")
            plt.grid(True)
            plt.savefig(f"../figure/scatter_{name}.png")

            sns.histplot(subdf['y'], kde=True, color='lightslategray', edgecolor='black')
            plt.title(f'Distribution of y for {name} Voice')
            plt.xlabel("y")
            plt.grid(True)
            plt.savefig(f"../figure/distribution_{name}.png")

# Run analysis
if __name__ == "__main__":
    analysis = MEGEDA("../data/time.csv", "../data/X.csv", "../data/y.csv")
    analysis.plot_time_series()
    analysis.plot_kde()
    analysis.plot_pairplot()
    analysis.show_correlation()
    analysis.plot_boxplot_by_category()
    analysis.analyze_by_category()

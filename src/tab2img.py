import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import utils.consts as consts
from utils.noise_generator import NoiseGenerator


def plot_hists_real_noisy_var(df):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharey=True)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Real vs Noisy", fontsize=14, y=0.95)

    tickers = df.columns.values

    for ticker, ax in zip(tickers, axs.ravel()):
        df_counts = df.loc[:, ticker].value_counts().reset_index()
        sns.barplot(x='index', y=ticker, data=df_counts, ax=ax)
        ax.bar_label(ax.containers[0])
        ax.set_title(ticker)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.show()


seed_value = 3232

df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA, 'others', 'bbdd_steno.csv')), sep=',')
print(df_data)

v_sex = df_data.loc[:, 'sex'].values.reshape(-1, 1)

v_sex_noisy1 = NoiseGenerator().corrupt_input('masking', v_sex, fraction_noise=10, seed_value=seed_value)
v_sex_noisy2 = NoiseGenerator().corrupt_input('masking', v_sex, fraction_noise=20, seed_value=seed_value)
v_sex_noisy3 = NoiseGenerator().corrupt_input('masking', v_sex, fraction_noise=30, seed_value=seed_value)

m_real_noisy = np.hstack((v_sex, v_sex_noisy1, v_sex_noisy2, v_sex_noisy3))
df_real_noisy = pd.DataFrame(m_real_noisy)

print(df_real_noisy)

plot_hists_real_noisy_var(df_real_noisy)



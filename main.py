import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from functions import plot_intensities_wavelenght, check_wavenumbers, find_groups, average_per_group, plot_curves_by_species, lda_1D_visualize, plot_lda_score_histogram


PLOT_CURVES=True
PLOT_LDA_DENSITY=True
PLOT_LDA_HISTOGRAM=True
LOGY=False
R=0.85 # 0.70, 0.85 provoca separação perfeita com LDA; 0.60 dá sobreposição; 0.90 também; usando min como função de agregação; com min, mean e max separa para R=0.85
def agg_function(x): return np.mean(x) #np.abs(np.max(x)-np.min(x))

# Load data with error handling
try:
    df = pd.read_csv('Data_species.csv')
except FileNotFoundError:
    raise SystemExit("Error: Data_species.csv not found in current directory")

if check_wavenumbers(df):
    print("All samples have identical wavenumber sequences")

# Assuming df is your DataFrame loaded from Data_species.csv
df['wavelength_str'] = df['wavenumber'].round().astype(int).astype(str)

# Extract unique sample-species mapping from the original DataFrame
sample_species = df[['sample', 'species']].drop_duplicates().set_index('sample')

# Pivot the DataFrame: 'sample' is the index
pivot_df = df.pivot(index='sample', columns='wavelength_str', values='intensity')

# Sort the columns by wavelength (convert column names to int for sorting)
sorted_cols = sorted(pivot_df.columns, key=lambda x: int(x))
pivot_df = pivot_df[sorted_cols]

# corr matrix
corr=pd.DataFrame.corr(pivot_df)

# adjacency matrix
adjacency=np.abs(corr) > R

# groups
groups=find_groups(adjacency)
print([g[0] for g in groups])
print('number of groups', len(groups))

# compute group averages
df=average_per_group(pivot_df,groups,agg_function) # 'sample' is the index

if PLOT_CURVES: plot_curves_by_species(df,sample_species, LOGY,R)

if PLOT_LDA_DENSITY: df['lda_score']= lda_1D_visualize(df,sample_species, PLOT=PLOT_LDA_DENSITY)      # get labels from species for LDA

# add scores and species to df
df = df.join(sample_species)
print(df.sort_values(by=['species']))

if PLOT_LDA_HISTOGRAM: plot_lda_score_histogram(df)

#plot_umap(df,sample_species)
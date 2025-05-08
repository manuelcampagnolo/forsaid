import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
#import umap

def plot_lda_score_histogram(df):
    """
    Plot histogram of 'lda_score' distributions grouped by 'species'.
    
    Args:
        df: DataFrame with columns 'lda_score' (numeric) and 'species' (categorical or string)
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='lda_score', hue='species', element='step', stat='density', common_norm=False, bins=30, alpha=0.6)
    plt.xlabel('LDA Score')
    plt.ylabel('Density')
    plt.title('Distribution of LDA Scores by Species')
    plt.legend(title='Species')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_umap(df, sample_species):
    X=df.values
    df = df.join(sample_species) # If 'sample' is an index 
    y= df['species'].astype('category').cat.codes
    embedding = umap.UMAP().fit_transform(X, y)
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*embedding.T, s=0.1, c=y, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    #cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    #cbar.set_ticks(np.arange(10))
    #species_list = df['species'].unique()
    #cbar.set_ticklabels(species_list)
    plt.title('UMAP using Labels')
    plt.show()


def lda_1D_visualize(df,  sample_species, PLOT):
    """Perform LDA and plot samples on discriminant plane.
    
    Args:
        features_df: DataFrame with samples as rows, features as columns
        sample_species: where to get text labels for each sample
    """
    X=df.copy()
    df = df.join(sample_species) # If 'sample' is an index 
    y= df['species'].astype('category').cat.codes
    class_names = df['species'].astype('category').cat.categories
    
    # Fit LDA with 1 component
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_proj = lda.fit_transform(X, y).ravel()  # Flatten to 1D array
    
    if PLOT:
        # Prepare plot
        plt.figure(figsize=(8, 6))
        
        # Plot KDE per class
        for class_code, class_name in enumerate(class_names):
            class_proj = X_proj[y == class_code]
            sns.kdeplot(class_proj, label=f'{class_name} density', fill=True, alpha=0.4)
            
            # Scatter points with vertical jitter for visibility
            jitter = np.random.uniform(-0.02, 0.02, size=class_proj.shape)
            plt.scatter(class_proj, jitter + class_code * 0.1, alpha=0.7, label=f'{class_name} samples')
        
        plt.xlabel('Linear Discriminant 1')
        plt.yticks([])  # Hide y-axis ticks as they have no meaning here
        plt.title('LDA 1D Projection with Class Densities')
        plt.legend()
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()
    # Return projection scores as Series with original index
    return pd.Series(X_proj, index=X.index, name='LDA_1D_score')


def plot_curves_by_species(df,sample_species, LOGY,R):
    # Merge species info into pivot_df (which has 'sample' as index or column)
    # If 'sample' is an index in pivot_df:
    pivot_df = df.join(sample_species)
    # Optional: reset index if you want 'sample' as a column
    pivot_df.reset_index(inplace=True)
    # plot
    plot_intensities_wavelenght(pivot_df, LOGY,R)

def average_per_group(df,L,agg_function):
    # df: original DataFrame with samples as rows and wavelengths (strings) as columns
    # L: list of groups, each group is a list of wavelength strings
    # Prepare a dictionary to hold averaged columns
    group_averages = {}
    for i, group in enumerate(L):
        # Select columns by index and compute row-wise mean
        group_averages[str(i)] = df.iloc[:, group].apply(agg_function,axis=1) # mean
    # Create new DataFrame from the averaged columns
    return pd.DataFrame(group_averages, index=df.index)


def find_groups(adjacency_matrix):
    groups = []
    n = adjacency_matrix.shape[0]
    start = 0
    while start < n:
        end = start + 1
        while end < n and np.all(adjacency_matrix.iloc[start:end+1, start:end+1]):
            end += 1
        #print(start, end)
        groups.append(list(range(start, end)))
        start = end
    return groups


def check_wavenumbers(df):
    # Process wavenumbers
    df['rounded'] = df['wavenumber'].round(1)
    grouped = df.groupby('sample')['rounded'].apply(list)
    # Validate sequences
    first_seq = grouped.iloc[0]
    for sample, seq in grouped.items():
        if seq != first_seq:
            raise SystemExit(f"Mismatch found:\nSample1: {first_seq}\nSample2: {seq}")

def plot_intensities_wavelenght(pivot_df,LOGY,R):
    # Extract wavelength columns (exclude 'sample' and 'species')
    exclude_cols = {'sample', 'species'}
    wavelength_cols = [col for col in pivot_df.columns if col not in exclude_cols]

    # Convert wavelength column names to integers for x-axis
    wavelengths = list(map(int, wavelength_cols))

    # Get unique species and assign colors
    species_list = pivot_df['species'].unique()
    colors = cm.get_cmap('RdBu', len(species_list))  # You can choose other colormaps
    species_color_map = {species: colors(i) for i, species in enumerate(species_list)}

    plt.figure(figsize=(10, 6))

    # Plot samples, colored by species
    for _, row in pivot_df.iterrows():
        species = row['species']
        intensities = row[wavelength_cols].values
        plt.plot(wavelengths, intensities, color=species_color_map[species], alpha=0.7)

    # Create legend handles manually (one per species)
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=species_color_map[sp], lw=2, label=sp) for sp in species_list]
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Intensity vs Wavelength: R={R}; {len(wavelength_cols)} groups')
    if LOGY: plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    plt.legend(handles=legend_handles, title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
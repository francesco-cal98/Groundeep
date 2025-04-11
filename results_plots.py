import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# File paths - replace these with your actual file paths
file1 = "dataset1.xlsx"  # Path to your first excel file
file2 = "dataset2.xlsx"  # Path to your second excel file

# Function to load data from Excel files
def load_data_from_excel(file1, file2):
    """
    Load data from two Excel files and prepare it for plotting.
    
    Parameters:
    file1 (str): Path to the first Excel file
    file2 (str): Path to the second Excel file
    
    Returns:
    tuple: (df1, df2, combined_df) - The processed dataframes
    """
    print(f"Loading data from {file1} and {file2}...")
    
    # Load data from Excel files
    try:
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
        
        # Add dataset identifiers
        df1['Dataset'] = 'Dataset 1'
        df2['Dataset'] = 'Dataset 2'
        
        # Extract X and Y parameters from network names
        df1 = extract_params(df1)
        df2 = extract_params(df2)
        
        # Combine datasets for easier plotting
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        print("Data loaded successfully!")
        return df1, df2, combined_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# Extract X and Y parameters from network names
def extract_params(df):
    """
    Extract the numerical parameters from network names.
    
    Parameters:
    df (DataFrame): DataFrame containing a 'Network Name' column
    
    Returns:
    DataFrame: The input DataFrame with added 'Param_X' and 'Param_Y' columns
    """
    # Create a copy to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Extract the X and Y parameters using regex
    params = df['Network Name'].str.extract(r'idbn_trained_uniform_(\d+)_(\d+)')
    df['Param_X'] = params[0].astype(int)
    df['Param_Y'] = params[1].astype(int)
    
    return df

# Set a consistent style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e']  # First color for Dataset 1, second for Dataset 2

# Create output directory if it doesn't exist
def ensure_output_dir(output_dir="network_comparison_plots"):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# 1. Side-by-side Bar Charts for Accuracy and Weber Fraction
def plot_side_by_side_bars(df1, df2, metric, title, output_dir):
    """
    Create side-by-side bar charts comparing a specific metric across networks.
    
    Parameters:
    df1, df2 (DataFrame): DataFrames containing network data
    metric (str): The metric to compare (e.g., 'Accuracy', 'Weber Fraction')
    title (str): The title for the plot
    output_dir (str): Directory to save the output image
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    plt.figure(figsize=(15, 7))
    
    # Sort data by parameters for better comparison
    df1_sorted = df1.sort_values(by=['Param_X', 'Param_Y'])
    df2_sorted = df2.sort_values(by=['Param_X', 'Param_Y'])
    
    bar_width = 0.35
    indices = np.arange(len(df1_sorted))
    
    # Create simple network labels (X_Y format)
    labels = [f"{x}_{y}" for x, y in zip(df1_sorted['Param_X'], df1_sorted['Param_Y'])]
    
    plt.bar(indices - bar_width/2, df1_sorted[metric], bar_width, 
            label=f'Dataset 1', color=colors[0], alpha=0.8)
    plt.bar(indices + bar_width/2, df2_sorted[metric], bar_width, 
            label=f'Dataset 2', color=colors[1], alpha=0.8)
    
    plt.xlabel('Network Configuration (X_Y Parameters)')
    plt.ylabel(metric)
    plt.title(f'{title} by Network Configuration')
    plt.xticks(indices, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    filename = f"{metric.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return plt

# 2. Scatter Plots with Network Size Parameters
def scatter_comparison(combined_df, x_metric, y_metric, title, output_dir):
    """
    Create scatter plots comparing two metrics across datasets.
    
    Parameters:
    combined_df (DataFrame): Combined DataFrame with both datasets
    x_metric, y_metric (str): The metrics to plot on x and y axes
    title (str): The title for the plot
    output_dir (str): Directory to save the output image
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, dataset in enumerate(['Dataset 1', 'Dataset 2']):
        subset = combined_df[combined_df['Dataset'] == dataset]
        scatter = plt.scatter(
            subset[x_metric], 
            subset[y_metric],
            c=[colors[i]],
            s=subset['Param_X'] / 20 + subset['Param_Y'] / 20,  # Size based on parameters
            label=dataset,
            alpha=0.7
        )
        
        # Add annotations for key points
        if x_metric == 'Beta Number' and y_metric == 'Accuracy':
            for j, row in subset.iterrows():
                plt.annotate(
                    f"{row['Param_X']}_{row['Param_Y']}", 
                    (row[x_metric], row[y_metric]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
    
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    filename = f"{x_metric.lower().replace(' ', '_')}_{y_metric.lower().replace(' ', '_')}_scatter.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return plt

# 3. Parameter Impact Matrix (heatmaps)
def parameter_heatmaps(df, metric, title, output_dir):
    """
    Create heatmaps showing the impact of parameters on a specific metric.
    
    Parameters:
    df (DataFrame): DataFrame containing network data
    metric (str): The metric to visualize (e.g., 'Accuracy')
    title (str): The title for the plot
    output_dir (str): Directory to save the output image
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    unique_x = sorted(df['Param_X'].unique())
    unique_y = sorted(df['Param_Y'].unique())
    
    # Create a matrix for the heatmap
    matrix = np.zeros((len(unique_x), len(unique_y)))
    
    # Fill the matrix
    for i, x in enumerate(unique_x):
        for j, y in enumerate(unique_y):
            subset = df[(df['Param_X'] == x) & (df['Param_Y'] == y)]
            if not subset.empty:
                matrix[i, j] = subset[metric].values[0]
            else:
                matrix[i, j] = np.nan  # Use NaN for missing values
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Create custom colormap from blue to red
    cmap = LinearSegmentedColormap.from_list('blue_to_red', ['#1f77b4', '#ff7f0e'])
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(unique_y)))
    ax.set_yticks(np.arange(len(unique_x)))
    ax.set_xticklabels(unique_y)
    ax.set_yticklabels(unique_x)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(metric)
    
    # Add text annotations
    for i in range(len(unique_x)):
        for j in range(len(unique_y)):
            if not np.isnan(matrix[i, j]) and matrix[i, j] != 0:
                text_color = "white" if 0.3 < matrix[i, j] > 0.85 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", color=text_color)
    
    ax.set_xlabel('Parameter Y')
    ax.set_ylabel('Parameter X')
    plt.title(f'{title} - {metric}')
    plt.tight_layout()
    
    # Save the figure
    dataset_name = title.lower().replace(' ', '_')
    filename = f"{dataset_name}_{metric.lower().replace(' ', '_')}_heatmap.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return plt

# 4. Radar/Spider Chart for comparing multiple metrics
def radar_chart(df1, df2, network_name, output_dir):
    """
    Create radar charts comparing multiple metrics for a specific network.
    
    Parameters:
    df1, df2 (DataFrame): DataFrames containing network data
    network_name (str): The name of the network to visualize
    output_dir (str): Directory to save the output image
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    # Check if the network exists in both datasets
    if network_name not in df1['Network Name'].values or network_name not in df2['Network Name'].values:
        print(f"Network {network_name} not found in both datasets. Skipping radar chart.")
        return None
    
    df1_network = df1[df1['Network Name'] == network_name].iloc[0]
    df2_network = df2[df2['Network Name'] == network_name].iloc[0]
    
    # Combine datasets for normalization
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Select metrics for radar
    metrics = ['Beta Number', 'Beta Size', 'Beta Spacing', 'Weber Fraction', 'Accuracy']
    
    # Normalize the values for better visualization
    min_vals = combined_df[metrics].min()
    max_vals = combined_df[metrics].max()
    df1_normalized = (df1_network[metrics] - min_vals) / (max_vals - min_vals)
    df2_normalized = (df2_network[metrics] - min_vals) / (max_vals - min_vals)
    
    # Setup the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Add the first dataset
    values1 = df1_normalized.values.tolist()
    values1 += values1[:1]  # Complete the circle
    ax.plot(angles, values1, color=colors[0], linewidth=2, label='Dataset 1')
    ax.fill(angles, values1, color=colors[0], alpha=0.25)
    
    # Add the second dataset
    values2 = df2_normalized.values.tolist()
    values2 += values2[:1]  # Complete the circle
    ax.plot(angles, values2, color=colors[1], linewidth=2, label='Dataset 2')
    ax.fill(angles, values2, color=colors[1], alpha=0.25)
    
    # Set the angular labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add a legend and title
    ax.legend(loc='upper right')
    plt.title(f'Comparison of Metrics for {network_name.split(".")[0]}')
    
    # Save the figure
    network_id = network_name.split('.')[0].replace('idbn_trained_uniform_', '')
    filename = f"radar_chart_{network_id}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return plt

# 5. Multi-metric Bar Chart
def multi_metric_bar_chart(df1, df2, output_dir):
    """
    Create a bar chart comparing average metric values across datasets.
    
    Parameters:
    df1, df2 (DataFrame): DataFrames containing network data
    output_dir (str): Directory to save the output image
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    metrics = ['Beta Number', 'Beta Size', 'Beta Spacing', 'Weber Fraction', 'Accuracy']
    
    df1_means = df1[metrics].mean()
    df2_means = df2[metrics].mean()
    
    # Calculate standard deviations for error bars
    df1_std = df1[metrics].std()
    df2_std = df2[metrics].std()
    
    # Set positions for bars
    bar_width = 0.35
    indices = np.arange(len(metrics))
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    plt.bar(indices - bar_width/2, df1_means, 
            width=bar_width, 
            label='Dataset 1', 
            color=colors[0],
            yerr=df1_std,
            capsize=5)
    
    plt.bar(indices + bar_width/2, df2_means, 
            width=bar_width, 
            label='Dataset 2', 
            color=colors[1],
            yerr=df2_std,
            capsize=5)
    
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title('Comparison of Average Metric Values Between Datasets')
    plt.xticks(indices, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    filename = "multi_metric_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return plt

# 6. Parallel Coordinates Plot
def parallel_coordinates_plot(df1, df2, output_dir):
    """
    Create a parallel coordinates plot to visualize multiple metrics.
    
    Parameters:
    df1, df2 (DataFrame): DataFrames containing network data
    output_dir (str): Directory to save the output image
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    # Combine datasets for plotting
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    metrics = ['Beta Number', 'Beta Size', 'Beta Spacing', 'Weber Fraction', 'Accuracy']
    
    # Normalize data for better visualization
    normalized_df = combined_df.copy()
    
    # Normalize metrics
    for metric in metrics:
        min_val = combined_df[metric].min()
        max_val = combined_df[metric].max()
        normalized_df[metric] = (combined_df[metric] - min_val) / (max_val - min_val)
    
    plt.figure(figsize=(12, 8))
    
    # Create x positions for the metrics
    x = list(range(len(metrics)))
    
    # Plot lines for each network
    for i, dataset in enumerate(['Dataset 1', 'Dataset 2']):
        subset = normalized_df[normalized_df['Dataset'] == dataset]
        
        for _, row in subset.iterrows():
            y = [row[metric] for metric in metrics]
            plt.plot(x, y, color=colors[i], alpha=0.3)
        
        # Add a thicker line for dataset average
        avg_values = [subset[metric].mean() for metric in metrics]
        plt.plot(x, avg_values, color=colors[i], linewidth=4, alpha=0.8, label=f'{dataset} Avg')
    
    # Customize the plot
    plt.xticks(x, metrics, rotation=45)
    plt.ylabel('Normalized Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Parallel Coordinates Plot of Network Metrics')
    plt.tight_layout()
    
    # Save the figure
    filename = "parallel_coordinates.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return plt

# 7. Correlation heatmap
def correlation_heatmap(df1, df2, output_dir):
    """
    Create correlation heatmaps for both datasets.
    
    Parameters:
    df1, df2 (DataFrame): DataFrames containing network data
    output_dir (str): Directory to save the output image
    
    Returns:
    tuple: (fig1, fig2) - The generated plots
    """
    metrics = ['Beta Number', 'Beta Size', 'Beta Spacing', 'Weber Fraction', 'Accuracy']
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Dataset 1 correlation heatmap
    corr1 = df1[metrics].corr()
    sns.heatmap(corr1, annot=True, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Dataset 1 - Correlation Matrix')
    
    # Dataset 2 correlation heatmap
    corr2 = df2[metrics].corr()
    sns.heatmap(corr2, annot=True, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Dataset 2 - Correlation Matrix')
    
    plt.tight_layout()
    
    # Save the figure
    filename = "correlation_heatmaps.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    
    return fig

# Main function to generate all plots
def generate_all_plots(file1, file2, output_dir="network_comparison_plots"):
    """
    Generate and save all comparison plots.
    
    Parameters:
    file1, file2 (str): Paths to the Excel files
    output_dir (str): Directory to save output images
    
    Returns:
    bool: True if successful, False otherwise
    """
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Load data
    df1, df2, combined_df = load_data_from_excel(file1, file2)
    
    if df1 is None or df2 is None or combined_df is None:
        return False
    
    print(f"Generating plots and saving to {output_dir}...")
    
    # Generate 1. Side-by-side bar charts
    print("Generating bar charts...")
    plot_side_by_side_bars(df1, df2, 'Accuracy', 'Accuracy Comparison', output_dir)
    plot_side_by_side_bars(df1, df2, 'Weber Fraction', 'Weber Fraction Comparison', output_dir)
    
    # Generate 2. Scatter plots
    print("Generating scatter plots...")
    scatter_comparison(combined_df, 'Beta Number', 'Accuracy', 
                      'Relationship Between Beta Number and Accuracy', output_dir)
    scatter_comparison(combined_df, 'Weber Fraction', 'Accuracy', 
                      'Relationship Between Weber Fraction and Accuracy', output_dir)
    
    # Generate 3. Parameter heatmaps
    print("Generating heatmaps...")
    parameter_heatmaps(df1, 'Accuracy', 'Dataset 1', output_dir)
    parameter_heatmaps(df2, 'Accuracy', 'Dataset 2', output_dir)
    
    # Generate 4. Radar chart for sample networks
    print("Generating radar charts...")
    # Try a few networks to ensure at least one works
    for network in df1['Network Name'].iloc[:3]:
        radar_chart(df1, df2, network, output_dir)
    
    # Generate 5. Multi-metric bar chart
    print("Generating multi-metric comparison...")
    multi_metric_bar_chart(df1, df2, output_dir)
    
    # Generate 6. Parallel coordinates plot
    print("Generating parallel coordinates plot...")
    parallel_coordinates_plot(df1, df2, output_dir)
    
    # Generate 7. Correlation heatmap
    print("Generating correlation heatmaps...")
    correlation_heatmap(df1, df2, output_dir)
    
    print(f"All plots have been generated and saved to {output_dir}!")
    return True

# Execute the plotting functions
if __name__ == "__main__":
    # Replace these with your actual file paths
    file1 = "/home/student/Desktop/Groundeep/model_coefficients_results_all_uniform.xlsx"  # Path to your first Excel file
    file2 = "/home/student/Desktop/Groundeep/model_coefficients_results_all_zipfian.xlsx"  # Path to your second Excel file
    
    # Generate all plots
    success = generate_all_plots(file1, file2)
    
    if success:
        print("Script completed successfully!")
    else:
        print("Script failed. Please check the error messages above.")
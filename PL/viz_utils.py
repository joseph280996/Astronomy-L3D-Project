import matplotlib.pyplot as plt

def plot_training_metrics(training_history, figsize=(10, 6)):
    """
    Plot MixMatch training metrics (losses and accuracies)
    
    Args:
        training_history: Dictionary containing training metrics
        figsize: Figure size
    Returns:
        matplotlib figure
    """
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot epochs
    epochs = range(1, len(training_history['epochs']) + 1)
    
    # Plot training loss on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, training_history['tr']['loss'], color=color, linestyle='-', 
             label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    
    ax1.plot(epochs, training_history['va']['loss'], color=color, 
             linestyle='--', label='Validation Loss')
    
    # Add title with final metrics
    plt.title(
        f'Training Metrics'
    )
    plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig
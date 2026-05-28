import numpy as np
import matplotlib.pyplot as plt


#============================================================================
#=================== PLOTTING CIRCLE PREDICTION RESULTS ============================

def decision_plot_circle(x, y, preds):
    fig, ax = plt.subplots(1,2, figsize =(10,5 ))
    theta_circle = np.linspace(0, 2*np.pi, 500)
    r = np.sqrt(2/np.pi)
    ax[0].plot(r*np.cos(theta_circle), r*np.sin(theta_circle),'k-', label = 'True circle',linewidth=2)
    ax[0].set_xlim(-1,1)
    ax[0].set_ylim(-1,1)

    ax[0].scatter(x[:, 0], x[:, 1], c= preds, cmap='spring',s=20, alpha=0.6, zorder=2)

    labels = []
    for i in range(len(x)):
        if preds[i] == y[i]:
            label = 'Correct classified' if 'Correct classified' not in labels else None
            ax[1].scatter(x[i, 0], x[i, 1], c= 'g',s=20, label= label,alpha=0.6, zorder=2)
            labels.append(label)
        else:
            label = 'Uncorrect classified' if 'Uncorrect classified' not in labels else None
            ax[1].scatter(x[i, 0], x[i, 1], c= 'r',s=20, label= label,alpha=0.6, zorder=2)
            labels.append(label)
        
    ax[1].legend()
    plt.tight_layout()
    plt.show()

#============================================================================
#=================== PLOTTING WAVY LINES PREDICTION RESULTS ============================

def decision_plot_wavy(x, y, preds, freq=1):
    """
    Args:
        x: points coordinates (shape: N, 2 ou 3)
        y: true labels (shape: N,)
        preds: model predictions (shape: N,)
        freq: frequency of the waves
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # extract 2D coordinates (if x has more than 2 features, we take only the first two for plotting)
    x_2d = x[:, :2] if x.shape[1] >= 2 else x
    
    
    scatter1 = ax[0].scatter(x_2d[:, 0], x_2d[:, 1], c=preds, cmap='spring', 
                            s=30, alpha=0.6, zorder=2, linewidth=0.5)
    
    # Plotting real decision boundaries
    s = np.linspace(-1, 1, 500)
    ax[0].plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-', linewidth=2.5)
    ax[0].plot(s, -s + np.sin(freq * np.pi * s), 'k-', linewidth=2.5)
    
    ax[0].set_xlim(-1.1, 1.1)
    ax[0].set_ylim(-1.1, 1.1)
    ax[0].set_aspect('equal')
    ax[0].grid(alpha=0.3)
    ax[0].set_xlabel('$x_0$', fontsize=12, fontweight='bold')
    ax[0].set_ylabel('$x_1$', fontsize=12, fontweight='bold')
    ax[0].legend(fontsize=10)
    cbar1 = plt.colorbar(scatter1, ax=ax[0])
    cbar1.set_label('Predicted Class', fontsize=10)
    

    correct_mask = preds == y
    
    # Plotting corrects (verde)
    ax[1].scatter(x_2d[correct_mask, 0], x_2d[correct_mask, 1], 
                 c='green', s=50, alpha=0.6, label='Correct classified', 
                 zorder=2, marker='o', edgecolors='darkgreen', linewidth=1)
    
    # Plotting incorrects (vermelho)
    ax[1].scatter(x_2d[~correct_mask, 0], x_2d[~correct_mask, 1], 
                 c='red', s=100, alpha=0.6, label='Incorrect classified', 
                 zorder=2, marker='X', edgecolors='darkred', linewidth=1)
    
    # Plotting real decision boundaries
    ax[1].plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-', linewidth=2.5)
    ax[1].plot(s, -s + np.sin(freq * np.pi * s), 'k-', linewidth=2.5)
    
    ax[1].set_xlim(-1.1, 1.1)
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].set_aspect('equal')
    ax[1].grid(alpha=0.3)
    ax[1].set_xlabel('$x_0$', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('$x_1$', fontsize=12, fontweight='bold')
    

    plt.tight_layout()
    plt.show()
    
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt

def decision_plot_circle(x, y, preds):
    fig, ax = plt.subplots(1,2, figsize =(10,5 ))
    theta_circle = np.linspace(0, 2*np.pi, 500)
    r = np.sqrt(2/np.pi)
    ax[0].plot(r*np.cos(theta_circle), r*np.sin(theta_circle),'k-', label = 'True circle')
    ax[0].set_xlim(-1,1)
    ax[0].set_ylim(-1,1)

    ax[0].scatter(x[:, 0], x[:, 1], c= preds, cmap='bwr', edgecolors='k',s=20)

    labels = []
    for i in range(len(x)):
        if preds[i] == y[i]:
            label = 'Correct classified' if 'Correct classified' not in labels else None
            ax[1].scatter(x[i, 0], x[i, 1], c= 'g', edgecolors='k',s=20, label= label)
            labels.append(label)
        else:
            label = 'Uncorrect classified' if 'Uncorrect classified' not in labels else None
            ax[1].scatter(x[i, 0], x[i, 1], c= 'r', edgecolors='k',s=20, label= label)
            labels.append(label)
        
    ax[1].legend()
    plt.tight_layout()
    plt.show()
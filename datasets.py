import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_iris_binary(task_type="setosa_vs_versicolor", test_size=0.2, random_state=42):
    
#Binary classification in quantum circuits with load and preprocess the Iris dataset
#task_type (str): "setosa_vs_versicolor" or "versicolor_vs_virginica"
#test_size (float): Proportion of the dataset to include in the test split
#random_state (int): Random state for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    
    #Load Data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    #Binary Classification (0:setosa,1:versicolor,2:virginica)
    if task_type == "setosa_vs_versicolor":
        mask = (y == 0) | (y == 1)
    elif task_type == "versicolor_vs_virginica":
        mask = (y == 1) | (y == 2)
    else:
        raise ValueError("Please provide a valid task_type.")
        
    X = X[mask] #data matrices
    y = y[mask] #answersheet
    
    #Relabel to -1 and 1 (For quantum expectation values)
    if task_type == "setosa_vs_versicolor":
        y = np.where(y == 0, -1, 1) # 0 becomes -1, 1 becomes 1
    elif task_type == "versicolor_vs_virginica":
        y = np.where(y == 1, -1, 1) # 1 becomes -1, 2 becomes 1
        
    #Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    #Reduce to 2 features (PCA) so we can plot decision boundaries later
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    #Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    #Quantum Angle Rescaling (Squeezing values between -pi and +pi)
    minmax = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_final = minmax.fit_transform(X_train_scaled)
    X_test_final = minmax.transform(X_test_scaled)
    
    return X_train_final, X_test_final, y_train, y_test

def plot_dataset(X, y):
    """
    Plot the 2D PCA representation of the dataset.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Reduce 4D data to 2D for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr', edgecolor='k', s=80)
    plt.title("Iris Dataset - 2D Representation")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, ticks=[-1, 1], label="Classes (-1 and 1)")
    plt.show()

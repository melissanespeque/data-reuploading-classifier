import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#=======================================================
#=================SYNTHETIC DATASETS====================
#=======================================================

#Cirle's generation function (obtained from the article)
def circle(samples, random_seed=42):
    np.random.seed(random_seed)
    centers = np.array([[0, 0]])
    radii = np.array([np.sqrt(2/np.pi)]) #def radius so it represents 0.5 the area of the square - should be a balanced dataset in classification
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1 #so that every point its inside the squared limit (-1,1)
        #random.rand = create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
        y = 0
        for c, r in zip(centers, radii):  
            if np.linalg.norm(x - c) < r: #calculates distance between point and center to classify if inside the circle or not
                y = 1 

        data.append([x, y])
            
    return data, (centers, radii) 

#Circle's visualization function
def plot_circle(data, centers, radii):
    
    x = np.array([i[0] for i in data]) #datapoints
    y = np.array([i[1] for i in data]) #labels definition
    
    center = centers[0]
    radius = radii[0]
    
    plt.figure(figsize=(6,6))
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='goldenrod', alpha=0.5, s=20, label='Outside') #plotting the dataset with its labels
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='fuchsia', alpha=0.5, s=20, label='Inside')
    
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),  #plotting the circle
            'k--', linewidth=2, label='Border')    
    
data, (centers, radii) = circle(samples=3000, random_seed=42) #running for N=3000
fig = plot_circle(data, centers, radii)
plt.show()

#=======================================================

#Diamond's shape pattern generation function
def diamond(samples, limit, scale, random_seed=42):
    np.random.seed(random_seed)
    
    x = np.random.uniform(0, limit, samples) #generates the data points inside the square limit
    y = np.random.uniform(0, limit, samples)
    
    logic = (np.floor((x + y) / scale).astype(int) + np.floor((x - y + scale/2) / scale).astype(int)) % 2  
    #(x + y) and (x - y) to create diagonals
    #Default function: (floor((x+y)/s) + floor((x-y+s/2)/s)) mod 2
    #scale determines the frequence of pattern - if bigger, diamonds get bigger; otherwise, denser
    return x, y, logic

#Diamond's visualization function
def plot_diamond(x_data, y_data, logic):
    colors = np.where(logic == 0, 'fuchsia', 'goldenrod')

    plt.figure(figsize=(6, 6))
    plt.scatter(x_data, y_data, c=colors, s=10)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal')
    plt.title("Diamonds Pattern Classification")
    
x_data, y_data, logic = diamond(samples=5000, limit=10, scale=4.0)
fig = plot_diamond(x_data, y_data, logic)
plt.show()

#=======================================================
#Wavy pattern generation function (obtained from the article)
def wavy_lines(samples, freq = 1):
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
        if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
        if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 2
        if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 3        
        data.append([x, y])

    return data, freq

#Wavy pattern visualization function
def plot_wavy(data,freq):
#separating into features and labels
    x = np.array([i[0] for i in data]) 
    y = np.array([i[1] for i in data])

    plt.figure(figsize=(6,6))

    plt.scatter(x[:,0], x[:,1], c=y, cmap='spring', s=10)

# curves
    s = np.linspace(-1, 1, 500)
    plt.plot( s, np.clip(s + np.sin(freq * np.pi * s), -1, 1),   'k-', linewidth=2)
    plt.plot( s, -s + np.sin(freq * np.pi * s), 'k-', linewidth=2)
#  visual adjustments
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal')
    plt.grid(alpha=0.3)
    plt.title('Wavy Lines Classification')
data, freq = wavy_lines(samples=3000, freq=1)
fig = plot_wavy(data,freq)
plt.show()

#=======================================================
#======================IRIS DATASET=====================
#=======================================================
def get_iris_binary(task_type="setosa_vs_versicolor", test_size=0.2, random_state=42):
    
#Binary classification in quantum circuits with load and preprocess the Iris dataset
#task_type (str): "setosa_vs_versicolor" or "versicolor_vs_virginica"
#test_size (float): Proportion of the dataset to include in the test split
#random_state (int): Random state for reproducibility
        
#  Returns:
#        X_train, X_test, y_train, y_test
    
    #Load Data
    iris = load_iris()
    X = iris.dat
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
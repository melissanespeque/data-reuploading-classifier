import numpy as np
import matplotlib.pyplot as plt

from datasets import *
from problem_gen import *
from test_set import *
from save_data import *
from fid_minimization import *
from weight_fid_minimization import *

problems = ['circle', '3 circles', 'wavy circle', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']


def minimizer(chi, problem, qubits, entanglement, layers, method, name,
              seed = 30, epochs=3000, batch_size=20,  eta=0.1):
    """
    This function creates data and minimizes whichever problem (from the selected ones) 
    INPUT:
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -seed: seed of numpy.random, needed for replicating results
        -epochs: number of epochs for a 'SGD' method. If there is another method, this input has got no importance
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    np.random.seed(seed)
    data, drawing = data_generator(problem)
    train_data = data[:200]
    test_data = data[200:]
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
        theta, alpha, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, f, loss_hist = fidelity_minimization(theta, alpha, train_data, reprs,
                                            entanglement, method, 
                                            batch_size, eta, epochs)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi)
        write_summary(chi, problem, qubits, entanglement, layers, method, name,
              theta, alpha, 0, loss_hist, f, acc_train, acc_test, seed, epochs=epochs)
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        theta, alpha, weight, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, weight, f, loss_hist = weighted_fidelity_minimization(theta, alpha, weight, train_data, reprs,
                                            entanglement, method)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi, weights=weight)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi, weights=weight)
        write_summary(chi, problem, qubits, entanglement, layers, method, name,
              theta, alpha, weight, loss_hist, f, acc_train, acc_test, seed, epochs=epochs)
        
'''
def plot_loss_hist(loss_hist):
    iterations = [point[0] for point in loss_hist]
    cost = [point[1] for point in loss_hist]

    plt.figure(figsize=(7,7))
    plt.plot(iterations, cost, linewidth = 2, marker = 'o', color = 'r')
    plt.show()
'''
def plot_loss_tot(chi, problem, qubits, entanglement, layers, method):
    plt.figure(figsize = (7,7))

    for i in layers:
        name_fold = name_folder(chi, problem, qubits, entanglement, i, method)
        file_text = np.loadtxt(name_fold + '/' + 'run_loss.txt')
        iteration = file_text[:,0]
        cost = file_text[:,1]
        if chi == 'fidelity_chi':
            plt.plot(iteration, 1+cost, marker = 'o', linewidth = 1, markersize = 2, linestyle='-', label = str(i)+' layer')
        else:
            plt.plot(iteration, cost, marker = 'o', linewidth = 1, markersize = 2, linestyle='-', label = str(i)+' layer')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.grid()
    plt.legend()
    plt.show()

                
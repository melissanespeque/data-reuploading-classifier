import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap 
from matplotlib.colors import Normalize

from test_set import *
from datasets import *
from problem_gen import *

### save_data ###
def write_summary(chi, problem, qubits, entanglement, layers, method, name,
          theta, alpha, weights, loss_history, chi_value, acc_train, acc_test, seed, epochs):
    """
    This function takes some informations of a given problem and saves some text files 
    with this information and the parameters which are solution of the problem
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose between
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. Must be an array with shape (classes, qubits)
        -chi_value: Value of the cost function after minimization
        -acc_train: accuracy for the training set
        -acc_test: accuracy for the test set
        -seed: seed of numpy.random, needed for replicating results
        -epochs: number of epochs for a 'SGD' method. If there is another method, this input has got no importance
        
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    create_folder(foldname)
    file_text = open(foldname + '/' + name + '_summary.txt','w')
    file_text.write('\nFigur of merit = '+chi)
    file_text.write('\nProblem = ' + problem)
    file_text.write('\nNumber of qubits = ' + str(qubits))
    if qubits != 1:
        file_text.write('\nEntanglement = ' + entanglement)
    file_text.write('\nNumber of layers = ' + str(layers))
    file_text.write('\nMinimization method = '+ method)
    file_text.write('\nRandom seed = '+ str(seed))
    if method == 'SGD':
        file_text.write('\nNumber of epochs = '+ str(epochs))
    file_text.write('\n\nBEST RESULT\n\n')
    file_text.write('\nTHETA = \n')
    file_text.write(str(theta))
    file_text.write('\nALPHA = \n')
    file_text.write(str(alpha))
    if chi == 'weighted_fidelity_chi':
        file_text.write('\nWEIGHTS = \n')
        file_text.write(str(weights))
    file_text.write('\nchi**2 = ' + str(chi_value))
    file_text.write('\nacc_train = ' + str(acc_train))
    file_text.write('\nacc_test = ' + str(acc_test))
    file_text.close()
    
    np.savetxt(foldname + '/' + name + '_theta.txt', theta.flatten())
    np.savetxt(foldname + '/' + name + '_alpha.txt', alpha.flatten())
    np.savetxt(foldname + '/' + name + '_loss.txt', np.array(loss_history))
    if chi == 'weighted_fidelity_chi':
        np.savetxt(foldname + '/' + name + '_weight.txt', weights.flatten())


def create_folder(directory): 
    """
    Auxiliar function for creating directories with name directory
    
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def name_folder(chi, problem, qubits, entanglement, layers, method):
    """
    This function takes information from the SGD_step_by_step function and saves the accuracies for training and test sets. It is required for studying the overlearning
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -accs_train: list or array with the accuracies of the training set for all epochs
        -accs_test: list or array with the accuracies of the test set for all epochs
    OUTPUT:
        -foldname: A name for a folder
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
    foldname = chi + '/'
    problem = problem.replace(' ', '_')
    foldname += problem + '/'
    foldname += str(qubits) + '_qubits/'
    if qubits != 1: 
        if entanglement.lower()[0] == 'y':
            foldname += 'entangled/'
        if entanglement.lower()[0] == 'n':
            foldname += 'not_entangled/'
            
    foldname += str(layers) + '_layers/'
    foldname += method
    
    return foldname


def painter(chi, problem, qubits, entanglement, layers, method, name, 
            seed = 30, standard_test = True, samples = 4000, bw = False, err = False):
    """
    This function takes written text files and paint the results of the problem 
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
        -standard_test: Whether we want to paint the set test used for checking when minimizing. If True, seed and samples are not taken in account
        -samples: number of samples of the test set
        -bw: painting in black and white
    OUTPUT:
        This function has got no outputs, but a file containing the representation of the test set is created
    """
    np.random.seed(seed)
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        
    if standard_test == True:
        data, drawing = data_generator(problem)
        test_data = data[200:]
            
    elif standard_test == False:
        test_data, drawing = data_generator(problem, samples = samples)
            
    if problem in ['circle', 'diamond']:
        classes = 2

    elif problem == 'wavy lines':
        classes = 4
        
    reprs = representatives(classes, qubits_lab)
    
    params = read_summary(chi, problem, qubits, entanglement, layers, method, name)
    
    if chi == 'fidelity_chi':
        theta, alpha = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi)
        
    if chi == 'weighted_fidelity_chi':
        theta, alpha, weight = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs,
                                           entanglement, chi, weights = weight)

    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    samples_paint(problem, drawing, sol_test, foldname, name, bw)


def read_summary(chi, problem, qubits, entanglement, layers, method, name):
        
    """
    This function reads the files saved by write_summary and returns theta, alpha and weight parameters
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines'
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        
    OUTPUT:
        -theta: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. It is an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. It is an array with shape (classes, qubits)
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
    if chi == 'fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
        if problem in ['circle', 'diamond', 'wavy lines']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))
        return theta, alpha
    
    if chi == 'weighted_fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
        if problem in ['circle', 'diamond', 'wavy lines']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))

        if problem == 'wavy lines':
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((4, qubits))
        if problem in ['circle','diamond']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((2, qubits))
        return theta, alpha, weight
    


def samples_paint(problem, settings, sol, foldname, filename, bw):
    """
    This function takes the problem and the points when they are already classified, and saves a picture of them
    INPUT: 
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -settings: parameters the function needs for drawing. Provided by problem_gen.problem_gen
        -sol: solutions of the points alreafy classified
        -foldname : name of the folder where we store results
        -filename: name of the files we will produce
        -bw: black and white, True/False
    OUTPUT:
        a file with the points and their classes, and whether they are right or wrong
    """
    if bw == False:
        colors_classes = get_cmap('plasma')
        norm_class = Normalize(vmin=-.5,vmax=np.max(sol[:,-3]) + .5)
    
        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)
        
    if bw == True:
        colors_classes = get_cmap('Greys')
        norm_class = Normalize(vmin=-.1,vmax=np.max(sol[:,-3]) + .1)
    
        colors_rightwrong = get_cmap('Greys')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)

    fig, axs = plt.subplots(ncols = 2, figsize=(10,5))
    ax = axs[0]
    if problem in ['circle']:
        centers, radii = settings
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            ax.add_artist(ca)
    
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        ax.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        ax.plot(s, -s + np.sin(freq * np.pi * s), 'k-')

    elif problem == 'diamond':
        limit, scale = settings
        
        s = np.linspace(-1,1,200)
        for k in range(-10,10):        
            ax.plot(s, np.clip(k*scale-s, -1, 1), 'k-', linewidth = 1)
            ax.plot(s, np.clip(s-k*scale+scale/2, -1, 1), 'k-', linewidth = 1)
        
    elif problem == 'non convex':
        freq, x_val, sin_val = settings
        s = np.linspace(-1,1,100)
        ax.plot(s, np.clip(-x_val * s + sin_val * np.sin(freq * np.pi * s), -1, 1), 'k-')

    ax.scatter(sol[:,0], sol[:,1], c=sol[:,-2], cmap = colors_classes, s=2, norm=norm_class)
    
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.tick_params(axis='both',labelsize=16)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.margins(0)
    #ax.axis('equal')
    ax.set_aspect('equal', adjustable='box')

    bx = axs[1]    
    bx.scatter(sol[:,0], sol[:,1], c=sol[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)  
    if problem == 'circle':
        centers, radii = settings
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            bx.add_artist(ca)
    
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        bx.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        bx.plot(s, -s + np.sin(freq * np.pi * s), 'k-')

    elif problem == 'diamond':
        limit, scale = settings
        s = np.linspace(-1,1,100)
        for k in range(-6,6):        
            bx.plot(s, np.clip(k*scale-s, -1, 1), 'k-', linewidth = 1)
            bx.plot(s, np.clip(s-k*scale+scale/2, -1, 1), 'k-', linewidth = 1)
        
        
    elif problem == 'non convex':
        freq, x_val, sin_val = settings
        s = np.linspace(-1,1,100)
        bx.plot(s, np.clip(-x_val * s + sin_val * np.sin(freq * np.pi * s), -1, 1), 'k-')

    
    bx.set_xlabel('x', fontsize=16)
    bx.tick_params(axis='x', labelsize = 16)
    bx.tick_params(axis='y', labelsize=0)
    bx.set_xlim([-1, 1])
    bx.set_ylim([-1, 1])
    bx.margins(0)
    #bx.axis('equal')
    bx.set_aspect('equal', adjustable='box')

    fig.savefig(foldname + '/' + filename)
    plt.close('all')

'''
### paint_world ###
def paint_world(chi, problem, qubits, entanglement, layers, method, name,
            seed = 30, standard_test = True, samples = 4000, bw = False, err = False):
    np.random.seed(seed)

    if chi == 'fidelity_chi':
        qubits_lab = qubits
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1

    if standard_test == True:
        data, drawing = data_generator(problem)
        if problem == 'sphere':
            test_data = data[500:]
        elif problem == 'hypersphere':
            test_data = data[1000:]
        else:
            test_data = data[200:]

    elif standard_test == False:
        test_data, drawing = data_generator(problem, samples=samples)

    if problem in ['circle', 'wavy circle', 'sphere', 'non convex', 'crown', 'hypersphere']:
        classes = 2
    if problem in ['tricrown']:
        classes = 3
    elif problem in ['3 circles', 'wavy lines', 'squares']:
        classes = 4

    reprs = representatives(classes, qubits_lab)

    params = read_summary(chi, problem, qubits, entanglement, layers, method, name)

    if chi == 'fidelity_chi':
        theta, alpha = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi)

    if chi == 'weighted_fidelity_chi':
        theta, alpha, weight = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs,
                                           entanglement, chi, weights=weight)

    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    angles = np.zeros((len(sol_test), 2))
    for i, x in enumerate(sol_test[:, :2]):
        theta_aux = code_coords(theta, alpha, x)
        C = circuit(theta_aux, entanglement)
        psi = Statevector(C)
        angles[i, 0] = np.arccos(np.abs(psi.data[0])**2 - np.abs(psi.data[1])**2) - np.pi/2
        angles[i, 1] = np.angle(psi.data[1] / psi.data[0])
        #print(angles[i])

    if bw == False:
        colors_classes = get_cmap('plasma')
        norm_class = Normalize(vmin=-.5, vmax=np.max(sol_test[:, -3]) + .5)

        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1, vmax=1.1)

    if bw == True:
        colors_classes = get_cmap('Greys')
        norm_class = Normalize(vmin=-.1, vmax=np.max(sol_test[:, -3]) + .1)

        colors_rightwrong = get_cmap('Greys')
        norm_rightwrong = Normalize(vmin=-.1, vmax=1.1)

    fig, ax = plt.subplots(nrows=2, figsize = (4,8.5))
    ax[0].plot(laea_x(np.pi, np.linspace(0, np.pi)), laea_y(np.pi, np.linspace(0, np.pi)), color='k')
    ax[0].plot(laea_x(-np.pi, np.linspace(0, -np.pi)), laea_y(-np.pi, np.linspace(0, -np.pi)), color='k')
    ax[1].plot(laea_x(np.pi, np.linspace(0, np.pi)), laea_y(np.pi, np.linspace(0, np.pi)), color='k')
    ax[1].plot(laea_x(-np.pi, np.linspace(0, -np.pi)), laea_y(-np.pi, np.linspace(0, -np.pi)), color='k')
    ax[0].scatter(laea_x(angles[:, 1], angles[:, 0]), laea_y(angles[:, 1], angles[:, 0]), c=sol_test[:, -2],
                  cmap=colors_classes, s=2, norm=norm_class)
    ax[1].scatter(laea_x(angles[:, 1], angles[:, 0]), laea_y(angles[:, 1], angles[:, 0]), c=sol_test[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)
    plt.show()


def laea_x(lamb, phi):
    return 2*np.sqrt(2) * np.cos(phi)*np.sin(lamb / 2) / np.sqrt(1 + np.cos(phi)*np.cos(lamb/2))


def laea_y(lamb, phi):
    return np.sqrt(2) * np.sin(phi) / np.sqrt(1 + np.cos(phi)*np.cos(lamb/2))

'''
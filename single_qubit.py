from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap 
from matplotlib.colors import Normalize

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
    if problem == 'sphere':
        train_data = data[:500] 
        test_data = data[500:]
    elif problem == 'hypersphere':
        train_data = data[:1000] 
        test_data = data[1000:]
    else:
        train_data = data[:200]
        test_data = data[200:]
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
        theta, alpha, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, f = fidelity_minimization(theta, alpha, train_data, reprs,
                                            entanglement, method, 
                                            batch_size, eta, epochs)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi)
        write_summary(chi, problem, qubits, entanglement, layers, method, name,
              theta, alpha, 0, f, acc_train, acc_test, seed, epochs=epochs)
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        theta, alpha, weight, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, weight, f = weighted_fidelity_minimization(theta, alpha, weight, train_data, reprs,
                                            entanglement, method)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi, weights=weight)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi, weights=weight)
        write_summary(chi, problem, qubits, entanglement, layers, method, name,
              theta, alpha, weight, f, acc_train, acc_test, seed, epochs=epochs)
        

def data_generator(problem, samples=None):
    """
    This function generates the data for a problem
    INPUT: 
        -problem: Name of the problem, one of: 'circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines'
        -samples Number of samples for the data
    OUTPUT:
        -data: set of training and test data
        -settings: things needed for drawing
    """
    problem = problem.lower()
    if problem not in problems:
        raise ValueError('problem must be one of {}'.format(problems))
    if samples == None:
        if problem == 'sphere': 
            samples = 4500
        elif problem == 'hypersphere':
            samples = 5000
        else: 
            samples = 4200
            
    if problem == 'circle':
        data, settings = _circle(samples)
    
    return data, settings 


def _circle(samples):
    centers = np.array([[0, 0]])
    radii = np.array([np.sqrt(2/np.pi)])
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for c, r in zip(centers, radii):  
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
            
    return data, (centers, radii)


def problem_generator(problem, qubits, layers, chi, qubits_lab=1):
    """
    This function generates everything needed for solving the problem
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account

        
    OUTPUT:
        -theta: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. It is an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. It is an array with shape (classes, qubits)
        -reprs: variable encoding the label states of the different classes
    """
    chi = chi.lower()
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
        
    if chi == 'weighted_fidelity_chi' and qubits_lab != 1: 
        qubits_lab = 1
        print('WARNING: number of qubits for the label states has been changed to 1')
    
    problem = problem.lower()
    if problem == 'circle':
        theta, alpha, reprs = q_circle(qubits, layers, qubits_lab, chi)        
    else:
        raise ValueError('Problem is not valid')
        
    if chi == 'fidelity_chi':
        return theta, alpha, reprs
    elif chi == 'weighted_fidelity_chi':
        weights = np.ones((len(reprs), qubits))
        return theta, alpha, weights, reprs
    

def q_circle(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs


def fidelity_minimization(theta, alpha, train_data, reprs, 
                       entanglement, method,
                       batch_size, eta, epochs):
    """
    This function takes the parameters of a problem and computes the optimal parameters for it, using different functions. It uses the fidelity minimization
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
        -epochs: number of epochs , only for 'SGD' method
    OUTPUT:
        -theta: optimized point for the theta parameters. The shape is correct (qubits, layers, 3)
        -alpha: optimized point for the alpha parameters. The shape is correct (qubits, layers, dim)
        -chi: value of the minimization function
    """
    
    
    if method == 'SGD':
        thetas, alphas, chis = _sgd(theta, alpha, train_data, reprs,
                                        entanglement, eta, batch_size, epochs)
        i = chis.index(max(chis))
        return thetas[i], alphas[i], chis[i]
    
    else:
        params, hypars = _translate_to_scipy(theta, alpha)
        results = minimize(_scipy_minimizing, params, 
                           args = (hypars, train_data, reprs, entanglement),
                                   method=method)
        theta, alpha = _translate_from_scipy(results['x'], hypars)
        
        return theta, alpha, results['fun']


def _translate_to_scipy(theta, alpha):
    """
    This function is a intermediate step for translating theta and alpha to a single variable for scipy.optimize.minimize
    """
    qubits = theta.shape[0]
    layers = theta.shape[1]
    dim = alpha.shape[-1]
    
    return np.concatenate((theta.flatten(), alpha.flatten())), (qubits, layers, dim)


def _translate_from_scipy(params, hypars):
    """
    This function is a intermediate step for getting theta and alpha from a single variable for scipy.optimize.minimize
    """
    (qubits, layers, dim) = hypars
    if dim <= 3:
        theta = params[:qubits * layers * 3]. reshape(qubits, layers, 3)
        alpha = params[qubits * layers * 3: qubits * layers * 3 + qubits * layers * dim].reshape(qubits, layers, dim)
        
    if dim == 4:
        theta = params[:qubits * layers * 6]. reshape(qubits, layers, 6)
        alpha = params[qubits * layers * 6: qubits * layers * 6 + qubits * layers * dim].reshape(qubits, layers, dim)
    return theta, alpha


def _scipy_minimizing(params, hypars, train_data, reprs, entanglement):
    """
    This function returns the chi^2 function for using scipy
    INPUT:
        -params: theta and alpha inside the same variable
        -hypars: hyperparameters needed to rebuild theta and alpha
        -train_data: training dataset for the classifier
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        - -Av_chi_square, which is the function we want to minimize
    """
    theta, alpha = _translate_from_scipy(params, hypars)
    return -Av_chi_square(theta, alpha, train_data, reprs, entanglement)


def _chi_square(theta, alpha, data, reprs, entanglement): #Chi for one point
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -chi^2 for data
    """
    #
    x, y = data
    theta_aux = code_coords(theta, alpha, x)
    C = circuit(theta_aux, entanglement)
    psi = Statevector(C)
    ans = np.sqrt(state_fidelity(reprs[y], psi))
    return ans


#def fidelity(qState1, qState2):
 #   """
  #  This function returns the relativy fidelity of two pure states
   # INPUT:
    #    -2 pure states of the same dimension
  #  OUTPUT:
   #     -relative fidelity
    #"""
    #return np.abs(np.dot(np.conj(qState1), qState2))


def code_coords(theta, alpha, x):  #Encoding of coordinates
    """
    This functions converts theta, alpha and x in a new set of variables encoding the three of them properly
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -x: one data for training, only the coordinates
    OUTPUT:
        -theta_aux: shifted thetas encoding alpha and x inside. Same shape as theta
    """
    theta_aux = theta.copy()
    qubits = theta.shape[0]
    layers = theta.shape[1]
    for q in range(qubits):
        for l in range(layers):
            if len(x) <= 3:
                for i in range(len(x)):
                    theta_aux[q, l, i] += alpha[q, l, i] * x[i]
            elif len(x) == 4:
                theta_aux[q, l, 0] += alpha[q, l, 0] * x[0]
                theta_aux[q, l, 1] += alpha[q, l, 1] * x[1]
                theta_aux[q, l, 3] += alpha[q, l, 2] * x[2]
                theta_aux[q, l, 4] += alpha[q, l, 3] * x[3]
            else:
                raise ValueError('Data has too many dimensions')
    
    return theta_aux


def circuit(theta_aux, entanglement):
    """
    This creates the Quantum circuit for the problem using QuantumState.QCircuit
    INPUT: 
        -theta_aux: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3 or 6). Alpha and x are coded inside theta_aux
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        -quantum circuits coding the problem and our Ansätze
    """
    hypar = theta_aux.shape #[qubits, layers, params_per_layer]
    entanglement = entanglement.lower()[0]
    if hypar[-1] not in [3, 6]: 
        raise ValueError('The number of parameters per gate is not correct')
    
    if hypar[-1] == 3:
        num_qubits = hypar[0]
        if num_qubits == 1:
            return single_qubit_circuit(theta_aux)


def single_qubit_circuit(theta_aux):

    qc = QuantumCircuit(1)

    for i in range(theta_aux.shape[1]):
        mat = U3(theta_aux[0, i, :])
        qc.append(UnitaryGate(mat),[0])
        
    return qc


def U3(theta3):
        t, phi, lam = theta3
        c = np.cos(t / 2)
        s = np.sin(t / 2)
        e_phi = np.exp(1j * phi / 2)
        e_lambda = np.exp(1j * lam / 2)
        
        return np.array([
        [ c * e_phi * e_lambda,        -s * e_phi * np.conj(e_lambda)],
        [ s * np.conj(e_phi) * e_lambda,  c * np.conj(e_phi) * np.conj(e_lambda)]
        ])


def Av_chi_square(theta, alpha, train_data, reprs, entanglement): #Chi in average
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -Averaged chi^2 for data
    """
    Av_Chi = 0
    for d in train_data:
        Av_Chi += _chi_square(theta, alpha, d, reprs, entanglement)

    return Av_Chi / len(train_data)


def tester(theta, alpha, test_data, reprs, entanglement, chi, weights=None):
    """
    This function takes the parameters of a solved problem and one data computes how many points are correct
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -test_data: set of data for testing
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -success normalized
    """
    acc = 0
    for i, d in enumerate(test_data):
        x, y = d
        y_ = _claim(theta, alpha, weights, x, reprs, entanglement, chi)
        if y == y_:
            acc += 1
    
    return acc / len(test_data)


def _claim(theta, alpha, weight, x, reprs, entanglement, chi):
    """
    This function takes the parameters of a solved problem and one data computes classification of this point
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -y_: the class of x, according to the classifier
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
        
    if chi == 'fidelity_chi':
        y_ = _claim_fidelity(theta, alpha, x, reprs, entanglement)
        
    if chi == 'weighted_fidelity_chi':
        y_ = _claim_weighted_fidelity(theta, alpha, weight, x, reprs, entanglement)
        
    return y_   


def _claim_fidelity(theta, alpha, x, reprs, entanglement):
    """
    This function is inside _claim for fidelity_chi
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        the class of x, according to the classifier
    """
    theta_aux = code_coords(theta, alpha, x)
    C = circuit(theta_aux, entanglement)
    psi = Statevector(C)
    Fidelities = [np.sqrt(state_fidelity(r, psi)) for r in reprs]
    
    return np.argmax(Fidelities)


def representatives(n_classes, qubits_lab):

    reprs = np.zeros((n_classes, 2**qubits_lab), dtype = 'complex')
    if qubits_lab == 1:
        if n_classes == 0:
            raise ValueError('Nonsense classifier')
        if n_classes == 1:
            raise ValueError('Nonsense classifier')
        if n_classes == 2:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([0, 1])
        if n_classes == 3:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([1 / 2, np.sqrt(3) / 2])
            reprs[2] = np.array([1 / 2, -np.sqrt(3) / 2])
        if n_classes == 4:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)])
            reprs[2] = np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
            reprs[3] = np.array([1 / np.sqrt(3), np.exp(-1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
        if n_classes == 6:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([0, 1])
            reprs[2] = 1 / np.sqrt(2) * np.array([1, 1])
            reprs[3] = 1 / np.sqrt(2) * np.array([1, -1])
            reprs[4] = 1 / np.sqrt(2) * np.array([1, 1j])
            reprs[5] = 1 / np.sqrt(2) * np.array([1, -1j])
    #if qubits_lab == 2:
     #   if n_classes == 0:
     #       raise ValueError('Nonsense classifier')
      #  if n_classes == 1:
       #     raise ValueError('Nonsense classifier')
        #if n_classes == 2:
         #   reprs[0] = np.array([1, 0, 0, 0])
          #  reprs[1] = np.array([0, 0, 0, 1])
     #   if n_classes == 3:
      #      reprs[0] = np.array([1, 0, 0, 0])
       #     reprs[1] = np.array([0, 1, 0, 0])
        #    reprs[2] = np.array([0, 0, 1, 0])
       # if n_classes == 4:
        #    reprs[0] = np.array([1, 0, 0, 0])
         #   reprs[1] = np.array([0, 1, 0, 0])
          #  reprs[2] = np.array([0, 0, 1, 0])
          #  reprs[3] = np.array([0, 0, 0, 1])
            
    return reprs


### save_data ###
def write_summary(chi, problem, qubits, entanglement, layers, method, name,
          theta, alpha, weights, chi_value, acc_train, acc_test, seed, epochs):
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


### painter ###
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
        if problem == 'sphere':
            test_data = data[500:]
        elif problem == 'hypersphere':
            test_data = data[1000:]
        else:
            test_data = data[200:]
            
    elif standard_test == False:
        test_data, drawing = data_generator(problem, samples = samples)
            
    if problem in ['circle','wavy circle','sphere', 'non convex', 'crown', 'hypersphere']:
        classes = 2
    if problem in ['tricrown']:
        classes = 3
    elif problem in ['3 circles','wavy lines','squares']:
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
        if problem in ['circle', '3 circles', 'wavy circles', 'wavy lines', 'non convex','crown','tricrown','squares']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
        elif problem == 'sphere': 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 3
        elif problem in ['hypersphere']: 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 6))
            dim = 4
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))
        return theta, alpha
    
    if chi == 'weighted_fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
        if problem in ['circle', '3 circles', 'wavy circles', 'wavy lines', 'non convex','crown','tricrown','squares']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
        elif problem == 'sphere': 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 3
        elif problem in ['hypersphere']: 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 6))
            dim = 4
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))

        if problem in ['3 circles','wavy lines','squares']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((4, qubits))
        if problem in ['circle','wavy circle','sphere', 'non convex', 'crown', 'hypersphere']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((2, qubits))
        if problem in ['tricrown']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((3, qubits))
        return theta, alpha, weight
    

def Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi, weights=None):
    """
    This function takes the parameters of a solved problem and one data computes how many points are correct
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -test_data: set of data for testing
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -solutions of the classification
        -success normalized
    """
    dim = len(test_data[0][0])
    solutions = np.zeros((len(test_data), dim + 3)) #data  #Esto se podrá mejorar en el futuro
    for i, d in enumerate(test_data):
        x, y = d
        y_ = _claim(theta, alpha, weights, x, reprs, entanglement, chi)
        solutions[i,:dim] = x
        solutions[i, -3] = y
        solutions[i, -2] = y_
        solutions[i, -1] = int(y == y_)
        
    acc = np.sum(solutions[:, -1]) / (i + 1)
    
    return solutions, acc


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
    if problem in ['circle', '3 circles', 'crown', 'tricrown']:
        centers, radii = settings
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            ax.add_artist(ca)
    elif problem == 'wavy circle':
        centers, radii, wave, freq = settings
        phi = np.linspace(0, 2*np.pi, 1000)
        for (c,r, w, f) in zip(centers, radii, wave, freq):
            ax.plot(c[0] + r*(1 + w * np.cos(f * phi)) * np.cos(phi),
                    c[1] + r*(1 + w * np.cos(f * phi)) * np.sin(phi),
                    'k-')
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        ax.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        ax.plot(s, -s + np.sin(freq * np.pi * s), 'k-')
    elif problem == 'squares':
        freq = settings
        s = np.linspace(-1,1,10)
        ax.plot(s, np.zeros(10), 'k-')
        ax.plot(np.zeros(10), s, 'k-')
        
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
    ax.axis('equal')
    
    bx = axs[1]    
    bx.scatter(sol[:,0], sol[:,1], c=sol[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)  
    if problem in ['circle', '3 circles', 'crown', 'tricrown']:
        centers, radii = settings
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            bx.add_artist(ca)
    elif problem == 'wavy circle':
        centers, radii, wave, freq = settings
        phi = np.linspace(0, 2*np.pi, 1000)
        bx.plot(c[0] + r*(1 + wave * np.cos(freq * phi)) * np.cos(phi),
                c[1] + r*(1 + wave * np.cos(freq * phi)) * np.sin(phi),
                'k-')
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        bx.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        bx.plot(s, -s + np.sin(freq * np.pi * s), 'k-')

    elif problem == 'squares':
        freq = settings
        s = np.linspace(-1,1,10)
        bx.plot(s, np.zeros(10), 'k-')
        bx.plot(np.zeros(10), s, 'k-')
        
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
    bx.axis('equal')
    
    fig.savefig(foldname + '/' + filename)
    plt.close('all')


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
        print(angles[i])

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
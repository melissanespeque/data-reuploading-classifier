from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity

## pack params for scipy.optimize.minimize
def pack_params(theta, w, alpha):

    params = np.concatenate([theta.ravel(), w.ravel() ,alpha.ravel()])

    return params

def single_unpack_params(params, n_layers, n_classes):

    theta_size = 1*n_layers*3 #n_qubits*n_layers*3
    w_size = 1*n_layers*3

    theta = params[:theta_size].reshape(n_layers, 3)
    w = params[theta_size:theta_size+w_size].reshape(n_layers, 3)
    alpha = params[theta_size+w_size: ].reshape(n_classes, 1)

    return theta, w, alpha

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
    if qubits_lab == 2:
        if n_classes == 0:
            raise ValueError('Nonsense classifier')
        if n_classes == 1:
            raise ValueError('Nonsense classifier')
        if n_classes == 2:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 0, 0, 1])
        if n_classes == 3:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 1, 0, 0])
            reprs[2] = np.array([0, 0, 1, 0])
        if n_classes == 4:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 1, 0, 0])
            reprs[2] = np.array([0, 0, 1, 0])
            reprs[3] = np.array([0, 0, 0, 1])
            
    return reprs

def single_qubit_params(n_layers, n_classes):
    
    theta = np.random.uniform(0, 2*np.pi, (n_layers, 3))
    w = np.random.uniform(0, 1, (n_layers, 3))
    alpha = np.random.uniform(0, 1, (n_classes))
    
    return theta, w, alpha

## This function is building a quantum circuit formed by 1 qubit. It has to be called for every datapoint.
def single_qubit_circuit(n_layers, x, theta, w):

    qc = QuantumCircuit(1)

    for i in range(n_layers):
        phi = theta[i] + w[i]*x

        qc.rz(phi[2], 0)
        qc.ry(phi[0], 0)
        qc.rz(phi[1], 0)
    
    return qc

## Given the circuit this function calculates the state at the end of the circuit.
def create_state_vector(n_layers, x, theta, w):
    sv = []
    for i in range(len(x)):
        qc = single_qubit_circuit(n_layers, x[i], theta, w)
        sv.append(Statevector.from_circuit(qc))

    return sv

def single_cost_function(params, x, y, n_layers, n_classes, reprs):

    theta, w, _ = single_unpack_params(params, n_layers, n_classes)
    cf = 0
    sv = create_state_vector(n_layers, x, theta, w)
    for i in range(len(x)):
        cf += 1 - state_fidelity(sv[i], reprs[y[i]])

    return cf

def single_qubit_predict(x, n_layers, n_classes, theta, w, reprs):
    
    predictions = []
    
    for xi in x:
        qc = single_qubit_circuit(n_layers, xi, theta, w)
        sv = Statevector(qc)
        
        #compute score for each class
        scores = np.zeros(n_classes)
        for c in range(n_classes):
            scores[c] += state_fidelity(sv, Statevector(reprs[c])).real
        
        predictions.append(np.argmax(scores))
   
    return np.array(predictions)

#def weighted_cost_function(params, n_qubits,n_layers, n_classes, x, y, reprs):

#    wcf_qc = 0

 #   theta, w, alpha = unpack_params(params, n_qubits, n_layers, n_classes)
  #  qc = [circuit(n_qubits, n_layers, theta, w, xi) for xi in x]

   # for i in range(len(qc)):
    #    sv = Statevector(qc[i])

     #   dm = DensityMatrix(sv)
      #  true_class = y[i]
       # Y = get_Y(n_classes, true_class, reprs)   

        #F_all = np.zeros((n_classes, n_qubits))
 #       for c in range(n_classes):
            
  #          for q in range(n_qubits):
   #             other_qubits = [idx for idx in range(n_qubits) if idx != q]
    #            rho=partial_trace(dm, other_qubits) 

     #           F_all[c,q] = state_fidelity(rho, DensityMatrix(reprs[c])).real
        
      #  for c in range(n_classes):
       #     summ = np.sum(alpha[c] * F_all[c, :]) 
        #    wcf_qc += (summ - Y[c]) ** 2 
            

    #return 0.5*wcf_qc
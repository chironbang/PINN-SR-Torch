import numpy as np
import torch
import torch.nn as nn

# Variable description


loss_history_Adam_Pretrain = np.empty([0])
loss_val_history_Adam_Pretrain = np.empty([0])
loss_u_history_Adam_Pretrain = np.empty([0])
loss_v_history_Adam_Pretrain = np.empty([0])
loss_f_u_history_Adam_Pretrain = np.empty([0])
loss_f_v_history_Adam_Pretrain = np.empty([0])
loss_lambda_u_history_Adam_Pretrain = np.empty([0])
loss_lambda_v_history_Adam_Pretrain = np.empty([0])
lambda_u_history_Adam_Pretrain = np.zeros((110,1))
lambda_v_history_Adam_Pretrain = np.zeros((110,1))

# L-BFGS-B loss history(Pretraining)
loss_history_Pretrain = np.empty([0])
loss_val_history_Pretrain = np.empty([0])
loss_u_history_Pretrain = np.empty([0])
loss_v_history_Pretrain = np.empty([0])
loss_f_u_history_Pretrain = np.empty([0])
loss_f_v_history_Pretrain = np.empty([0])
loss_lambda_u_history_Pretrain = np.empty([0])
loss_lambda_v_history_Pretrain = np.empty([0])
lambda_u_history_Pretrain = np.zeros((110,1))
lambda_v_history_Pretrain = np.zeros((110,1))    
step_Pretrain = 0

# L-BFGS-S loss history
loss_history = np.empty([0])
loss_val_history = np.empty([0])
loss_u_history = np.empty([0])
loss_v_history = np.empty([0])
loss_f_u_history = np.empty([0])
loss_f_v_history = np.empty([0])
loss_lambda_u_history = np.empty([0])
loss_lambda_v_history = np.empty([0])
lambda_u_history = np.zeros((110,1))
lambda_v_history = np.zeros((110,1))    
step = 0

# Adam loss history
loss_history_Adam = np.empty([0])
loss_val_history_Adam = np.empty([0])
loss_u_history_Adam = np.empty([0])
loss_v_history_Adam = np.empty([0])
loss_f_u_history_Adam = np.empty([0])
loss_f_v_history_Adam = np.empty([0])
loss_lambda_u_history_Adam = np.empty([0])
loss_lambda_v_history_Adam = np.empty([0])
lambda_u_history_Adam = np.zeros((110,1))
lambda_v_history_Adam = np.zeros((110,1))

# Alter loss history
loss_history_Alter = np.empty([0])
loss_val_history_Alter = np.empty([0])
loss_u_history_Alter = np.empty([0])
loss_v_history_Alter = np.empty([0])
loss_f_u_history_Alter = np.empty([0])
loss_f_v_history_Alter = np.empty([0])
loss_lambda_u_history_Alter = np.empty([0])
loss_lambda_v_history_Alter = np.empty([0])
lambda_u_history_Alter = np.zeros((110,1))
lambda_v_history_Alter = np.zeros((110,1))

# STRidge loss histroy
loss_u_history_STRidge = np.empty([0])
loss_f_u_history_STRidge = np.empty([0])
loss_lambda_u_history_STRidge = np.empty([0])
tol_u_history_STRidge = np.empty([0])
lambda_u_history_STRidge = np.zeros((110, 1))
ridge_u_append_counter_STRidge = np.array([0])

loss_v_history_STRidge = np.empty([0])
loss_f_v_history_STRidge = np.empty([0])
loss_lambda_v_history_STRidge = np.empty([0])
tol_v_history_STRidge = np.empty([0])
lambda_v_history_STRidge = np.zeros((110, 1))
ridge_v_append_counter_STRidge = np.array([0])

lib_fun = []
lib_descr = []


device = 'cuda'



class MLP(nn.Module):
    def __init__(self, layers, lb, ub):
        super(MLP, self).__init__()
        self.lb = lb
        self.ub = ub
        self.weights, self.biases = self.initialize_NN(layers)

    def initialize_NN(self, layers):
        weights = nn.ModuleList()
        biases = nn.ParameterList()
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = nn.Linear(layers[l], layers[l + 1])
            nn.init.xavier_uniform_(W.weight)
            weights.append(W)
            b = nn.Parameter(torch.zeros(1, layers[l + 1], dtype=torch.float32))
            biases.append(b)
        return weights, biases

    def forward(self, X):
        num_layers = len(self.weights)
        
        lb_tensor = torch.tensor(self.lb, dtype=torch.float32).to(device)
        ub_tensor = torch.tensor(self.ub, dtype=torch.float32).to(device)
        
        H = 2.0 * (X - lb_tensor) / (ub_tensor - lb_tensor) - 1.0
        
        for l in range(0, num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W.weight.T), b))

        W = self.weights[-1]
        b = self.biases[-1]
        Y = torch.add(torch.matmul(H, W.weight.T), b)
        
        return Y
    
    
    
def build_library(data, derivatives, derivatives_description, PolyOrder, data_description = None):         
    ## polynomial terms
    P = PolyOrder
    lib_poly = [1]
    lib_poly_descr = [''] # it denotes '1'
    for i in range(len(data)): # polynomial terms of univariable
        for j in range(1, P+1):
            lib_poly.append(data[i]**j)
            lib_poly_descr.append(data_description[i]+"**"+str(j))

    for i in range(1,P): # polynomial terms of bivariable. Assume we only have 2 variables.
        for j in range(1,P-i+1):
            lib_poly.append(data[0]**i*data[1]**j)
            lib_poly_descr.append(data_description[0]+"**"+str(i)+data_description[1]+"**"+str(j))

    ## derivative terms
    lib_deri = derivatives
    lib_deri_descr = derivatives_description

    ## Multiplication of derivatives and polynomials (including the multiplication with '1')
    lib_poly_deri = []
    lib_poly_deri_descr = []
    for i in range(len(lib_poly)):
        for j in range(len(lib_deri)):

            lib_poly_deri.append(lib_poly[i]*lib_deri[j])
            lib_poly_deri_descr.append(lib_poly_descr[i]+lib_deri_descr[j])

    return lib_poly_deri,lib_poly_deri_descr


def callTrainSTRidge(x_f, y_f, t_f, Phi, u_t, v_t, lambda_u, lambda_v, it=0):
    lam = 1e-5
    d_tol = 1
    maxit = 100
    STR_iters = 10

    l0_penalty = None

    normalize = 2
    split = 0.8
    print_best_tol = False     

    # Process of lambda_u            
    lambda_u2, loss_u_history_STRidge2, loss_f_u_history_STRidge2, loss_lambda_u_history_STRidge2, tol_u_history_STRidge2, \
        optimaltol_u_history2, _, _ = TrainSTRidge(Phi, u_t, lam, d_tol, maxit, lambda_u, lambda_v, STR_iters, l0_penalty, normalize,
                                                  split, print_best_tol, uv_flag = True, it=it)
    lambda_u = torch.nn.Parameter(torch.tensor(lambda_u2, dtype=torch.float32))

    global loss_u_history_STRidge
    global loss_f_u_history_STRidge
    global loss_lambda_u_history_STRidge
    global tol_u_history_STRidge

    loss_u_history_STRidge = np.append(loss_u_history_STRidge, loss_u_history_STRidge2)
    loss_f_u_history_STRidge = np.append(loss_f_u_history_STRidge, loss_f_u_history_STRidge2)
    loss_lambda_u_history_STRidge = np.append(loss_lambda_u_history_STRidge, loss_lambda_u_history_STRidge2)
    tol_u_history_STRidge = np.append(tol_u_history_STRidge, tol_u_history_STRidge2)

    # Process of lambda_v    
    lambda_v2, loss_v_history_STRidge2, loss_f_v_history_STRidge2, loss_lambda_v_history_STRidge2, tol_v_history_STRidge2, \
        optimaltol_v_history2, _, _ = TrainSTRidge(Phi, v_t, lam, d_tol, maxit, lambda_u, lambda_v, STR_iters, l0_penalty, normalize,
                                                  split, print_best_tol, uv_flag = False, it=it)

    lambda_v = torch.nn.Parameter(torch.tensor(lambda_v2, dtype=torch.float32))

    global loss_v_history_STRidge
    global loss_f_v_history_STRidge
    global loss_lambda_v_history_STRidge
    global tol_v_history_STRidge

    loss_v_history_STRidge = np.append(loss_v_history_STRidge, loss_v_history_STRidge2)
    loss_f_v_history_STRidge = np.append(loss_f_v_history_STRidge, loss_f_v_history_STRidge2)
    loss_lambda_v_history_STRidge = np.append(loss_lambda_v_history_STRidge, loss_lambda_v_history_STRidge2)
    tol_v_history_STRidge = np.append(tol_v_history_STRidge, tol_v_history_STRidge2)

    return lambda_u, lambda_v




def TrainSTRidge(R, Ut, lam, d_tol, maxit, lambda_u, lambda_v, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, 
                 print_best_tol = False, uv_flag = True, it=0):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# ============================================================================= 

    # Split data into 80% training and 20% test, then search for the best tolderance.
    R = R.cpu().detach().numpy()
    Ut = Ut.cpu().detach().numpy()
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TestR = R[test,:]
    TestY = Ut[test,:]

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol

    # Get the standard least squares estimator            
    if uv_flag:
        w_best = torch.clone(lambda_u)
        w_best = w_best.cpu().detach().numpy()
    else:
        w_best = torch.clone(lambda_v)
        w_best = w_best.cpu().detach().numpy()            
    # err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
    err_f = np.mean((TestY - TestR.dot(w_best))**2)
    
    l0_penalty_0_u, l0_penalty_0_v = 0, 0

    if l0_penalty == None and it == 0: 
        # l0_penalty = 0.05*np.linalg.cond(R)
        if uv_flag:
            l0_penalty_0_u = 10*err_f
            l0_penalty = l0_penalty_0_u
        else:
            l0_penalty_0_v = 10*err_f
            l0_penalty = l0_penalty_0_v

    elif l0_penalty == None:
        if uv_flag:
            l0_penalty = l0_penalty_0_u
        else:
            l0_penalty = l0_penalty_0_v

    err_lambda = l0_penalty*np.count_nonzero(w_best)
    err_best = err_f + err_lambda
    tol_best = 0

    loss_history_STRidge = np.empty([0])
    loss_f_history_STRidge = np.empty([0])
    loss_lambda_history_STRidge = np.empty([0])
    tol_history_STRidge = np.empty([0])
    loss_history_STRidge = np.append(loss_history_STRidge, err_best)
    loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
    loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
    tol_history_STRidge = np.append(tol_history_STRidge, tol_best)

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,lambda_u, lambda_v, normalize = normalize, uv_flag = uv_flag)
        # err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
        err_f = np.mean((TestY - TestR.dot(w))**2)

        err_lambda = l0_penalty*np.count_nonzero(w)
        err = err_f + err_lambda

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = 1.2*tol

            loss_history_STRidge = np.append(loss_history_STRidge, err_best)
            loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
            loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
            tol_history_STRidge = np.append(tol_history_STRidge, tol)

        else:
            tol = 0.8*tol

    if print_best_tol: print ("Optimal tolerance:", tol_best)

    optimaltol_history = np.empty([0])
    optimaltol_history = np.append(optimaltol_history, tol_best)

    return np.real(w_best), loss_history_STRidge, loss_f_history_STRidge, loss_lambda_history_STRidge, tol_history_STRidge, optimaltol_history, l0_penalty_0_u, l0_penalty_0_v


def STRidge(X0, y, lam, maxit, tol, lambda_u, lambda_v, normalize = 2, print_results = False, uv_flag = True):                 
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Get the standard ridge esitmate            
    # Inherit w from previous trainning
    if uv_flag:
        w = torch.clone(lambda_u)
        w = w.cpu().detach().numpy()/Mreg
    else:
        w = torch.clone(lambda_v)
        w = w.cpu().detach().numpy()/Mreg

    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]

    if uv_flag:
        global ridge_u_append_counter_STRidge
        ridge_u_append_counter = 0

        global lambda_u_history_STRidge
        lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, np.multiply(Mreg,w), axis = 1)
        ridge_u_append_counter += 1
    else:
        global ridge_v_append_counter_STRidge
        ridge_v_append_counter = 0

        global lambda_v_history_STRidge
        lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, np.multiply(Mreg,w), axis = 1)
        ridge_v_append_counter += 1

    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                if normalize != 0: 
                    w = np.multiply(Mreg,w)
                    if uv_flag:
                        lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, w, axis = 1)
                        ridge_u_append_counter += 1
                        ridge_u_append_counter_STRidge = np.append(ridge_u_append_counter_STRidge, ridge_u_append_counter)
                    else:
                        lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, w, axis = 1)
                        ridge_v_append_counter += 1
                        ridge_v_append_counter_STRidge = np.append(ridge_v_append_counter_STRidge, ridge_v_append_counter)
                    return w
                else: 
                    if uv_flag:
                        lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, w, axis = 1)
                        ridge_u_append_counter += 1
                        ridge_u_append_counter_STRidge = np.append(ridge_u_append_counter_STRidge, ridge_u_append_counter)
                    else:
                        lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, w, axis = 1)
                        ridge_v_append_counter += 1
                        ridge_v_append_counter_STRidge = np.append(ridge_v_append_counter_STRidge, ridge_v_append_counter)
                    return w
            else: break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0

        if lam != 0: 
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            if uv_flag:
                lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, np.multiply(Mreg,w), axis = 1)
                ridge_u_append_counter += 1
            else:
                lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, np.multiply(Mreg,w), axis = 1)
                ridge_v_append_counter += 1
        else: 
            w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
            if uv_flag:
                lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, np.multiply(Mreg,w), axis = 1)
                ridge_u_append_counter += 1
            else:
                lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, np.multiply(Mreg,w), axis = 1)
                ridge_v_append_counter += 1

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: 
        w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]

    if normalize != 0: 
        w = np.multiply(Mreg,w)
        if uv_flag:
            lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, w, axis = 1)
            ridge_u_append_counter += 1
            ridge_u_append_counter_STRidge = np.append(ridge_u_append_counter_STRidge, ridge_u_append_counter)
        else:
            lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, w, axis = 1)
            ridge_v_append_counter += 1
            ridge_v_append_counter_STRidge = np.append(ridge_v_append_counter_STRidge, ridge_v_append_counter)
        return w
    else:
        if uv_flag:
            lambda_u_history_STRidge = np.append(lambda_u_history_STRidge, w, axis = 1)
            ridge_u_append_counter += 1
            ridge_u_append_counter_STRidge = np.append(ridge_u_append_counter_STRidge, ridge_u_append_counter)
        else:
            lambda_v_history_STRidge = np.append(lambda_v_history_STRidge, w, axis = 1)
            ridge_v_append_counter += 1
            ridge_v_append_counter_STRidge = np.append(ridge_v_append_counter_STRidge, ridge_v_append_counter)
        return w



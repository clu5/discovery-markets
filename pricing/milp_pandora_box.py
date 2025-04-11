import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
from tqdm import tqdm
from queue import PriorityQueue
import pandas as pd
import time
from generate_problems import generate_problem

def cartesian_product_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def check_valid_P(P):
    ''' checks if P is a valid transition matrix
        in:
            P: Q x Q x T -> [0,1] - transition probabilities 
        out: 
            valid - boolean
    '''
    for t in range(P.shape[2]):
        if not np.allclose(np.sum(P[:, :, t], axis=0), 1):
            return False
    return True

def sample_history(n_samples, Q, P):
    '''
    output: list of histories of length n_samples sampled according to the transition probabilities P

    we do this by sampling from strategies with geometric stopping time
    '''
    histories = [] # size (n_samples, T)
    for i in range(n_samples):
        trajectory = []
        curq = 0
        trajectory.append(curq)
        for t in range(P.shape[2]-1):
            if np.random.rand() > 0.5:
                break
            else:
                curq = np.random.choice(Q, p=P[:, curq, t])
                trajectory.append(curq)
        
        # if len(trajectory) < T:
        #     trajectory += [0 for _ in range(T - len(trajectory))]
        histories.append(set(trajectory))
    return histories

def get_q_by_history(tau):
    '''
    input: tau - history of length T
    output: q - output of the history
    '''
    s = set()
    for q in tau:
        s.add(q)
    return s



def compute_profit(histories, X, V, weights = None):
    '''
    input:
        histories - list of trajectories, with q in index form
        X - price
        V - list of value vectors, one for each trajectory
        weights - list of weights, one for each trajectory
    output:
        profit - profit for the seller
    '''
    if weights is None:
        weights = [1.0 for _ in range(len(histories))]
        
    profit = 0.0
    for tau, v, w in zip(histories, V, weights):
        surplus = v - X
        # find the q that maximizes surplus, given that q is in tau
        q = max(tau, key=lambda q: surplus[q])
        profit += X[q] * w
    return profit



def milp(mu, histories, n_samples, n_types, Q, T, P, u):
    '''
    input: 
        mu - probability distribution over types
        histories - list of histories of length n_samples sampled according to the transition probabilities P
        n_samples - number of samples
        n_types - number of types
        Q - number of actions
        T - number of time steps
        P - transition probabilities
        u - utility for each outcome
    output:
        payment - optimal payment with respect to strategy sigma for the buyer
    '''
    # Build the LP model
    model = gp.Model('milp')
    model.setParam('OutputFlag', 0)

    # Add variables
    x = []
    for i in range(Q):
        x.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='x'+str(i)))
    model.addConstr(x[0] == 0.0)
    y = []
    z = []
    for i in range(n_types):
        y_theta = []
        z_theta = []
        for j in range(n_samples):
            y_theta_j = []
            z_theta_j = []
            q_tau = histories[j]
            for q in range(Q):
                y_theta_j.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='y'+str(i)+str(j)+str(q)))
                z_theta_j.append(model.addVar(vtype=GRB.BINARY, name='z'+str(i)+str(j)+str(q)))
                if q not in q_tau:
                    model.addConstr(z_theta_j[q] == 0)
            model.addConstr(gp.quicksum(z_theta_j) == 1.0)
            y_theta.append(y_theta_j)
            z_theta.append(z_theta_j)
        y.append(y_theta)
        z.append(z_theta)
    a = []
    for i in range(n_types):
        a_i = []
        for j in range(n_samples):
            a_i.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='a'+str(i)+str(j)))
        a.append(a_i)
    
    # Set objective
    obj = 0.0
    for i in range(n_types):
        for j in range(n_samples):
            q_tau = histories[j]
            for q in q_tau:
            #for q in range(Q):
                obj += mu[i]*y[i][j][q]*(1.0 / float(n_samples))
    
    M_c = float(2**10)
    for i in range(n_types):
        for j in range(n_samples):
            q_tau = histories[j]
            for q in q_tau:
                buyer_utility = u[i][q] - x[q]
                model.addConstr(a[i][j] - buyer_utility >= 0, name='lb'+str(i)+str(j)+str(q))
                model.addConstr(a[i][j] - buyer_utility <= M_c*(1.0 - z[i][j][q]), name='ub'+str(i)+str(j)+str(q))

    for i in range(n_types):
        for j in range(n_samples):
            q_tau = histories[j]
            for q in q_tau:
                model.addConstr(y[i][j][q] <= x[q], name='feasible_x'+str(i)+str(j)+str(q))
                model.addConstr(y[i][j][q] <= M_c*z[i][j][q] , name='feasible_z'+str(i)+str(j)+str(q))
                model.addConstr(y[i][j][q] >= x[q] - M_c*(1.0 - z[i][j][q]), name='feasible_xz'+str(i)+str(j)+str(q))

    model.setObjective(obj, GRB.MAXIMIZE)
    model.write('specific_strategy_program.lp')
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(model.printStats())
        return model.objVal, [x[i].x for i in range(Q)], 'optimal'
    elif model.status == GRB.Status.UNBOUNDED:
        return np.PINF, None, 'unbounded'
    else:
        return np.NINF, None, 'infeasible'

def main():
    problems = None
    samples = [10, 50, 100, 500, 1000, 5000, 10000]
    for i, n_samples in tqdm(enumerate(samples)):
    #generate_problem(2, T, 10)
        with open(f'problems_3_5_100.pkl', 'rb') as f:
            problems = pickle.load(f)
        n_problems = 1
        logger = open('log.txt', 'w')
        columns=['problem_id', 'opt_obj', 'solver_time']
        df = pd.DataFrame(columns=columns)
        for i in range(n_problems):
            n_types, Q, T, P, C, u = problems[i]['ntypes'], problems[i]['Q'], problems[i]['T'], problems[i]['P'], problems[i]['c'], problems[i]['v']
            #n_samples = 10
            mu = np.ones(n_types) / float(n_types)
            histories = []
            for j in range(n_samples):
                histories.append(np.random.choice(Q, T))
            #histories = cartesian_product_transpose(*([np.arange(10)]*5))
            #n_samples = len(histories)
            #histories = sample_history(n_samples, Q, P)
            if not check_valid_P(P):
                continue
            start_t = time.time()
            opt_obj, opt_sol, marker = milp(mu, histories, n_samples, n_types, Q, T, P, u)
            solver_time = time.time() - start_t
            logger.write('problem_id: {}, opt_obj: {}, solver_time: {}, program_status: {}\n'.format(i, opt_obj, solver_time, marker))
            df = pd.concat([df, pd.DataFrame([[i, opt_obj, solver_time]], columns=columns)])
        logger.close()
        df.to_csv(f'result/n_samples_{n_samples}.csv')

if __name__ == '__main__':
    main()
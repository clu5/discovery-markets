import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
from tqdm import tqdm


def generate_random_instance(Q, T, ntypes):
    # pi = [1.0/ntypes for _ in range(ntypes)]
    # P = np.random.rand(T, Q)
    # P = np.apply_along_axis(lambda x: x - (np.sum(x) - 1)/len(x), 1, P)
    # V = np.random.rand(ntypes, Q)
    # c = np.random.rand(T)
    # return pi, P, V, c
    pi = [1.0/ntypes for _ in range(ntypes)]
    P = np.random.rand(T, Q)
    P = np.apply_along_axis(lambda x: x - (np.sum(x) - 1)/len(x), 1, P)
    P_0 = np.zeros(Q)
    P = np.concatenate((P, P_0[None, :]), axis=0)
    V = np.random.rand(ntypes, Q)
    c = np.random.rand(T)
    c = np.concatenate((c, [0]))
    return pi, P, V, c


def compute_best_response(P, v, c, x):
    T, Q = len(P), len(P[0])
    opt_u, opt_t = -np.inf, -1
    for t in range(T):
        u = 0
        for q in range(Q):
            u += P[t][q]*(v[q] - x[q])
        u -= c[t]
        if u > opt_u:
            opt_u, opt_t = u, t
    return opt_t, opt_u
    

def myerson_price_rev(pi, P, V, c):
    ntypes = len(V)
    T, Q = len(P), len(P[0])
    myerson_prices = [0.0 for q in range(Q)]
    for q in range(Q):
        values = V[:, q]
        values_indexs = [(values[i], i) for i in range(ntypes)]
        values_indexs.sort(reverse=True)
        cur_p, opt_r, opt_x = 0.0, -np.inf, -np.inf
        for k in range(ntypes):
            cur_p += pi[values_indexs[k][1]]
            cur_r = values_indexs[k][0]*cur_p
            if cur_r > opt_r:
                opt_r, opt_x = cur_r, values_indexs[k][0]
        myerson_prices[q] = opt_x
    agent_actions = [compute_best_response(P, V[_,:], c, myerson_prices) for _ in range(ntypes)]
    rev = 0.0
    for theta in range(ntypes):
        theta_rev = 0.0
        for q in range(Q):
            theta_rev += P[agent_actions[theta][0]][q]*myerson_prices[q]
        rev += pi[theta]*theta_rev
    return rev
    

def compute_bayesian_price_deterministic_outcome(pi, P, V, c):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    M = gp.Model("bayesian_price", env=env)

    ntypes = len(V)
    T, Q = len(P), len(P[0])
    
    # Create variables: leader strategy x, follower strategy y, follower utilities a_theta, and linear variables z
    
    #   x_i = x[q i]
    x = []
    for i in range(Q):
        x += [M.addVar(lb=0.0, name="x{}".format(i))]
    M.update()
    
    #   y^theta_t = y[Type theta][Action t]
    y = []
    for theta in range(ntypes):
        y_theta = []
        for t in range(T):
            y_theta += [M.addVar(vtype=gp.GRB.BINARY, name="y{}_{}".format(theta, t))]
        M.update()
        M.addConstr(gp.quicksum(y_theta) == 1.0, name="sa{}".format(theta))
        y.append(y_theta)

    #   a_theta = a[Type theta]
    a = []
    for theta in range(ntypes):
        a += [M.addVar(name="a{}".format(theta))]
    M.update()

    # z[q][theta][t] = y[theta][t] * x[q]
    z = []
    for q in range(Q):
        z_q = []
        for theta in range(ntypes):
            z_q_theta = []
            for t in range(T):
                z_q_theta += [M.addVar(lb=0.0, name="z{}_{}_{}".format(q, theta, t))]
            z_q.append(z_q_theta)
        z.append(z_q)
    M.update()

    # Create expression for objective function
    objective = 0

    for theta in range(ntypes):
        for t in range(T):
            for q in range(Q):
                objective += pi[theta]*z[q][theta][t]*P[t][q]
    M.setObjective(objective, gp.GRB.MAXIMIZE)

    # Add IC constraints for follower
    M_c = float(2**10) # Large constant assumed to be larger than any achievable utility
    for t in range(T):
        for theta in range(ntypes):
            follower_utility = 0
            for q in range(Q):
                follower_utility += (V[theta][q] - x[q])*P[t][q]
            follower_utility -= c[t]

            M.addConstr(a[theta] - follower_utility >= 0, name="lb{}_{}".format(theta, t))
            M.addConstr(a[theta] - follower_utility <= M_c*(1 - y[theta][t]), name="ub{}_{}".format(theta, t))

    # Add feasiblity constraint for z
    for q in range(Q):
        for theta in range(ntypes):
            for t in range(T):
                M.addConstr(z[q][theta][t] <= x[q], name="feas_x_{}".format(q))
                M.addConstr(z[q][theta][t] <= y[theta][t], name="feas_y_{}_{}".format(theta, t))
                M.addConstr(z[q][theta][t] >= x[q] - M_c*(1 - y[theta][t]), name="feas_xy_{}_{}_{}".format(q, theta, t))

    M.optimize()
    return M


def main():
    # Example from the paper
    Q, T, ntypes = 100, 50, 10
    pi = [1.0/ntypes for _ in range(ntypes)]
    P = np.random.rand(T, Q)
    P = np.apply_along_axis(lambda x: x - (np.sum(x) - 1)/len(x), 1, P)
    P_0 = np.zeros(Q)
    P = np.concatenate((P, P_0[None, :]), axis=0)
    V = np.random.rand(ntypes, Q)
    c = np.random.rand(T)
    c = np.concatenate((c, [0]))
    # pi = [0.5, 0.5]
    # P = [[0.5, 0.5], [0.5, 0.5]]
    # V = [[1, 1], [1, 0]]
    # c = [0, 0]

    M = compute_bayesian_price_deterministic_outcome(pi, P, V, c)
    print("Optimal value:", M.objVal)
    print("Leader strategy:", [x.x for x in M.getVars() if x.varName[0] == "x"])
    print("Follower strategy:", [y.x for y in M.getVars() if y.varName[0] == "y"])
    print("Follower utilities:", [a.x for a in M.getVars() if a.varName[0] == "a"])


def generate_csv():
    trials = 100
    columns=["Bayesian_rev", "Myerson_rev", "Myerson>Bayesian", "Bayesian>Myerson"]
    results = pd.DataFrame([], columns=columns)
    
    for k in tqdm(range(trials)):
        Q, T, ntypes = 10, 5, 5
        pi, P, V, c = generate_random_instance(Q, T, ntypes)
        dynamic_model = compute_bayesian_price_deterministic_outcome(pi, P, V, c)
        dynamic_rev = dynamic_model.objVal
        myerson_rev = myerson_price_rev(pi, P, V, c)
        new_row = pd.DataFrame([[dynamic_rev, myerson_rev, myerson_rev-dynamic_rev>0.00001, dynamic_rev-myerson_rev>0.00001]], columns=columns)
        results = pd.concat([results, new_row])
    results.to_csv("results/myerson_bayesian.csv")

if __name__ == "__main__":
    generate_csv()

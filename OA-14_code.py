"""
Group members:
Artem Khomytskyi - 20221686 | Timofii Kuzmenko - 20221690 | Davyd Azarov - 20221688
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import combinations
import time

"""
Flying Robots Optimization
-------------------------
Implements optimization for N autonomous flying robots to minimize fuel consumption or total distance traveled,
subject to physical (Newton's second law) and operational (communication, waypoints, altitude) constraints.
- Uses CVXPY with SCS solver for fuel minimization (quadratic cost) and ECOS for distance minimization (norm-based cost).
- Implements three cost functions: fuel (quadratic), distance (L2 norm), and squared distance (L2 norm squared).
- Includes strict constraint version (no tolerance) for fuel cost to compare with relaxed version.
- Generates all required plots for visualization and validates constraint satisfaction.
- Parameters and waypoints are set to create a challenging coordination problem (waypoints spaced > communication radius).
"""

# ----------------------
# Parameters
# ----------------------
N = 5  # Number of robots
h = 0.1  # Time step
T = 50  # Number of time steps
mi = 1.0  # Robot mass (equal for all)
beta = 0.5  # Drag coefficient
g = np.array([0, 0, -9.81])  # Gravitational acceleration
hmax = 7.0  # Maximum altitude
d = 7.0  # Communication radius (increased to cover waypoint spread)
np.random.seed(42)
# Relaxation epsilon for constraints (to improve solver stability)
EPS = 1e-3  # Small tolerance for altitude and communication constraints

# ----------------------
# Waypoint generation
# ----------------------
# Generate waypoints spaced farther apart than communication radius to create coordination challenge
waypoints = []  # List of (robot_idx, waypoint, time_step)
assigned_times = np.linspace(20, T-20, N, dtype=int)
for i in range(N):
    angle = 2 * np.pi * i / N
    xw = 2.0 * np.cos(angle)
    yw = 2.0 * np.sin(angle)
    zw = 5
    waypoints.append((i, np.array([xw, yw, zw]), assigned_times[i]))

# Initial and final positions
x0 = np.zeros((N, 3))  # All robots start at origin
# Final positions: only enforce z=0 (ground); x, y are flexible for feasibility
xT = np.zeros((N, 3))
for i in range(N):
    for (j, w, t) in waypoints:
        if j == i:
            xT[i] = np.array([w[0], w[1], 0])

# ----------------------
# Optimization variables
# ----------------------
# x[i,k,:]: position of robot i at time k (3D)
# v[i,k,:]: velocity of robot i at time k (3D)
# u[i,k,:]: thrust of robot i at time k (3D)
x = cp.Variable((N, T+1, 3))
v = cp.Variable((N, T+1, 3))
u = cp.Variable((N, T+1, 3))

# ----------------------
# Constraints (relaxed)
# ----------------------
constraints = []

# Initial and final conditions
for i in range(N):
    constraints += [x[i,0,:] == x0[i]]  # Initial position
    constraints += [x[i,T,2] == 0]      # Final z=0 (ground)
    constraints += [v[i,0,:] == 0, v[i,T,:] == 0]  # Start/end at rest

# Dynamics (discretized Newton's second law)
for i in range(N):
    for k in range(T):
        a = (u[i,k,:] - beta*v[i,k,:] + mi*g) / mi
        constraints += [x[i,k+1,:] == x[i,k,:] + h*v[i,k,:]]
        constraints += [v[i,k+1,:] == v[i,k,:] + h*a]

# Altitude constraints (relaxed with EPS for solver stability)
for i in range(N):
    for k in range(T+1):
        constraints += [x[i,k,2] <= hmax + EPS]
        constraints += [x[i,k,2] >= -EPS]
# Communication constraints (relaxed with EPS for solver stability)
for k in range(T+1):
    for i, j in combinations(range(N), 2):
        constraints += [cp.norm(x[i,k,:] - x[j,k,:], 2) <= d + EPS]
# Waypoint constraints
for (i, w, t) in waypoints:
    constraints += [x[i,t,:] == w]

# ----------------------
# Cost functions
# ----------------------
# 1. Fuel (quadratic): sum over i, k of ||u_i(k)||^2
fuel_cost = cp.sum([cp.sum_squares(u[i,k,:]) for i in range(N) for k in range(T+1)])
# 2. Distance (L2 norm): sum over i, k of ||x_i(k+1) - x_i(k)||
dist_cost = cp.sum([cp.norm(x[i,k+1,:] - x[i,k,:], 2) for i in range(N) for k in range(T)])
# 3. Squared distance: sum over i, k of ||x_i(k+1) - x_i(k)||^2
squared_dist_cost = cp.sum([cp.sum_squares(x[i,k+1,:] - x[i,k,:]) for i in range(N) for k in range(T)])

# ----------------------
# Solve for fuel cost (SCS)
# ----------------------
print("=== Flying Robots Optimization ===")
print(f"Configuration: {N} robots, {T} time steps\nGenerated {len(waypoints)} waypoints\n")

print("Solving fuel minimization (SCS)...")
fuel_start = time.time()
prob_fuel = cp.Problem(cp.Minimize(fuel_cost), constraints)
prob_fuel.solve(solver=cp.SCS, eps=1e-8, max_iters=10000, verbose=False)
fuel_end = time.time()
xf = x.value.copy() if x.value is not None else None
vf = v.value.copy() if v.value is not None else None
uf = u.value.copy() if u.value is not None else None
fuel_status = prob_fuel.status
fuel_obj = prob_fuel.value if prob_fuel.value is not None else float('inf')
fuel_time = fuel_end - fuel_start
fuel_solver = 'SCS'

# ----------------------
# Solve for distance cost (ECOS)
# ----------------------
print("Solving distance minimization (ECOS)...")
dist_start = time.time()
prob_dist = cp.Problem(cp.Minimize(dist_cost), constraints)
prob_dist.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, verbose=False)
dist_end = time.time()
xd = x.value.copy() if x.value is not None else None
vd = v.value.copy() if v.value is not None else None
ud = u.value.copy() if u.value is not None else None
dist_status = prob_dist.status
dist_obj = prob_dist.value if prob_dist.value is not None else float('inf')
dist_time = dist_end - dist_start
dist_solver = 'ECOS'

# ----------------------
# Solve for squared distance cost (ECOS)
# ----------------------
print("Solving squared distance minimization (ECOS)...")
squared_dist_start = time.time()
prob_squared_dist = cp.Problem(cp.Minimize(squared_dist_cost), constraints)
prob_squared_dist.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, verbose=False)
squared_dist_end = time.time()
xsd = x.value.copy() if x.value is not None else None
vsd = v.value.copy() if v.value is not None else None
usd = u.value.copy() if u.value is not None else None
squared_dist_status = prob_squared_dist.status
squared_dist_obj = prob_squared_dist.value if prob_squared_dist.value is not None else float('inf')
squared_dist_time = squared_dist_end - squared_dist_start
squared_dist_solver = 'ECOS'

# ----------------------
# Strict constraints (no relaxation, EPS=0) for fuel cost
# ----------------------
strict_constraints = []

# Initial and final conditions
for i in range(N):
    strict_constraints += [x[i,0,:] == x0[i]]
    strict_constraints += [x[i,T,2] == 0]
    strict_constraints += [v[i,0,:] == 0, v[i,T,:] == 0]
# Dynamics
for i in range(N):
    for k in range(T):
        a = (u[i,k,:] - beta*v[i,k,:] + mi*g) / mi
        strict_constraints += [x[i,k+1,:] == x[i,k,:] + h*v[i,k,:]]
        strict_constraints += [v[i,k+1,:] == v[i,k,:] + h*a]
# Altitude constraints (strict)
for i in range(N):
    for k in range(T+1):
        strict_constraints += [x[i,k,2] <= hmax]
        strict_constraints += [x[i,k,2] >= 1e-6]
# Communication constraints (strict)
for k in range(T+1):
    for i, j in combinations(range(N), 2):
        strict_constraints += [cp.norm(x[i,k,:] - x[j,k,:], 2) <= d]
# Waypoint constraints
for (i, w, t) in waypoints:
    strict_constraints += [x[i,t,:] == w]

# Solve fuel minimization with strict constraints
print("Solving fuel minimization with strict constraints (ECOS)...")
strict_fuel_start = time.time()
prob_strict_fuel = cp.Problem(cp.Minimize(fuel_cost), strict_constraints)
prob_strict_fuel.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, verbose=False)
strict_fuel_end = time.time()
xsf = x.value.copy() if x.value is not None else None
vsf = v.value.copy() if v.value is not None else None
usf = u.value.copy() if u.value is not None else None
strict_fuel_status = prob_strict_fuel.status
strict_fuel_obj = prob_strict_fuel.value if prob_strict_fuel.value is not None else float('inf')
strict_fuel_time = strict_fuel_end - strict_fuel_start
strict_fuel_solver = 'ECOS'

# ----------------------
# Constraint violation reporting function
# ----------------------
def get_violations(xval, hmax, d, eps):
    if xval is None:
        return ["No solution found"], [], []
    altitude_viol = []
    below_ground_viol = []
    comm_viol = []
    viol_tol = 1e-6  # Stricter tolerance for reporting violations
    for i in range(N):
        if np.any(xval[i,:,2] > hmax + viol_tol):
            altitude_viol.append(f"Robot {i+1}: Altitude violation")
        if np.any(xval[i,:,2] < -viol_tol):
            below_ground_viol.append(f"Robot {i+1}: Below ground")
    for k in range(xval.shape[1]):
        for i, j in combinations(range(N), 2):
            dist_ij = np.linalg.norm(xval[i,k,:] - xval[j,k,:])
            if dist_ij > d + viol_tol:
                comm_viol.append(f"Communication violation at t={k}: Robots {i+1}-{j+1}, dist={dist_ij:.3f}")
    return altitude_viol, below_ground_viol, comm_viol

# ----------------------
# Results output and constraint violation reporting
# ----------------------
print("\n=== Results ===")
print(f"Fuel Objective: Status = {fuel_status}, Value = {fuel_obj:.3f}, Time = {fuel_time:.2f}s, Solver = {fuel_solver}")
print(f"Distance Objective: Status = {dist_status}, Value = {dist_obj:.3f}, Time = {dist_time:.2f}s, Solver = {dist_solver}")
print(f"Squared Distance Objective: Status = {squared_dist_status}, Value = {squared_dist_obj:.3f}, Time = {squared_dist_time:.2f}s, Solver = {squared_dist_solver}")
print(f"Strict Fuel Objective: Status = {strict_fuel_status}, Value = {strict_fuel_obj:.3f}, Time = {strict_fuel_time:.2f}s, Solver = {strict_fuel_solver}\n")

# Fuel solution violations
alt_v, below_v, comm_v = get_violations(xf, hmax, d, EPS)
if len(alt_v) + len(below_v) + len(comm_v) == 0:
    print("Fuel solution: All constraints satisfied.\n")
else:
    print(f"Fuel solution violations: {len(alt_v) + len(below_v) + len(comm_v)}\n")
    for v in alt_v + below_v + comm_v:
        print(v)
    print()

# Distance solution violations
alt_vd, below_vd, comm_vd = get_violations(xd, hmax, d, EPS)
if len(alt_vd) + len(below_vd) + len(comm_vd) == 0:
    print("Distance solution: All constraints satisfied.\n")
else:
    print(f"Distance solution violations: {len(alt_vd) + len(below_vd) + len(comm_vd)}\n")
    for v in alt_vd + below_vd + comm_vd:
        print(v)
    print()

# Squared distance solution violations
alt_vsd, below_vsd, comm_vsd = get_violations(xsd, hmax, d, EPS)
if len(alt_vsd) + len(below_vsd) + len(comm_vsd) == 0:
    print("Squared Distance solution: All constraints satisfied.\n")
else:
    print(f"Squared Distance solution violations: {len(alt_vsd) + len(below_vsd) + len(comm_vsd)}\n")
    for v in alt_vsd + below_vsd + comm_vsd:
        print(v)
    print()

# Strict fuel solution violations
alt_vsf, below_vsf, comm_vsf = get_violations(xsf, hmax, d, 0)
if len(alt_vsf) + len(below_vsf) + len(comm_vsf) == 0:
    print("Strict Fuel solution: All constraints satisfied.\n")
else:
    print(f"Strict Fuel solution violations: {len(alt_vsf) + len(below_vsf) + len(comm_vsf)}\n")
    for v in alt_vsf + below_vsf + comm_vsf:
        print(v)
    print()

# Metrics comparison (total fuel and distance for each solution)
fuel_total_fuel = np.sum(np.linalg.norm(uf, axis=2))*h if uf is not None else float('inf')
fuel_total_dist = np.sum(np.linalg.norm(np.diff(xf, axis=1), axis=2)) if xf is not None else float('inf')
dist_total_fuel = np.sum(np.linalg.norm(ud, axis=2))*h if ud is not None else float('inf')
dist_total_dist = np.sum(np.linalg.norm(np.diff(xd, axis=1), axis=2)) if xd is not None else float('inf')
squared_dist_total_fuel = np.sum(np.linalg.norm(usd, axis=2))*h if usd is not None else float('inf')
squared_dist_total_dist = np.sum(np.linalg.norm(np.diff(xsd, axis=1), axis=2)) if xsd is not None else float('inf')
strict_fuel_total_fuel = np.sum(np.linalg.norm(usf, axis=2))*h if usf is not None else float('inf')
strict_fuel_total_dist = np.sum(np.linalg.norm(np.diff(xsf, axis=1), axis=2)) if xsf is not None else float('inf')

print("Metrics Comparison:")
print(f"Total Fuel - Fuel Opt: {fuel_total_fuel:.3f}, Distance Opt: {dist_total_fuel:.3f}, Squared Dist Opt: {squared_dist_total_fuel:.3f}, Strict Fuel Opt: {strict_fuel_total_fuel:.3f}")
print(f"Total Distance - Fuel Opt: {fuel_total_dist:.3f}, Distance Opt: {dist_total_dist:.3f}, Squared Dist Opt: {squared_dist_total_dist:.3f}, Strict Fuel Opt: {strict_fuel_total_dist:.3f}\n")

# ----------------------
# Visualization functions (trajectories, thrust, fuel, pairwise distances)
# ----------------------
def plot_trajectories(xval, waypoints, title):
    if xval is None:
        print(f"Cannot plot {title}: No solution found")
        return
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0,1,N))
    for i in range(N):
        ax.plot(xval[i,:,0], xval[i,:,1], xval[i,:,2], label=f'Robot {i+1}', color=colors[i])
        # Plot waypoints
        for (j, w, t) in waypoints:
            if j == i:
                ax.scatter(w[0], w[1], w[2], marker='*', s=100, color=colors[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_thrust(uval, title):
    if uval is None:
        print(f"Cannot plot {title}: No solution found")
        return
    plt.figure(figsize=(10,6))
    for i in range(N):
        thrust = np.linalg.norm(uval[i,:,:], axis=1)
        plt.plot(thrust, label=f'Robot {i+1}')
    plt.xlabel('Time step')
    plt.ylabel('Thrust magnitude')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_cumulative_fuel(uval, title):
    if uval is None:
        print(f"Cannot plot {title}: No solution found")
        return
    plt.figure(figsize=(10,6))
    for i in range(N):
        thrust = np.linalg.norm(uval[i,:,:], axis=1)
        cum_fuel = np.cumsum(thrust)*h
        plt.plot(cum_fuel, label=f'Robot {i+1}')
    plt.xlabel('Time step')
    plt.ylabel('Cumulative fuel')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_pairwise_distances(xval, title):
    if xval is None:
        print(f"Cannot plot {title}: No solution found")
        return
    plt.figure(figsize=(10,6))
    for i, j in combinations(range(N), 2):
        dist = np.linalg.norm(xval[i,:,:] - xval[j,:,:], axis=1)
        plt.plot(dist, label=f'R{i+1}-R{j+1}')
    plt.xlabel('Time step')
    plt.ylabel('Pairwise distance')
    plt.title(title)
    plt.legend()
    plt.show()

# ----------------------
# Generate and show plots for each solution
# ----------------------
# Fuel cost
plot_trajectories(xf, waypoints, '3D trajectories (fuel cost)')
plot_thrust(uf, 'Thrust magnitude over time (fuel cost)')
plot_cumulative_fuel(uf, 'Cumulative fuel consumption (fuel cost)')
plot_pairwise_distances(xf, 'Pairwise distances (fuel cost)')

# Distance cost
plot_trajectories(xd, waypoints, '3D trajectories (distance cost)')
plot_thrust(ud, 'Thrust magnitude over time (distance cost)')
plot_cumulative_fuel(ud, 'Cumulative fuel consumption (distance cost)')
plot_pairwise_distances(xd, 'Pairwise distances (distance cost)')

# Squared distance cost
plot_trajectories(xsd, waypoints, '3D trajectories (squared distance cost)')
plot_thrust(usd, 'Thrust magnitude over time (squared distance cost)')
plot_cumulative_fuel(usd, 'Cumulative fuel consumption (squared distance cost)')
plot_pairwise_distances(xsd, 'Pairwise distances (squared distance cost)')

# Strict fuel cost
plot_trajectories(xsf, waypoints, '3D trajectories (strict fuel cost)')
plot_thrust(usf, 'Thrust magnitude over time (strict fuel cost)')
plot_cumulative_fuel(usf, 'Cumulative fuel consumption (strict fuel cost)')
plot_pairwise_distances(xsf, 'Pairwise distances (strict fuel cost)')

print("Done.\n")
"""Independent explicit charge/discharge MILP; constructed checks are Simulated."""

import numpy as np

from scipy import sparse

from scipy.optimize import Bounds, LinearConstraint, milp

from model.planning import Physical

class ExplicitBuilder:
    def __init__(self):
        self.cost, self.low, self.high, self.integer = [], [], [], []
        self.rows, self.cols, self.values, self.rowlow, self.rowhigh = [], [], [], [], []

    def variables(self, shape, *, low=0., high=np.inf, cost=0., integer=0):
        count = int(np.prod(shape))
        indices = np.arange(len(self.cost), len(self.cost)+count).reshape(shape)
        for target, value in ((self.low, low), (self.high, high),
                              (self.cost, cost), (self.integer, integer)):
            target.extend(np.broadcast_to(value, shape).ravel().tolist())
        return indices

    def constraint(self, coefficients, low=-np.inf, high=np.inf):
        r = len(self.rowlow)
        for index, value in coefficients:
            self.rows.append(r)
            self.cols.append(int(index))
            self.values.append(float(value))
        self.rowlow.append(float(low))
        self.rowhigh.append(float(high))

    def equal(self, coefficients, value=0.):
        self.constraint(coefficients, low=value, high=value)

    def solve(self):
        matrix = sparse.coo_matrix((self.values, (self.rows, self.cols)),
                                   shape=(len(self.rowlow), len(self.cost))).tocsr()
        answer = milp(np.asarray(self.cost), integrality=np.asarray(self.integer),
                      bounds=Bounds(np.asarray(self.low), np.asarray(self.high)),
                      constraints=LinearConstraint(matrix, self.rowlow, self.rowhigh),
                      options={"mip_rel_gap": 1e-10, "time_limit": 30.})
        if not answer.success:
            raise AssertionError(f"Independent explicit MILP failed: {answer.status}: {answer.message}")
        constraint_values = matrix@answer.x
        violation = max(0., float(np.max(np.asarray(self.rowlow)-constraint_values)),
                         float(np.max(constraint_values-np.asarray(self.rowhigh))),
                         float(np.max(np.asarray(self.low)-answer.x)),
                         float(np.max(answer.x-np.asarray(self.high))))
        return answer, violation

def explicit_milp(load, pv, price, s0, *, physical, kind="dayahead", original_q=None,
                  fixed_m=None, fee="B", lam=0., target=None, groups=None, split=0,
                  allow_amend=True, first_discharge_cap=None,
                  terminal_equal=False, terminal_floor=None, charge_limit=None, discharge_limit=None):
    """Explicit physical and contract formulation used only by this validator."""
    load, pv, price = np.asarray(load), np.asarray(pv), np.asarray(price)
    k, n = load.shape
    labels, group_index = np.unique(np.zeros(k, int) if groups is None else groups, return_inverse=True)
    ng = len(labels)
    weights = np.bincount(group_index, minlength=ng)/k
    model = ExplicitBuilder()
    charge_limit = physical.power_energy if charge_limit is None else charge_limit
    discharge_limit = physical.power_energy if discharge_limit is None else discharge_limit
    q = model.variables((n,), cost=price if kind != "fixed" else 0.)
    m = model.variables((ng, n))
    charge = model.variables((ng, n), high=charge_limit)
    discharge_high = np.asarray([np.minimum(discharge_limit, load[group_index == g].min(axis=0))
                                 for g in range(ng)])
    if first_discharge_cap is not None:
        discharge_high[:, 0] = np.minimum(discharge_high[:, 0], max(0., first_discharge_cap))
    discharge = model.variables((ng, n), high=discharge_high)
    state = model.variables((ng, n+1), low=physical.s_min, high=physical.s_max)
    mode = model.variables((ng, n), high=1., integer=1)
    called = model.variables((k, n))
    emergency = model.variables((k, n), cost=np.broadcast_to(physical.emergency_multiple*price/k, (k, n)))
    curtailed = model.variables((k, n), high=pv)
    up = down = None
    if kind in ("amend", "signal"):
        up = model.variables((ng, n), cost=weights[:, None]*1.5*price)
        down = model.variables((ng, n), cost=weights[:, None]*(.5 if fee == "A" else -.5)*price)
        # With up-down=m-q, increasing up and down together costs +2p under A
        # or +p under B. A minimum therefore cannot include fictitious round trips.
        for g in range(ng):
            for t in range(n):
                model.equal([(m[g, t], 1.), (q[t], -1.), (up[g, t], -1.), (down[g, t], 1.)])
    if kind == "amend":
        for t in range(n):
            model.equal([(q[t], 1.)], original_q[t])
    elif kind == "fixed":
        for t in range(n):
            model.equal([(q[t], 1.)], fixed_m[t])
            for g in range(ng):
                model.equal([(m[g, t], 1.)], fixed_m[t])
    elif kind == "dayahead":
        for g in range(ng):
            for t in range(n):
                model.equal([(m[g, t], 1.), (q[t], -1.)])
    elif kind != "signal":
        raise ValueError(kind)
    if kind == "signal":
        for g in range(ng):
            for t in range(split if allow_amend else n):
                model.equal([(m[g, t], 1.), (q[t], -1.)])
            if g:
                # Independent expression of nonanticipativity: actual battery
                # charge and discharge commands coincide before the signal.
                for t in range(split):
                    model.equal([(charge[g, t], 1.), (charge[0, t], -1.)])
                    model.equal([(discharge[g, t], 1.), (discharge[0, t], -1.)])
    y = model.variables((ng,), high=physical.s_max if target is None else target,
                        cost=-lam*weights) if lam else None
    for g in range(ng):
        if s0 is not None:
            model.equal([(state[g, 0], 1.)], s0)
        if terminal_equal:
            model.equal([(state[g, -1], 1.), (state[g, 0], -1.)])
        if terminal_floor is not None:
            model.constraint([(state[g, -1], 1.)], low=terminal_floor)
        for t in range(n):
            model.equal([(state[g, t+1], 1.), (state[g, t], -1.),
                         (charge[g, t], -physical.eta_c), (discharge[g, t], 1/physical.eta_d)])
            model.constraint([(charge[g, t], 1.), (mode[g, t], -charge_limit)], high=0.)
            model.constraint([(discharge[g, t], 1.), (mode[g, t], discharge_limit)],
                             high=discharge_limit)
        if y is not None:
            model.constraint([(y[g], 1.), (state[g, -1], -1.)], high=0.)
    for s in range(k):
        g = group_index[s]
        for t in range(n):
            model.constraint([(called[s, t], 1.), (m[g, t], -1.)], high=0.)
            model.equal([(called[s, t], 1.), (emergency[s, t], 1.),
                         (discharge[g, t], 1.), (charge[g, t], -1.), (curtailed[s, t], -1.)],
                        load[s, t]-pv[s, t])
    answer, violation = model.solve()
    x = answer.x
    return {
        "q": x[q], "m_by_group": x[m], "c_by_group": x[charge], "b_by_group": x[discharge],
        "soc_by_group": x[state], "e": x[emergency], "called": x[called], "curtailed": x[curtailed],
        "group_weights": weights, "scenario_groups": group_index,
        "objective": float(answer.fun), "mip_gap": float(answer.mip_gap),
        "max_constraint_violation": violation,
        "up": None if up is None else x[up], "down": None if down is None else x[down],
    }

def cases():
    generated = []
    for index, (n, k) in enumerate(((4, 2), (5, 3), (6, 4), (4, 4), (5, 2), (6, 3))):
        rng = np.random.default_rng(81031+index)
        load = np.round(rng.uniform(.3, 9., (k, n)), 3)
        pv = np.round(rng.uniform(0., 7., (k, n)), 3)
        if index in (1, 4):
            load[0, 0] = 0.
        price = np.round(rng.uniform(.2, 1.9, n), 3)
        physical = Physical(eta_c=.84+.02*(index%3), eta_d=.88+.015*(index%2),
                            s_min=1., s_max=12., power_energy=3.8, emergency_multiple=5.)
        generated.append({
            "name": f"constructed_n{n}_k{k}_{index}", "seed": 81031+index,
            "load": load, "pv": pv, "price": price, "s0": 4.+index,
            "physical": physical, "original_q": np.round(rng.uniform(0., 8., n), 3),
            "fixed_m": np.round(rng.uniform(0., 8., n), 3),
            "lam": (0., .65, 1.8)[index%3], "target": (None, 8.4, 7.5)[index%3],
            "groups": np.asarray([10+10*(s%min(k, 3)) for s in range(k)]),
            "split": n//2,
        })
    return generated

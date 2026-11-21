"""
Symbolic and numerical ground-truth checks for the adversarial physics traps.
Every check ends in a real assertion against the value stored in problems/*.json.
"""

import json
import os
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "problems", name)) as f:
        return json.load(f)["ground_truth"]


def verify_trap_01():
    print("\n--- Trap 1: sphere separation ---")
    c, g, R, m = sp.symbols("c g R m", positive=True)  # c = cos(theta)
    v_sq = 2 * g * R * (1 - c)                 # energy conservation
    N = m * g * c - m * v_sq / R               # radial equation
    sol = sp.solve(sp.Eq(N, 0), c)[0]
    print(f"N = {sp.simplify(N)}  ->  cos(theta_c) = {sol}")
    assert sol == sp.Rational(load("trap_01_sphere_separation.json")["cos_theta_c"])
    print("PASSED")


def verify_trap_02():
    print("\n--- Trap 2: bound charge on a dielectric shell ---")
    r, a, b, P0 = sp.symbols("r a b P0", positive=True)
    gt = load("trap_02_bound_charge_shell.json")
    sym = {"P0": P0, "a": a, "b": b}   # reuse our symbols so assumptions match
    P_r = P0 * r / b
    rho_b = sp.simplify(-sp.diff(r ** 2 * P_r, r) / r ** 2)
    sigma_outer = P_r.subs(r, b)        # normal = +r_hat
    sigma_inner = -P_r.subs(r, a)       # normal = -r_hat
    Q_total = (sp.integrate(rho_b * 4 * sp.pi * r ** 2, (r, a, b))
               + sigma_outer * 4 * sp.pi * b ** 2
               + sigma_inner * 4 * sp.pi * a ** 2)
    print(f"rho_b = {rho_b}, sigma_outer = {sigma_outer}, sigma_inner = {sigma_inner}")
    assert sp.simplify(rho_b - sp.sympify(gt["rho_b"], locals=sym)) == 0
    assert sp.simplify(sigma_outer - sp.sympify(gt["sigma_outer"], locals=sym)) == 0
    assert sp.simplify(sigma_inner - sp.sympify(gt["sigma_inner"], locals=sym)) == 0
    assert sp.simplify(Q_total) == 0
    print("PASSED (rho_b, both surface charges, total bound charge = 0)")


def verify_trap_03():
    print("\n--- Trap 3: irreversible adiabatic expansion ---")
    gt = load("trap_03_irreversible_expansion.json")
    n, T1, P1, P_ext = 1, 300, 400_000, 100_000
    R_gas = sp.Rational(8314, 1000)
    Cv, Cp = sp.Rational(3, 2) * R_gas, sp.Rational(5, 2) * R_gas
    gamma = sp.Rational(5, 3)
    T2 = sp.symbols("T2", positive=True)
    V1 = n * R_gas * T1 / P1
    V2 = n * R_gas * T2 / P_ext
    # First law, Q = 0:  n Cv (T2 - T1) = -P_ext (V2 - V1)
    T2_val = sp.solve(sp.Eq(n * Cv * (T2 - T1), -P_ext * (V2 - V1)), T2)[0]
    T2_rev = T1 * (sp.Rational(P_ext, P1)) ** (R_gas / Cp)
    dS = n * Cp * sp.log(T2_val / T1) - n * R_gas * sp.log(sp.Rational(P_ext, P1))
    print(f"T2 (irreversible) = {float(T2_val):.2f} K, T2 (wrong, reversible) = {float(T2_rev):.2f} K")
    print(f"Entropy generated = {float(dS):.3f} J/K")
    assert abs(float(T2_val) - gt["T2_K"]) < 1e-9
    assert abs(float(T2_rev) - gt["T2_wrong_reversible_K"]) < 0.01
    assert dS > 0                                   # irreversible: S_gen > 0
    V2_val = V2.subs(T2, T2_val)
    assert abs(float(P_ext * V2_val ** gamma / (P1 * V1 ** gamma)) - 1) > 0.05
    print("PASSED (T2 = 210 K, S_gen > 0, P V^gamma is NOT conserved)")


if __name__ == "__main__":
    verify_trap_01()
    verify_trap_02()
    verify_trap_03()

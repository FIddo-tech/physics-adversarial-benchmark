# Physics Adversarial Benchmark

Three physics problems where language models commonly go wrong, each with an exact symbolic ground truth checked in SymPy. Every check ends in a real assertion against the value stored in the problem file.

## The traps

| Problem | Domain | Verified result |
|---|---|---|
| Particle sliding off a sphere | Mechanics | cos(theta_c) = 2/3 |
| Dielectric shell with P = P0 (r/b) r_hat | Electromagnetism | rho_b = -3 P0 / b, sigma_outer = P0, sigma_inner = -P0 a / b, total bound charge = 0 |
| Gas expanding irreversibly against constant pressure | Thermodynamics | T2 = 210 K (a reversible-adiabatic shortcut gives about 172 K), entropy generated > 0 |

Each problem file in problems/ also records the common LLM failure for that trap.

## Run it

    pip install -r requirements.txt
    python verify_traps.py

import numpy as np
import matplotlib.pyplot as plt

# ==========================
# User parameters
# ==========================
FWHM = 500
sigma = FWHM / 2.355

R1 = 50
R2 = 200

grid_size = 5000
extent = 5 * sigma

# ==========================
# Construct grid
# ==========================
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y)

# ==========================
# 2D Gaussian (normalized)
# ==========================
norm = 1 / (2 * np.pi * sigma**2)
Z = norm * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

radius_map = np.sqrt(X**2 + Y**2)

# ==========================
# Masks
# ==========================
mask_R1 = radius_map <= R1
mask_R2 = radius_map <= R2

# ==========================
# Numerical integration
# ==========================
p1 = np.sum(Z[mask_R1]) * dx * dy
p2 = np.sum(Z[mask_R2]) * dx * dy

# ==========================
# Analytic solutions
# ==========================
p1_analytic = 1 - np.exp(-(R1**2) / (2 * sigma**2))
p2_analytic = 1 - np.exp(-(R2**2) / (2 * sigma**2))

# ==========================
# Density ratio
# ==========================
ratio_numeric = (p1 / R1**2) / (p2 / R2**2)
ratio_analytic = (p1_analytic / R1**2) / (p2_analytic / R2**2)

# ==========================
# Output
# ==========================
print(f"Sigma = {sigma:.3f}")
print(f"p1 (R1={R1}) numeric  = {p1:.6f}")
print(f"p2 (R2={R2}) numeric  = {p2:.6f}")
print()
print(f"p1 analytic = {p1_analytic:.6f}")
print(f"p2 analytic = {p2_analytic:.6f}")
print()
print(f"Density ratio (numeric)  = {ratio_numeric:.6f}")
print(f"Density ratio (analytic) = {ratio_analytic:.6f}")
print(f"Absolute difference      = {abs(ratio_numeric - ratio_analytic):.6e}")

# ==========================
# Plot
# ==========================
plt.figure(figsize=(6, 6))
plt.imshow(Z, extent=[-extent, extent, -extent, extent], origin='lower')
plt.gca().add_patch(plt.Circle((0, 0), R1, color='red', fill=False, linewidth=2))
plt.gca().add_patch(plt.Circle((0, 0), R2, color='white', fill=False, linewidth=2))

plt.colorbar(label='Probability Density')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Gaussian with Two Circular Apertures')
plt.gca().set_aspect('equal')
plt.show()


from scipy.optimize import brentq

# ==========================
# Fixed radii
# ==========================
R1 = 50
R2 = 200

# ==========================
# User input: desired ratio
# ==========================
desired_ratio = 1.2   # <-- change this

# ==========================
# Ratio function
# ==========================
def density_ratio(sigma):
    p1 = 1 - np.exp(-(R1**2) / (2 * sigma**2))
    p2 = 1 - np.exp(-(R2**2) / (2 * sigma**2))
    return (p1 / R1**2) / (p2 / R2**2)

# Function whose root we want
def objective(sigma):
    return density_ratio(sigma) - desired_ratio

# ==========================
# Solve for sigma
# ==========================
# Reasonable bracket for sigma
sigma_min = 1.0
sigma_max = 1000.0

sigma_solution = brentq(objective, sigma_min, sigma_max)

FWHM_solution = 2.355 * sigma_solution

# ==========================
# Output
# ==========================
print(f"Desired ratio = {desired_ratio}")
print(f"Solved sigma  = {sigma_solution:.6f}")
print(f"Solved FWHM   = {FWHM_solution:.6f}")

# Optional: verify
check_ratio = density_ratio(sigma_solution)
print(f"Recovered ratio = {check_ratio:.6f}")



import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Fixed radii
# ==========================
R1 = 50
R2 = 200

# ==========================
# FWHM range
# ==========================
FWHM_values = np.linspace(10, 2000, 500)
ratios = []

for FWHM in FWHM_values:
    sigma = FWHM / 2.355
    
    p1 = 1 - np.exp(-(R1**2) / (2 * sigma**2))
    p2 = 1 - np.exp(-(R2**2) / (2 * sigma**2))
    
    ratio = (p1 / R1**2) / (p2 / R2**2)
    ratios.append(ratio)

ratios = np.array(ratios)

# ==========================
# Plot
# ==========================
plt.figure(figsize=(7, 5))
plt.plot(FWHM_values, np.log10(ratios))

plt.xlabel("FWHM")
plt.ylabel("log10(Density Ratio)")
plt.title("log10[(p1/R1²)/(p2/R2²)] vs FWHM")
plt.grid(True)
plt.show()

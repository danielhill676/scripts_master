import math

# GATOS DATA

names_GATOS = ['NGC 6300', 'NGC 5643', 'NGC 7314', 'NGC 4388', 'NGC 4941', 'NGC 7213',
         'NGC 7582', 'NGC 6814', 'NGC 5506', 'NGC 7465', 'NGC 1068', 'NGC 1365', 'NGC 3227']

names_GATOS_ordered = [
    'NGC 1068', 'NGC 1365', 'NGC 3227', 'NGC 4388', 'NGC 4941',
    'NGC 5506', 'NGC 5643', 'NGC 6300', 'NGC 6814', 'NGC 7213',
    'NGC 7314', 'NGC 7465', 'NGC 7582'
]

log10_sigma_torus = [4.03, 2.60, 3.08, 2.44, 2.92, 3.28, 4.15,
                     2.91, 2.26, 1.75, 2.72, 3.42, 3.18, 1.80,
                     1.61, 2.17, 2.58, 2.62]

log10_sigma_50pc = [3.36, 2.72, 2.54, 2.25, 2.62, 3.19, 2.71,
                    2.91, 2.00, 2.24, 2.45, 3.20, 3.13, 1.70,
                    1.21, 2.25, 2.68, 2.80]

log10_sigma_200pc = [2.54, 2.97, 1.81, 1.55, 1.92, 2.36, 2.23,
                     2.77, 2.16, 1.45, 2.37, 2.65, 2.45, 0.99,
                     0.66, 1.56, 2.10, 2.96]

ra = ['17:16:59.473', '14:32:40.778', '22:35:46.230', '12:25:46.820', '13:04:13.143', 
      '22:09:16.260', '23:18:23.621', '19:42:40.576', '14:13:14.901', '23:02:00.952',
      '02:42:40.771', '03:33:36.458', '10:23:30.570']

dec = ['−62:49:13.98', '−44:10:28.60', '−26:03:00.90', '+12:39:43.45', '−05:33:05.83',
       '−47:09:59.95', '−42:22:14.06', '−10:19:25.50', '−03:12:27.22', '+15:57:53.55',
       '−00:00:47.84', '−36:08:26.37', '+19:51:54.30']

distance_mpc = [14.0, 16.9, 17.4, 18.1, 20.5, 22.0, 22.5, 22.8, 26.4, 27.2, 14.0, 18.3, 23.0]

hubble_type = ['SB(rs)b', 'SAB(rs)c', 'SAB(rs)bc', 'SA(s)b', '(R)SAB(r)ab', 'SA(s)a',
               "(R’)SB(s)ab", 'SAB(rs)bc', 'Sa peculiar', "(R’)SB(s)0", '(R)SA(rs)b', 
               "(R’)SBb(s)b", 'SAB(s)a pec']

agn_type = ['Sy2', 'Sy2', 'Sy1.9, S1h', 'Sy1.9, S1h', 'Sy2', 'Sy1.5, radio-source',
            'Sy2 , S1i', 'Sy1.5', 'Sy1.9, S1i', 'Sy2, S3', 'Sy2', 'S1.8', 'S1.5']

pa_phot = [120, 98, 3, 91, 22, 124, 156, 108, 89, 162, 73, 23, 156]

i_phot = [53, 30, 70, 90, 37, 39, 68, 52, 90, 64, 35, 63, 68]

log10_L_14_150 = [42.3, 43.0, 42.2, 43.0, 42.0, 42.3, 43.2, 42.6, 43.2, 42.0, 42.7, 42.3, 42.8]

log10_L_2_10 = [41.7, 42.4, 42.2, 42.5, 41.4, 41.9, 43.5, 42.2, 43.0, 41.9, 42.8, 42.1, 42.4]

log10_L_AGN_LEdd = [-1.9, -1.3, -1.2, -1.1, -2.4, -3.0, -1.7, -1.6, -2.3, -2.2, -0.3, -2.8, -1.2]

log10_N_H_Xabs = [23.3, 25.4, 21.6, 23.5, 23.7, 20.0, 24.3, 21.0, 22.4, 21.5, 25.0, 22.2, 21.0]

log10_L_2_10_ordered = [
    42.8, 42.1, 42.4, 42.5, 41.4,
    43.0, 42.4, 41.7, 42.2, 41.9,
    42.2, 41.9, 43.5
]

log10_L_AGN_LEdd_ordered = [
    -0.3, -2.8, -1.2, -1.1, -2.4,
    -2.3, -1.3, -1.9, -1.6, -3.0,
    -1.2, -2.2, -1.7
]


# NUGA DATA

names_NUGA = ['N613', 'N1326', 'N1365', 'N1433', 'N1566', 'N1672', 'N1808']

LAGN_NUGA =[42.1,40.7,42.8,40.0,41.4,39.3,40.6,44.7]
MBH_NUGA = [7.57,7.40,7.84,7.40,7.13,7.40,7.79,8.23]
MBH_NUGA_alt = [7.57,7.40,7.84,7.40,7.13,7.40,100,8.23]
log_LX = [41.2, 39.9, 41.8, 39.2, 40.5, 38.4, 39.8]

LAGN_NUGA_filtered = [42.1, 40.7, 41.4, 39.3, 40.6]
names_NUGA_filtered = ['NGC 613', 'NGC 1326', 'NGC 1566', 'NGC 1672', 'NGC 1808']
MBH_NUGA_filtered   = [7.57, 7.40, 7.13, 7.40, 7.79]
MBH_NUGA_filtered_alt   = [7.57, 6.81, 6.83, 7.70, 7.79]
MBH_NUGA_filtered_alt2   = [7.6, 7.11, 6.48, 6, 7.2]
log_LX_filtered     = [41.2, 39.9, 40.5, 38.4, 39.8]

log_LX_bullshit = [41.3, 39.8,40.9,39.2,39.9]
EDDR_NUGA_bullshit = [-3.5,-4,-3.5,-6.3,-4.5]

LEDD_NUGA = []

for i in range(len(MBH_NUGA_filtered)):
    LEDD = math.log10( 1.26e38 * 10**MBH_NUGA_filtered_alt2[i])
    LEDD_NUGA.append(LEDD)
print(LEDD_NUGA)

EDDR_NUGA = []

for i in range(len(LEDD_NUGA)):
    EDDR = LAGN_NUGA_filtered[i] - LEDD_NUGA[i]
    EDDR_NUGA.append(EDDR)
print(EDDR_NUGA)

# combinbed

names = ['NGC 613', 'NGC 1068', 'NGC 1326', 'NGC 1365', 'NGC 1566', 'NGC 1672', 'NGC 1808',
         'NGC 3227', 'NGC 4388', 'NGC 4941', 'NGC 5506', 'NGC 5643', 'NGC 6300', 'NGC 6814',
         'NGC 7213', 'NGC 7314', 'NGC 7465', 'NGC 7582']

L_X_master =[]
EDD_R_master = []

g_count = 0
n_count = 0
for i in range(len(names)):
    print(names[i])
    if names[i] in names_GATOS_ordered:
        print('GATOS')
        L_X_master.append(log10_L_2_10_ordered[g_count])
        EDD_R_master.append(log10_L_AGN_LEdd_ordered[g_count])
        g_count += 1
    else:
        print('NUGA')
        print(n_count)
        print(log10_sigma_50pc[i] - log10_sigma_200pc[i])
        L_X_master.append(log_LX_bullshit[n_count])
        EDD_R_master.append(EDDR_NUGA_bullshit[n_count])
        n_count += 1

print(L_X_master)
print(EDD_R_master)


# Inactive pairs
names_inactive = ['NGC 2775','NGC 3717','NGC 4254','NGC 3175']
LBOL = [40.6,40.7,40.5,40]
L_X_inactive = []

for i in range(len(LBOL)):
    L_X = math.log10((12.76*(1+(LBOL[i]-math.log10(3.82e33))/12.15)**18.78)*3.82e33)
    L_X_inactive.append(L_X)
fCO32_lim = [0.40*200*1e-3,0.55*200*1e-3,0.55*200*1e-3,0.55*200*1e-3]
D = [21,24,15,14]
LPCO32 = [3.25e7*f*(d**2)*(1/324.8**2) for f,d in zip(fCO32_lim,D)]
ratio = 0.7  # change to 0.7 or 2.9 ?
LPCO10 = [l/ratio for l in LPCO32]
MH2 = [1.1*l for l in LPCO10]
beam = [math.pi*((0.13/206265)*d*1e6/2)**2 for d in D]
inactive_torus = [math.log10(m/b) for m,b in zip(MH2,beam)]
print(inactive_torus)

M_star = [10.4,10.3,10.2,]

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



torus_200_ratio = [a-b for a,b in zip(log10_sigma_torus, log10_sigma_200pc)]

fig, ax = plt.subplots(figsize=(8, 9))

# Scatter plot with individual marker logic
for i, name in enumerate(names):
    marker = 'o' if name in names_GATOS else 's'  # circle or square
    sc = ax.scatter(L_X_master[i], torus_200_ratio[i],
                    c=EDD_R_master[i],
                    cmap='plasma',
                    vmin=min(EDD_R_master),
                    vmax=max(EDD_R_master),
                    marker=marker,
                    edgecolors='black',
                    s=250)
    ax.text(L_X_master[i], torus_200_ratio[i] + 0.05, name,fontsize=10,
            ha='center', va='bottom')
    
for i, name in enumerate(names_inactive):
    sc_I = ax.scatter(L_X_inactive[i], inactive_torus[i],
                    marker='.',
                    color = 'black',
                    edgecolors='black',
                    s=20)

    # if name == 'NGC 4254':
    #     ax.text(L_X_inactive[i]+0.12, inactive_torus[i] + 0.10, name, ha='center',fontsize=10)  # Shift upward
    # elif name == 'NGC 3717':
    #     ax.text(L_X_inactive[i]+0.24, inactive_torus[i] + 0.04, name, ha='center',fontsize=10)  # Shift upward
    # elif name == 'NGC 3175':
    #     ax.text(L_X_inactive[i]-0.1, inactive_torus[i] +0.04, name, ha='center',fontsize=10)  # Shift upward
    # else:
    #     ax.text(L_X_inactive[i], inactive_torus[i] + 0.05, name, ha='center',fontsize=10)
    
# Labels
ax.set_xlabel(r'log $L^X_{2-10 keV}$', fontsize=18)
ax.set_ylabel(r'log $(\Sigma^\text{torus}_{\text{gas}} [M_\odot \text{pc}^{-2}]$', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
# Square axes (equal aspect ratio)
ax.set_aspect('auto', adjustable='box')

# Remove gridlines
ax.grid(False)

# Horizontal colorbar above the plot with thinner size
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
cbar.set_label('log Edd. ratio', fontsize=12)

# Ensure the plot is square by adjusting the aspect ratio of the figure
ax.set_xlim(min(L_X_master) - 1.5, max(L_X_master) + 0.3)  # Adjust x-limits for better view
ax.set_ylim(min(torus_200_ratio) - 0.5, max(torus_200_ratio) + 0.2)  # Adjust y-limits for better view

ax.yaxis.set_major_locator(MultipleLocator(0.25))

# Add a blue cross in the bottom left corner
cross_x = 40
cross_y = -0.25
cross_x_len = 0.3
cross_y_len = 0.4

# Horizontal line (cross)
ax.plot([cross_x - cross_x_len/2, cross_x + cross_x_len/2], [cross_y, cross_y], color='blue', lw=1)

# Vertical line (cross)
ax.plot([cross_x, cross_x], [cross_y - cross_y_len/2, cross_y + cross_y_len/2], color='blue', lw=1)

# Tight layout to prevent clipping
plt.tight_layout()
plt.show()
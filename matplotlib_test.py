#test for plotting
import numpy as np
import matplotlib.pyplot as plt


fig_m8_a = plt.figure(figsize=(9, 13))
columns = 4
rows = 5
ax_m8_a = []

w = 8
h = 8

for i in range(1,columns*rows +1):
    img = np.random.randint(10, size=(h,w))
    ax_m8_a.append( fig_m8_a.add_subplot(rows, columns, i))
    plt.imshow(img)
    cbar = plt.colorbar()
    ax_m8_a[-1].set_title(f"Plot no:{i}")
plt.tight_layout()
plt.show()
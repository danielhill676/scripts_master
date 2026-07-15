from scipy.stats import ttest_1samp, norm

# tstat, p = ttest_1samp(data, popmean=0)

p = 0.02085

sigma = norm.isf(p/2)

# print(f"t = {tstat:.2f}")
print(f"p = {p:.4g}")
print(f"Gaussian-equivalent significance = {sigma:.2f}σ")
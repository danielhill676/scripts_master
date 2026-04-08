import pandas as pd

# Load data
df_A = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples/GB24_df_new.csv")
df_B = pd.read_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples/GB24_df_new_pt2.csv")

# Merge (many-to-one)
df_merged = df_A.merge(df_B, on="Name", how="left")

# Save if needed
df_merged.to_csv("/Users/administrator/Astro/LLAMA/ALMA/comp_samples/GB24_df_new_final.csv", index=False)
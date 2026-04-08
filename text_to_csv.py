import pandas as pd
import re

raw_text = """Name α(a)  2000 δ(a)  2000 D(b) log10 L2−10keV (c)  X λEdd(d) Hubble & AGN type(e) Hubble stage T (e) PA(f) i(f)  —- h m s ◦ ′ ′′ Mpc erg s−1 —- —- —- o o  AGN sample: CO(3-2)  NGC1566 04:20:00.395 -54:56:16.60 7.2 40.50 -2.89 (R’1)SAB(rs)bc; Sy1.5 4.0 44 49 NGC1808 05:07:42.329 -37:30:45.85 9.3 39.80 -4.51 (R’1)SAB(s:)b; Sy2 1.2 146 83 NGC1672 04:45:42.496 -59:14:49.92 11.4 39.10 -6.41 (R’1)SB(r)bc; Sy2 3.3 155 29 NGC1068 02:42:40.709 -00:00:47.94 14.0 42.82 -0.70 (R)SA(rs)b; Sy2 3.0 289 41 NGC6300 17:16:59.543. -62:49:14.04 14.0 41.73 -1.72 SB(rs)b Sy2 3.1 95 57 NGC1326 03:23:56.416 -36:27:52.68 14.9 39.90 -4.02 (R1)SB(rl)0/a; LINER -0.7 71 53 NGC5643 14:32:40.699 -44:10:27.93 16.9 42.41 -1.23 SAB(rs)c; Sy2 5.0 301 30 NGC613 01:34:18.189 -29:25:06.59 17.2 41.20 -3.48 SB(rs)bc; HII Sy 4.0 122 36 NGC7314 22:35:46.201 -26:03:01.58 17.4 42.18 -2.07 SAB(rs)bc; Sy1.9 4.0 191 55 NGC4388 12:25:46.781 +12:39:43.75 18.1 42.45 -1.07 SA(s)b: sp; Sy2 Sy1.9 2.8 82 79 NGC1365 03:33:36.369 -36:08:25.50 18.3 42.09 -2.15 (R’)SBb(s)b; HII Sy1.8 3.2 40 41 NGC4941 13:04:13.103 -05:33:05.73 20.5 41.40 -2.10 (R)SAB(r)ab:; Sy2 2.1 212 41 NGC7213 22:09:16.209 -47:10:00.12 22.0 41.85 -3.01 SA(s)a:; LINER Sy1.5 0.9 133 35 NGC7582 23:18:23.643 -42:22:13.54 22.5 43.49 -1.70 (R’1)SB(s)ab; Sy2 2.1 344 59 NGC6814 19:42:40.587 -10:19:25.10 22.8 42.24 -1.62 SAB(rs)bc; HII Sy1.5 4.0 84 57 NGC3227 10:23:30.577 +19:51:54.28 23.0 42.37 -1.20 SAB(s) pec; Sy1.5 1.5 152 52 NGC5506 14:13:14.878 -03:12:27.66 26.4 42.98 -1.22 Sa pec sp ; Sy1.9 1.2 275 80 NGC7465 23:02:00.961 +15:57:53.21 27.2 41.93 -2.10 (R’)SB(s)0; Sy2 -1.8 66 54.5 NGC7172 22:02:01.891 -31:52:10.48 37.0 42.84 -1.60 Sa pec sp; Sy2 HII 0.6 92 85 NGC5728 14:42:23.872 -17:15:11.01 44.5 43.19 -1.72 (R1)SAB(r)a; HII Sy2 1.2 15 59  AGN sample: CO(2-1)  NGC4826 12:56:43.643 +21:40:59.30 4.4 37.78 -6.66 (R)SA(rs)ab; HII Sy2 2.2 112 60 NGC5236 13:37:00.94 -29:51:56.16 4.9 38.70 —- SAB(s)c; HII Sbrst 5.0 225 24 NGC2903 09:32:10.10 +21:30:02.88 10.0 38.00 -6.62 SAB(rs)bc; HII 4.0 204 67 NGC3351 10:43:57.731 +11:42:13.35 10.0 38.03 -6.75 SB(r)b; HII Sbrst 3.1 193 45 NGC1637 04:41:28.10 -02:51:28.80 11.7 38.04 —- SAB(rs)c; AGN 5.0 21 31 NGC4501 12:31:59.220 14:25:12.69 14.0 39.68 -5.78 SAb -SA(rs)b; HII Sy2 3.3 135 59 NGC4438 12:27:45.675 13:00:31.18 16.5 38.72 -6.35 SA(s)0/a pec:; LINER 2.8 30 60 NGC4569 12:36:49.80 +13:09:46.30 16.8 39.60 -5.50 SAB(rs)ab; LINER Sy 2.4 23 70 NGC4579 12:37:43.58 +11:49:02.49 16.8 41.42 -3.82 SAB(rs)b; LINER Sy1.9 2.8 95 36 NGC3718 11:32:34.880 +53:04:04.32 17.0 40.64 -4.64 SB(s)a pec; Sy1 LINER 1.1 . 120 60 NGC3368 10: 46:45.50 11:49:12.00 18.0 39.30 -5.53 SABab SAB(rs)ab;Sy LINER 2.1 165 60 NGC2110 05:52:11.377 -07:27:22.48 34.8 42.67 -1.87 SAB0-; Sy2 -3.0 175 46 NGC2782 09:14:05.111 40:06:49.24. 35.0 39.50 -6.10 SAB(rs)a; Sy1 Sbrst 1.1 75 20 NGC7172 22:02:01.891 -31:52:10.48 37.0 42.84 -1.60 Sa pec sp; Sy2 HII 0.6 92 85 MCG-06-30-15 13:35:53.770 -34:17:44.16 38.3 42.86 -1.28 S?; Sy1.2 2.0 116 59 NGC2992 09:45:41.943 -14:19:34.57 39.2 42.20 -2.29 Sa pec; Sy1.9 0.9 29 80 NGC3081 09:59:29.546 -22:49:34.78 40.3 43.10 -1.47 ( R1)SAB(r)0/a; Sy2 0.0 71 60 ESO137-34 16:35:13.996 -58:04:47.77 41.5 42.80 -1.99 SAB(s)0/a? Sy2 0.6 18 41 NGC5728 14:42:23.872 -17:15:11.01 44.5 43.19 -1.72 (R1)SAB(r)a; HII Sy2 1.2 15 59 ESO21-g004 13:32:40.621 -77:50:40.40 45.1 42.32 —- SA(s)0/a: 0.20 100 65  AGN sample: CO(1-0)  M51 13:29:52.68 +47:11:42.72 8.6 39.00 -5.25 SA(s)bc pec; HII Sy2.5 4.0 173 21 NGC6300 17:16:59.543. -62:49:14.04 14.0 41.73 -1.72 SB(rs)b Sy2 3.1 95 57 NGC4321 12:22:54.954 +15:49:20.49 16.8 40.40 -3.80 SAB(s)bc; LINER HII 4.0 153 32 NGC5643 14:32:40.699 -44:10:27.93 16.9 42.41 -1.23 SAB(rs)c; Sy2 5.0 301 30 NGC7314 22:35:46.201 -26:03:01.58 17.4 42.18 -2.07 SAB(rs)bc; Sy1.9 4.0 191 55 NGC4388 12:25:46.781 +12:39:43.75 18.1 42.45 -1.07 SA(s)b: sp; Sy2 Sy1.9 2.8 82 79 NGC6221 16:52:46.346 -59:13:01.08 22.9 41.26 -2.90 Sb: pec ; HII LIRG 3.1 1 51 NGC3227 10:23:30.577 +19:51:54.28 23.0 42.37 -1.20 SAB(s) pec; Sy1.5 1.5 152 52 """
tokens = re.split(r"\s+", raw_text)

def is_name(tok):
    return bool(re.match(r"^(NGC|IC|ESO|MCG|M)[A-Za-z0-9\-\+]*$", tok))

def is_number(tok):
    tok = tok.replace("<", "").replace("−", "-")
    try:
        float(tok)
        return True
    except:
        return False

rows = []
i = 0

while i < len(tokens):
    if is_name(tokens[i]):
        name = tokens[i]
        j = i + 1
        values = []

        # Collect until next galaxy name
        while j < len(tokens) and not is_name(tokens[j]):
            tok = tokens[j].replace("<", "").replace("−", "-")
            if is_number(tok):
                values.append(float(tok))
            j += 1

        # # Map values safely
        # row = {
        #     "Name": name,
        #     "D_Mpc": values[0] if len(values) > 0 else None,
        #     "log10SigmaGas_50": values[1] if len(values) > 1 else None,
        #     "log10SigmaGas_200": values[2] if len(values) > 2 else None,
        #     "CCI": values[3] if len(values) > 3 else None,
        #     "log10SigmaHot_50": values[4] if len(values) > 4 else None,
        #     "log10SigmaHot_200": values[5] if len(values) > 5 else None,
        #     "HCI": values[6] if len(values) > 6 else None,
        # }

        # Map values safely
        row = {
            "Name": name,
            "D": values[0] if len(values) > 0 else None,
            "log LX": values[1] if len(values) > 1 else None,
            "lambda_edd": values[2] if len(values) > 2 else None,
            "Hubble Stage": values[3] if len(values) > 3 else None,
            "PA": values[4] if len(values) > 4 else None,
            "I": values[5] if len(values) > 5 else None
        }

        rows.append(row)
        i = j
    else:
        i += 1

df = pd.DataFrame(rows)

# Save
csv_path = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples/GB24_df_new_pt2.csv"


df.to_csv(csv_path, index=False)

print("Rows:", len(df))

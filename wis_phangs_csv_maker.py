import pandas as pd
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.exceptions import RemoteServiceError
import requests
import time
from astropy.table import MaskedColumn
import numpy as np
import os
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad

def normalize_name(col):
    s = pd.Series(col.astype(str))
    return (
        s.str.replace('–', '-', regex=False)
        .str.replace('−', '-', regex=False)
        .str.strip()
        .str.upper()
    )

################## WISDOM X DATA ##################

wisdom = pd.DataFrame([
    ["FRL49", 0.30, 0.04, 0.17, 0.01, 0.41, 0.12],
    ["MRK567", 0.76, 0.05, 0.23, 0.01, 0.50, 0.05],
    ["NGC0383", 0.11, 0.01, 0.13, 0.02, 0.16, 0.03],
    ["NGC0449", 0.60, 0.02, 0.40, 0.01, 0.64, 0.01],
    ["NGC0524", 0.23, 0.01, 0.19, 0.01, 0.43, 0.02],
    ["NGC0612", 0.63, 0.03, 0.52, 0.01, 0.47, 0.02],
    ["NGC0708", 0.69, 0.01, 0.32, 0.03, 0.51, 0.06],
    ["NGC1387", 0.15, 0.02, 0.10, 0.01, 0.27, 0.02],
    ["NGC1574", 0.02, 0.01, 0.04, 0.01, 0.31, 0.05],
    ["NGC3169", 0.99, 0.03, 0.33, 0.01, 0.81, 0.03],
    ["NGC3368", 0.49, 0.04, 0.19, 0.01, 0.79, 0.04],
    ["NGC3607", 0.32, 0.01, 0.24, 0.02, 0.57, 0.01],
    ["NGC4061", 0.14, 0.01, 0.15, 0.02, 0.32, 0.06],
    ["NGC4429", 0.15, 0.01, 0.09, 0.02, 0.29, 0.06],
    ["NGC4435", 0.18, 0.01, 0.19, 0.03, 0.37, 0.08],
    ["NGC4438", 0.30, 0.04, 0.17, 0.01, 0.60, 0.04],
    ["NGC4501", 0.75, 0.02, 0.38, 0.01, 0.79, 0.03],
    ["NGC4697", 0.13, 0.01, 0.20, 0.02, 0.51, 0.04],
    ["NGC4826", 0.25, 0.02, 0.04, 0.01, 0.37, 0.03],
    ["NGC5064", 0.26, 0.03, 0.14, 0.02, 0.26, 0.07],
    ["NGC5765b", 0.67, 0.03, 0.43, 0.01, 0.64, 0.01],
    ["NGC5806", 0.36, 0.03, 0.32, 0.01, 0.53, 0.09],
    ["NGC6753", 0.46, 0.02, 0.26, 0.01, 0.44, 0.08],
    ["NGC6958", 0.11, 0.01, 0.10, 0.01, 0.42, 0.09],
    ["NGC7052", 0.22, 0.02, 0.14, 0.03, 0.34, 0.09],
    ["NGC7172", 0.21, 0.01, 0.18, 0.02, 0.64, 0.03]
], columns=["Name", "Asymmetry", "Asymmetry_err", "Smoothness", "Smoothness_err", "Gini", "Gini_err"])

simulations = pd.DataFrame([
    ["noB", 1.57, 0.08, 0.52, 0.027, 0.81, 0.03],
    ["B_M30_R1", 0.69, 0.05, 0.27, 0.020, 0.39, 0.02],
    ["B_M30_R2", 1.24, 0.08, 0.44, 0.037, 0.67, 0.03],
    ["B_M30_R3", 1.47, 0.11, 0.56, 0.026, 0.77, 0.02],
    ["B_M60_R1", 0.40, 0.04, 0.20, 0.013, 0.24, 0.02],
    ["B_M60_R3", 1.10, 0.06, 0.39, 0.015, 0.60, 0.02],
    ["B_M60_R2", 0.75, 0.07, 0.30, 0.027, 0.42, 0.04],
    ["B_M90_R1", 0.35, 0.04, 0.20, 0.012, 0.23, 0.02],
    ["B_M90_R2", 0.46, 0.05, 0.21, 0.016, 0.25, 0.02],
    ["B_M90_R3", 0.69, 0.05, 0.27, 0.015, 0.36, 0.02]
], columns=["Name", "Asymmetry", "Asymmetry_err", "Smoothness", "Smoothness_err", "Gini", "Gini_err"])

phangs = pd.DataFrame([
    ["IC1954", 0.97, 0.01, 0.30, 0.01, 0.57, 0.01],
    ["IC5273", 0.94, 0.02, 0.37, 0.01, 0.75, 0.01],
    ["NGC0628", 0.70, 0.02, 0.32, 0.01, 0.65, 0.01],
    ["NGC0685", 1.38, 0.01, 0.38, 0.01, 0.83, 0.01],
    ["NGC1087", 0.83, 0.03, 0.27, 0.01, 0.67, 0.01],
    ["NGC1097", 0.43, 0.01, 0.24, 0.01, 0.47, 0.05],
    ["NGC1300", 0.61, 0.02, 0.19, 0.01, 0.61, 0.10],
    ["NGC1317", 0.56, 0.01, 0.22, 0.01, 0.39, 0.01],
    ["NGC1365", 0.71, 0.02, 0.17, 0.01, 0.52, 0.04],
    ["NGC1385", 1.43, 0.01, 0.32, 0.01, 0.65, 0.01],
    ["NGC1433", 0.63, 0.01, 0.23, 0.01, 0.47, 0.06],
    ["NGC1511", 1.45, 0.02, 0.32, 0.01, 0.70, 0.01],
    ["NGC1512", 0.66, 0.02, 0.36, 0.02, 0.49, 0.05],
    ["NGC1546", 0.32, 0.01, 0.13, 0.01, 0.27, 0.03],
    ["NGC1559", 1.31, 0.01, 0.38, 0.01, 0.70, 0.01],
    ["NGC1566", 0.71, 0.05, 0.32, 0.01, 0.74, 0.04],
    ["NGC1637", 0.58, 0.03, 0.33, 0.01, 0.71, 0.01],
    ["NGC1672", 0.53, 0.02, 0.17, 0.01, 0.46, 0.06],
    ["NGC1792", 0.66, 0.04, 0.18, 0.01, 0.42, 0.02],
    ["NGC2090", 0.69, 0.01, 0.29, 0.01, 0.44, 0.01],
    ["NGC2566", 0.41, 0.01, 0.16, 0.01, 0.75, 0.06],
    ["NGC2903", 0.74, 0.01, 0.24, 0.01, 0.73, 0.04],
    ["NGC2997", 0.62, 0.02, 0.20, 0.01, 0.64, 0.04],
    ["NGC3059", 1.22, 0.01, 0.29, 0.01, 0.74, 0.01],
    ["NGC3137", 1.26, 0.01, 0.40, 0.01, 0.64, 0.02],
    ["NGC3351", 0.27, 0.03, 0.33, 0.01, 0.54, 0.18],
    ["NGC3511", 0.52, 0.03, 0.24, 0.01, 0.48, 0.02],
    ["NGC3507", 0.83, 0.05, 0.42, 0.02, 0.75, 0.03],
    ["NGC3521", 0.41, 0.03, 0.18, 0.01, 0.29, 0.01],
    ["NGC3596", 0.87, 0.03, 0.35, 0.01, 0.68, 0.01],
    ["NGC3621", 0.92, 0.03, 0.27, 0.01, 0.47, 0.01],
    ["NGC3626", 1.27, 0.01, 0.52, 0.01, 0.73, 0.01],
    ["NGC3627", 0.69, 0.04, 0.38, 0.01, 0.80, 0.01],
    ["NGC4207", 0.62, 0.01, 0.37, 0.01, 0.74, 0.02],
    ["NGC4254", 0.70, 0.01, 0.24, 0.01, 0.45, 0.01],
    ["NGC4293", 0.38, 0.03, 0.44, 0.01, 0.83, 0.03],
    ["NGC4298", 0.63, 0.01, 0.24, 0.01, 0.44, 0.01],
    ["NGC4303", 0.47, 0.01, 0.18, 0.01, 0.61, 0.07],
    ["NGC4321", 0.56, 0.01, 0.29, 0.01, 0.59, 0.08],
    ["NGC4424", 1.16, 0.03, 0.37, 0.02, 0.80, 0.03],
    ["NGC4457", 0.95, 0.01, 0.31, 0.01, 0.75, 0.01],
    ["NGC4496A", 1.61, 0.01, 0.51, 0.01, 0.85, 0.03],
    ["NGC4535", 0.41, 0.05, 0.21, 0.01, 0.82, 0.08],
    ["NGC4536", 0.34, 0.03, 0.18, 0.01, 0.68, 0.06],
    ["NGC4540", 1.31, 0.01, 0.47, 0.01, 0.74, 0.01],
    ["NGC4548", 0.46, 0.05, 0.35, 0.01, 0.92, 0.07],
    ["NGC4569", 0.71, 0.01, 0.22, 0.01, 0.73, 0.04],
    ["NGC4579", 0.93, 0.02, 0.35, 0.01, 0.71, 0.01],
    ["NGC4654", 0.37, 0.02, 0.13, 0.01, 0.45, 0.03],
    ["NGC4689", 0.85, 0.01, 0.27, 0.01, 0.45, 0.01],
    ["NGC4694", 1.50, 0.03, 0.35, 0.01, 0.91, 0.02],
    ["NGC4731", 1.56, 0.02, 0.52, 0.01, 0.91, 0.01],
    ["NGC4781", 0.97, 0.01, 0.27, 0.01, 0.57, 0.03],
    ["NGC4941", 0.83, 0.03, 0.35, 0.02, 0.92, 0.01],
    ["NGC5134", 1.62, 0.01, 0.51, 0.01, 0.85, 0.01],
    ["NGC5248", 0.39, 0.02, 0.17, 0.01, 0.52, 0.06],
    ["NGC5530", 1.00, 0.01, 0.37, 0.01, 0.60, 0.01],
    ["NGC5643", 0.89, 0.01, 0.28, 0.01, 0.82, 0.02],
    ["NGC6300", 0.78, 0.01, 0.36, 0.01, 0.87, 0.01],
    ["NGC7496", 0.53, 0.01, 0.28, 0.01, 0.77, 0.06]
], columns=["Name", "Asymmetry", "Asymmetry_err", "Smoothness", "Smoothness_err", "Gini", "Gini_err"])


wis_properties = {
    "FRL49": {
        "Type": "E★",
        "Distance (Mpc)": 85.7,
        "PA": 160,
        "axis_ratio": 0.8  ,
        "log_MH2": 8.68,
        "log_SigmaH2_1kpc": 2.91,
        "log_Mstar": 10.30,
        "sigma_kms": None,
        "ReKs_arcsec": 3,
        "log_SFR": 0.78,
        "log_mu": 9.31,
        "Beam_arcsec": 0.19,
        "Beam_pc": 77.2,
        "Mass_Ref": "Lelli+ subm.",
        "Data_Ref": "Lelli+ subm."
    },

    "MRK567": {
        "Type": "S",
        "Distance (Mpc)": 140.6,
        "PA": 70,
        "axis_ratio": 0.9  ,
        "log_MH2": 8.79,
        "log_SigmaH2_1kpc": 3.28,
        "log_Mstar": 11.26,
        "sigma_kms": None,
        "ReKs_arcsec": 6,
        "log_SFR": 1.30,
        "log_mu": 9.24,
        "Beam_arcsec": 0.14,
        "Beam_pc": 93.4,
        "Mass_Ref": "C17",
        "Data_Ref": None
    },

    "NGC0383": {
        "Type": "E",
        "Distance (Mpc)": 66.6,
                "PA": 25,
        "axis_ratio": 0.88  ,
        "log_MH2": 9.18,
        "log_SigmaH2_1kpc": 2.66,
        "log_Mstar": 11.82,
        "sigma_kms": 239,
        "ReKs_arcsec": 11,
        "log_SFR": 0.00,
        "log_mu": 9.92,
        "Beam_arcsec": 0.13,
        "Beam_pc": 42.8,
        "Mass_Ref": "MASSIVE",
        "Data_Ref": "North et al. (2019)"
    },

    "NGC0449": {
        "Type": "S",
        "Distance (Mpc)": 66.3,
        "log_MH2": 9.50,
        "log_SigmaH2_1kpc": 2.24,
        "log_Mstar": 10.07,
        "sigma_kms": 250,
        "ReKs_arcsec": 7,
        "log_SFR": 1.19,
        "log_mu": 8.60,
        "Beam_arcsec": 0.66,
        "Beam_pc": 211.2,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None,
                        "PA": 75,
        "axis_ratio": 0.54  
    },

    "NGC0524": {
        "Type": "E",
        "Distance (Mpc)": 23.3,
        "log_MH2": 7.95,
        "log_SigmaH2_1kpc": 1.41,
        "log_Mstar": 11.40,
        "sigma_kms": 220,
        "ReKs_arcsec": 24,
        "log_SFR": -0.56,
        "log_mu": 9.75,
        "Beam_arcsec": 0.32,
        "Beam_pc": 36.7,
        "Mass_Ref": "z0MGS",
        "Data_Ref": "Smith et al. (2019)"
                                ,"PA": 55,
        "axis_ratio": 0.9  
    },

    "NGC0612": {
        "Type": "E",
        "Distance (Mpc)": 130.4,
        "log_MH2": 10.30,
        "log_SigmaH2_1kpc": 1.73,
        "log_Mstar": 11.76,
        "sigma_kms": None,
        "ReKs_arcsec": 9,
        "log_SFR": 0.85,
        "log_mu": 9.13,
        "Beam_arcsec": 0.19,
        "Beam_pc": 122.2,
        "Mass_Ref": "MKs",
        "Data_Ref": "Ruffa+ in prep"
                                        ,"PA": 172,
        "axis_ratio": 0.63
    },

    "NGC0708": {
        "Type": "E",
        "Distance (Mpc)": 58.3,
        "log_MH2": 8.48,
        "log_SigmaH2_1kpc": 2.04,
        "log_Mstar": 11.75,
        "sigma_kms": 230,
        "ReKs_arcsec": 24,
        "log_SFR": -0.29,
        "log_mu": 9.30,
        "Beam_arcsec": 0.09,
        "Beam_pc": 24.1,
        "Mass_Ref": "MASSIVE",
        "Data_Ref": "North et al. (2021)"
                                                ,"PA": 40,
        "axis_ratio": 0.88
    },

    "NGC1387": {
        "Type": "E",
        "Distance (Mpc)": 19.9,
        "log_MH2": 8.33,
        "log_SigmaH2_1kpc": 2.04,
        "log_Mstar": 10.67,
        "sigma_kms": 87,
        "ReKs_arcsec": 16,
        "log_SFR": -0.68,
        "log_mu": 9.51,
        "Beam_arcsec": 0.42,
        "Beam_pc": 40.3,
        "Mass_Ref": "z0MGS",
        "Data_Ref": "Boyce+ in prep"
                                                        ,"PA": 131,
        "axis_ratio": 0.98
    },

    "NGC1574": {
        "Type": "E",
        "Distance (Mpc)": 19.3,
        "log_MH2": 7.64,
        "log_SigmaH2_1kpc": 2.02,
        "log_Mstar": 10.79,
        "sigma_kms": 180,
        "ReKs_arcsec": 21,
        "log_SFR": -0.91,
        "log_mu": 9.41,
        "Beam_arcsec": 0.17,
        "Beam_pc": 15.4,
        "Mass_Ref": "z0MGS",
        "Data_Ref": "Ruffa+ in prep"
                                                                ,"PA": 50,
        "axis_ratio": 0.91
    },

    "NGC3169": {
        "Type": "S",
        "Distance (Mpc)": 18.7,
        "log_MH2": 9.53,
        "log_SigmaH2_1kpc": 2.29,
        "log_Mstar": 10.84,
        "sigma_kms": 165,
        "ReKs_arcsec": 86,
        "log_SFR": 0.29,
        "log_mu": 8.26,
        "Beam_arcsec": 0.60,
        "Beam_pc": 54.0,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                        ,"PA": 53,
        "axis_ratio": 0.74
    },

    "NGC3368": {
        "Type": "S",
        "Distance (Mpc)": 18.0,
        "log_MH2": 9.03,
        "log_SigmaH2_1kpc": 2.46,
        "log_Mstar": 10.67,
        "sigma_kms": 102,
        "ReKs_arcsec": 37,
        "log_SFR": -0.29,
        "log_mu": 8.87,
        "Beam_arcsec": 0.20,
        "Beam_pc": 17.9,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                ,"PA": 135,
        "axis_ratio": 0.68
    },

    "NGC3607": {
        "Type": "E",
        "Distance (Mpc)": 22.2,
        "log_MH2": 8.42,
        "log_SigmaH2_1kpc": 1.86,
        "log_Mstar": 11.34,
        "sigma_kms": 207,
        "ReKs_arcsec": 22,
        "log_SFR": -0.54,
        "log_mu": 9.80,
        "Beam_arcsec": 0.55,
        "Beam_pc": 59.0,
        "Mass_Ref": "A3D",
        "Data_Ref": None
                                                                                        ,"PA": 125,
        "axis_ratio": 0.89
    },

    "NGC4061": {
        "Type": "E",
        "Distance (Mpc)": 94.1,
        "log_MH2": 9.43,
        "log_SigmaH2_1kpc": 2.43,
        "log_Mstar": 11.64,
        "sigma_kms": None,
        "ReKs_arcsec": 8,
        "log_SFR": -0.71,
        "log_mu": 9.71,
        "Beam_arcsec": 0.13,
        "Beam_pc": 59.2,
        "Mass_Ref": "MASSIVE",
        "Data_Ref": None
                                                                                                ,"PA": 65,
        "axis_ratio": 0.7
    },

    "NGC4429": {
        "Type": "E",
        "Distance (Mpc)": 16.5,
        "log_MH2": 8.00,
        "log_SigmaH2_1kpc": 1.60,
        "log_Mstar": 11.17,
        "sigma_kms": 177,
        "ReKs_arcsec": 49,
        "log_SFR": -0.84,
        "log_mu": 9.19,
        "Beam_arcsec": 0.16,
        "Beam_pc": 12.8,
        "Mass_Ref": "A3D",
        "Data_Ref": "Davis et al. (2018)"
                                                                                                        ,"PA": 97,
        "axis_ratio": 0.34
    },

    "NGC4435": {
        "Type": "E",
        "Distance (Mpc)": 16.5,
        "log_MH2": 8.63,
        "log_SigmaH2_1kpc": 1.63,
        "log_Mstar": 10.69,
        "sigma_kms": 153,
        "ReKs_arcsec": 29,
        "log_SFR": -0.84,
        "log_mu": 9.18,
        "Beam_arcsec": 0.24,
        "Beam_pc": 19.1,
        "Mass_Ref": "A3D",
        "Data_Ref": None
                                                                                                                ,"PA": 15,
        "axis_ratio": 0.68
    },

    "NGC4438": {
        "Type": "S",
        "Distance (Mpc)": 16.5,
        "log_MH2": 9.56,
        "log_SigmaH2_1kpc": 2.38,
        "log_Mstar": 10.75,
        "sigma_kms": 142,
        "ReKs_arcsec": 23,
        "log_SFR": -0.30,
        "log_mu": 9.42,
        "Beam_arcsec": 0.56,
        "Beam_pc": 45.1,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                                                        ,"PA": 21,
        "axis_ratio": 0.54
    },

    "NGC4501": {
        "Type": "S",
        "Distance (Mpc)": 15.3,
        "log_MH2": 8.90,
        "log_SigmaH2_1kpc": 2.09,
        "log_Mstar": 11.00,
        "sigma_kms": 102,
        "ReKs_arcsec": 58,
        "log_SFR": 0.43,
        "log_mu": 8.94,
        "Beam_arcsec": 0.63,
        "Beam_pc": 42.6,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                                                                ,"PA": 140,
        "axis_ratio": 0.44
    },

    "NGC4697": {
        "Type": "E",
        "Distance (Mpc)": 11.4,
        "log_MH2": 7.20,
        "log_SigmaH2_1kpc": 0.77,
        "log_Mstar": 11.07,
        "sigma_kms": 169,
        "ReKs_arcsec": 40,
        "log_SFR": -1.08,
        "log_mu": 9.59,
        "Beam_arcsec": 0.55,
        "Beam_pc": 30.5,
        "Mass_Ref": "A3D",
        "Data_Ref": "Davis et al. (2017)"
                                                                                                                                        ,"PA": 68,
        "axis_ratio": 0.63
    },

    "NGC4826": {
        "Type": "S",
        "Distance (Mpc)": 7.4,
        "log_MH2": 7.89,
        "log_SigmaH2_1kpc": 2.59,
        "log_Mstar": 10.20,
        "sigma_kms": 90,
        "ReKs_arcsec": 69,
        "log_SFR": -0.71,
        "log_mu": 8.62,
        "Beam_arcsec": 0.18,
        "Beam_pc": 6.5,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                                                                                ,"PA": 110,
        "axis_ratio": 0.57
    },

    "NGC5064": {
        "Type": "S",
        "Distance (Mpc)": 34.0,
        "log_MH2": 9.90,
        "log_SigmaH2_1kpc": 2.75,
        "log_Mstar": 10.93,
        "sigma_kms": 210,
        "ReKs_arcsec": 18,
        "log_SFR": 0.11,
        "log_mu": 9.19,
        "Beam_arcsec": 0.06,
        "Beam_pc": 9.9,
        "Mass_Ref": "z0MGS",
        "Data_Ref": "Onishi+ in prep"
                                                                                                                                                        ,"PA": 35,
        "axis_ratio": 0.52
    },

    "NGC5765b": {
        "Type": "S",
        "Distance (Mpc)": 114.0,
        "log_MH2": 10.08,
        "log_SigmaH2_1kpc": 2.96,
        "log_Mstar": 11.21,
        "sigma_kms": None,
        "ReKs_arcsec": 7,
        "log_SFR": 1.43,
        "log_mu": 9.30,
        "Beam_arcsec": 0.32,
        "Beam_pc": 178.4,
        "Mass_Ref": "MKs",
        "Data_Ref": None
                                                                                                                                                                ,"PA": 10,
        "axis_ratio": 0.9
    },

    "NGC5806": {
        "Type": "S",
        "Distance (Mpc)": 21.4,
        "log_MH2": 8.97,
        "log_SigmaH2_1kpc": 1.97,
        "log_Mstar": 10.57,
        "sigma_kms": 110,
        "ReKs_arcsec": 30,
        "log_SFR": -0.03,
        "log_mu": 8.80,
        "Beam_arcsec": 0.30,
        "Beam_pc": 31.0,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                                                                                                        ,"PA": 170,
        "axis_ratio": 0.42
    },

    "NGC6753": {
        "Type": "S",
        "Distance (Mpc)": 42.0,
        "log_MH2": 9.62,
        "log_SigmaH2_1kpc": 2.72,
        "log_Mstar": 10.78,
        "sigma_kms": 214,
        "ReKs_arcsec": 20,
        "log_SFR": 0.32,
        "log_mu": 8.78,
        "Beam_arcsec": 0.14,
        "Beam_pc": 28.4,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                                                                                                                ,"PA": 20,
        "axis_ratio": 0.88
    },

    "NGC6958": {
        "Type": "E",
        "Distance (Mpc)": 35.4,
        "log_MH2": 8.66,
        "log_SigmaH2_1kpc": 1.99,
        "log_Mstar": 10.76,
        "sigma_kms": 168,
        "ReKs_arcsec": 12,
        "log_SFR": -0.58,
        "log_mu": 9.35,
        "Beam_arcsec": 0.13,
        "Beam_pc": 19.0,
        "Mass_Ref": "z0MGS",
        "Data_Ref": "Thater+ in prep"
                                                                                                                                                                                        ,"PA": 90,
        "axis_ratio": 0.86
    },

    "NGC7052": {
        "Type": "E",
        "Distance (Mpc)": 51.6,
        "log_MH2": 9.26,
        "log_SigmaH2_1kpc": 2.23,
        "log_Mstar": 11.75,
        "sigma_kms": 266,
        "ReKs_arcsec": 15,
        "log_SFR": -0.07,
        "log_mu": 9.82,
        "Beam_arcsec": 0.13,
        "Beam_pc": 32.1,
        "Mass_Ref": "MASSIVE",
        "Data_Ref": "Smith et al. (2021)"
                                                                                                                                                                                                ,"PA": 65,
        "axis_ratio": 0.48
    },

    "NGC7172": {
        "Type": "E",
        "Distance (Mpc)": 33.9,
        "log_MH2": 9.78,
        "log_SigmaH2_1kpc": 2.53,
        "log_Mstar": 10.76,
        "sigma_kms": 180,
        "ReKs_arcsec": 19,
        "log_SFR": 0.38,
        "log_mu": 8.97,
        "Beam_arcsec": 0.14,
        "Beam_pc": 22.2,
        "Mass_Ref": "z0MGS",
        "Data_Ref": None
                                                                                                                                                                                                        ,"PA": 98,
        "axis_ratio": 0.46
    }
}


phangs_properties = [
    {"name":"NGC 0247","logMstar":9.53,"r25":10.6,"Re":5.0,"la":3.3,"logSFR":-0.75,"logLCO":6.79,"Corr":1.42,"logMstarHI":9.24,"is_limit":False},
    {"name":"NGC 0253","logMstar":10.64,"r25":14.4,"Re":4.7,"la":2.8,"logSFR":0.70,"logLCO":8.96,"Corr":1.00,"logMstarHI":9.33,"is_limit":False},
    {"name":"NGC 0300","logMstar":9.27,"r25":5.9,"Re":2.0,"la":1.3,"logSFR":-0.82,"logLCO":6.61,"Corr":1.50,"logMstarHI":9.32,"is_limit":False},
    {"name":"NGC 0628","logMstar":10.34,"r25":14.1,"Re":3.9,"la":2.9,"logSFR":0.24,"logLCO":8.41,"Corr":1.73,"logMstarHI":9.70,"is_limit":False},
    {"name":"NGC 0685","logMstar":10.07,"r25":8.7,"Re":5.0,"la":3.1,"logSFR":-0.38,"logLCO":7.87,"Corr":1.25,"logMstarHI":9.57,"is_limit":False},
    {"name":"NGC 1068","logMstar":10.91,"r25":12.4,"Re":0.9,"la":7.3,"logSFR":1.64,"logLCO":9.23,"Corr":1.30,"logMstarHI":9.06,"is_limit":False},
    {"name":"NGC 1097","logMstar":10.76,"r25":20.9,"Re":2.6,"la":4.3,"logSFR":0.68,"logLCO":8.93,"Corr":1.31,"logMstarHI":9.61,"is_limit":False},
    {"name":"NGC 1087","logMstar":9.94,"r25":6.9,"Re":3.2,"la":2.1,"logSFR":0.11,"logLCO":8.32,"Corr":1.06,"logMstarHI":9.10,"is_limit":False},
    {"name":"NGC 1313","logMstar":9.26,"r25":7.0,"Re":2.5,"la":2.1,"logSFR":-0.14,"logLCO":None,"Corr":None,"logMstarHI":9.28,"is_limit":False},
    {"name":"NGC 1300","logMstar":10.62,"r25":16.4,"Re":6.5,"la":3.7,"logSFR":0.07,"logLCO":8.50,"Corr":1.28,"logMstarHI":9.38,"is_limit":False},
    {"name":"NGC 1317","logMstar":10.62,"r25":8.5,"Re":1.8,"la":2.4,"logSFR":-0.32,"logLCO":8.10,"Corr":1.28,"logMstarHI":None,"is_limit":False},
    {"name":"IC 1954","logMstar":9.67,"r25":5.6,"Re":2.4,"la":1.5,"logSFR":-0.44,"logLCO":7.78,"Corr":1.10,"logMstarHI":8.85,"is_limit":False},
    {"name":"NGC 1365","logMstar":11.00,"r25":34.2,"Re":2.8,"la":13.1,"logSFR":1.24,"logLCO":9.49,"Corr":1.36,"logMstarHI":9.94,"is_limit":False},
    {"name":"NGC 1385","logMstar":9.98,"r25":8.5,"Re":3.4,"la":2.6,"logSFR":0.32,"logLCO":8.37,"Corr":1.09,"logMstarHI":9.19,"is_limit":False},
    {"name":"NGC 1433","logMstar":10.87,"r25":16.8,"Re":4.3,"la":6.9,"logSFR":0.05,"logLCO":8.47,"Corr":1.38,"logMstarHI":9.40,"is_limit":False},
    {"name":"NGC 1511","logMstar":9.92,"r25":8.2,"Re":2.4,"la":1.7,"logSFR":0.35,"logLCO":8.22,"Corr":1.09,"logMstarHI":9.57,"is_limit":False},
    {"name":"NGC 1512","logMstar":10.72,"r25":23.1,"Re":4.8,"la":6.2,"logSFR":0.11,"logLCO":8.26,"Corr":1.45,"logMstarHI":9.88,"is_limit":False},
    {"name":"NGC 1546","logMstar":10.37,"r25":9.5,"Re":2.2,"la":2.1,"logSFR":-0.08,"logLCO":8.44,"Corr":1.13,"logMstarHI":8.68,"is_limit":False},
    {"name":"NGC 1559","logMstar":10.37,"r25":11.8,"Re":3.9,"la":2.4,"logSFR":0.60,"logLCO":8.66,"Corr":1.11,"logMstarHI":9.52,"is_limit":False},
    {"name":"NGC 1566","logMstar":10.79,"r25":18.6,"Re":3.2,"la":3.9,"logSFR":0.66,"logLCO":8.89,"Corr":1.22,"logMstarHI":9.80,"is_limit":False},
    {"name":"NGC 1637","logMstar":9.95,"r25":5.4,"Re":2.8,"la":1.8,"logSFR":-0.20,"logLCO":7.98,"Corr":1.10,"logMstarHI":9.20,"is_limit":False},
    {"name":"NGC 1672","logMstar":10.73,"r25":17.4,"Re":3.4,"la":5.8,"logSFR":0.88,"logLCO":9.05,"Corr":1.25,"logMstarHI":10.21,"is_limit":False},
    {"name":"NGC 1809","logMstar":9.77,"r25":10.9,"Re":4.5,"la":2.4,"logSFR":0.76,"logLCO":7.49,"Corr":4.24,"logMstarHI":9.60,"is_limit":False},
    {"name":"NGC 1792","logMstar":10.62,"r25":13.1,"Re":4.1,"la":2.4,"logSFR":0.57,"logLCO":8.95,"Corr":1.11,"logMstarHI":9.25,"is_limit":False},
    {"name":"NGC 2090","logMstar":10.04,"r25":7.7,"Re":1.9,"la":1.7,"logSFR":-0.39,"logLCO":7.67,"Corr":1.47,"logMstarHI":9.37,"is_limit":False},
    {"name":"NGC 2283","logMstar":9.89,"r25":5.5,"Re":3.2,"la":1.9,"logSFR":-0.28,"logLCO":7.69,"Corr":1.16,"logMstarHI":9.70,"is_limit":False},
    {"name":"NGC 2566","logMstar":10.71,"r25":14.5,"Re":5.1,"la":4.0,"logSFR":0.93,"logLCO":9.06,"Corr":1.13,"logMstarHI":9.37,"is_limit":False},
    {"name":"NGC 2775","logMstar":11.07,"r25":14.3,"Re":4.6,"la":4.1,"logSFR":-0.06,"logLCO":8.40,"Corr":1.29,"logMstarHI":8.65,"is_limit":False},
    {"name":"NGC 2835","logMstar":10.00,"r25":11.4,"Re":3.3,"la":2.2,"logSFR":0.10,"logLCO":7.71,"Corr":1.72,"logMstarHI":9.48,"is_limit":False},
    {"name":"NGC 2903","logMstar":10.64,"r25":17.4,"Re":3.7,"la":3.5,"logSFR":0.49,"logLCO":8.76,"Corr":1.18,"logMstarHI":9.54,"is_limit":False},
    {"name":"NGC 2997","logMstar":10.73,"r25":21.0,"Re":6.1,"la":4.0,"logSFR":0.64,"logLCO":8.97,"Corr":1.25,"logMstarHI":9.86,"is_limit":False},
    {"name":"NGC 3059","logMstar":10.38,"r25":11.2,"Re":5.0,"la":3.2,"logSFR":0.38,"logLCO":8.59,"Corr":1.07,"logMstarHI":9.75,"is_limit":False},
    {"name":"NGC 3137","logMstar":9.88,"r25":13.2,"Re":4.1,"la":3.0,"logSFR":-0.30,"logLCO":7.60,"Corr":1.35,"logMstarHI":9.68,"is_limit":False},
    {"name":"NGC 3239","logMstar":9.18,"r25":5.7,"Re":3.1,"la":2.0,"logSFR":-0.41,"logLCO":6.62,"Corr":1.54,"logMstarHI":9.16,"is_limit":True},
    {"name":"NGC 3351","logMstar":10.37,"r25":10.5,"Re":3.0,"la":2.1,"logSFR":0.12,"logLCO":8.13,"Corr":1.55,"logMstarHI":8.93,"is_limit":False},
    {"name":"NGC 3489","logMstar":10.29,"r25":5.9,"Re":1.3,"la":1.4,"logSFR":-1.59,"logLCO":6.89,"Corr":1.37,"logMstarHI":7.40,"is_limit":False},
    {"name":"NGC 3511","logMstar":10.03,"r25":12.2,"Re":4.4,"la":2.4,"logSFR":-0.09,"logLCO":8.15,"Corr":1.07,"logMstarHI":9.37,"is_limit":False},
    {"name":"NGC 3507","logMstar":10.40,"r25":10.0,"Re":3.7,"la":2.3,"logSFR":-0.00,"logLCO":8.34,"Corr":1.17,"logMstarHI":9.32,"is_limit":False},
    {"name":"NGC 3521","logMstar":11.03,"r25":16.0,"Re":3.9,"la":4.9,"logSFR":0.57,"logLCO":8.98,"Corr":1.18,"logMstarHI":9.83,"is_limit":False},
    {"name":"NGC 3596","logMstar":9.66,"r25":6.0,"Re":1.6,"la":2.0,"logSFR":-0.52,"logLCO":7.81,"Corr":1.13,"logMstarHI":8.85,"is_limit":False},
    {"name":"NGC 3599","logMstar":10.04,"r25":6.9,"Re":1.7,"la":2.0,"logSFR":-1.35,"logLCO":6.70,"Corr":1.35,"logMstarHI":None,"is_limit":True},
    {"name":"NGC 3621","logMstar":10.06,"r25":9.8,"Re":2.7,"la":2.0,"logSFR":-0.00,"logLCO":8.13,"Corr":1.27,"logMstarHI":9.66,"is_limit":False},
    {"name":"NGC 3626","logMstar":10.46,"r25":8.6,"Re":1.8,"la":2.1,"logSFR":-0.68,"logLCO":7.75,"Corr":1.14,"logMstarHI":8.89,"is_limit":False},
    {"name":"NGC 3627","logMstar":10.84,"r25":16.9,"Re":3.6,"la":3.7,"logSFR":0.59,"logLCO":8.98,"Corr":1.16,"logMstarHI":9.09,"is_limit":False},
    {"name":"NGC 4207","logMstar":9.72,"r25":3.4,"Re":1.4,"la":0.7,"logSFR":-0.72,"logLCO":7.71,"Corr":1.03,"logMstarHI":8.58,"is_limit":False},
    {"name":"NGC 4254","logMstar":10.42,"r25":9.6,"Re":2.4,"la":1.8,"logSFR":0.49,"logLCO":8.93,"Corr":1.15,"logMstarHI":9.48,"is_limit":False},
    {"name":"NGC 4293","logMstar":10.52,"r25":14.3,"Re":4.7,"la":2.8,"logSFR":-0.30,"logLCO":8.12,"Corr":1.57,"logMstarHI":7.67,"is_limit":False},
    {"name":"NGC 4298","logMstar":10.04,"r25":5.5,"Re":3.0,"la":1.6,"logSFR":-0.34,"logLCO":8.26,"Corr":1.09,"logMstarHI":8.87,"is_limit":False},
    {"name":"NGC 4303","logMstar":10.51,"r25":17.0,"Re":3.4,"la":3.1,"logSFR":0.73,"logLCO":9.00,"Corr":1.40,"logMstarHI":9.67,"is_limit":False},
    {"name":"NGC 4321","logMstar":10.75,"r25":13.5,"Re":5.5,"la":3.6,"logSFR":0.55,"logLCO":9.02,"Corr":1.25,"logMstarHI":9.43,"is_limit":False},
    {"name":"NGC 4424","logMstar":9.93,"r25":7.2,"Re":3.7,"la":2.2,"logSFR":-0.53,"logLCO":7.59,"Corr":1.16,"logMstarHI":8.30,"is_limit":False},
    {"name":"NGC 4457","logMstar":10.42,"r25":6.1,"Re":1.5,"la":2.2,"logSFR":-0.52,"logLCO":8.21,"Corr":1.15,"logMstarHI":8.36,"is_limit":False},
    {"name":"NGC 4459","logMstar":10.68,"r25":9.6,"Re":2.1,"la":3.3,"logSFR":-0.65,"logLCO":7.46,"Corr":2.41,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4476","logMstar":9.81,"r25":4.3,"Re":1.2,"la":1.2,"logSFR":-1.39,"logLCO":7.05,"Corr":1.09,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4477","logMstar":10.59,"r25":8.5,"Re":2.1,"la":2.1,"logSFR":-1.10,"logLCO":6.76,"Corr":1.58,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4496A","logMstar":9.55,"r25":7.3,"Re":3.0,"la":1.9,"logSFR":-0.21,"logLCO":7.55,"Corr":1.15,"logMstarHI":9.24,"is_limit":False},
    {"name":"NGC 4535","logMstar":10.54,"r25":18.7,"Re":6.3,"la":3.8,"logSFR":0.34,"logLCO":8.61,"Corr":1.78,"logMstarHI":9.56,"is_limit":False},
    {"name":"NGC 4536","logMstar":10.40,"r25":16.7,"Re":4.4,"la":2.7,"logSFR":0.53,"logLCO":8.62,"Corr":1.06,"logMstarHI":9.54,"is_limit":False},
    {"name":"NGC 4540","logMstar":9.79,"r25":5.0,"Re":2.0,"la":1.4,"logSFR":-0.78,"logLCO":7.69,"Corr":1.16,"logMstarHI":8.44,"is_limit":False},
    {"name":"NGC 4548","logMstar":10.70,"r25":13.1,"Re":5.4,"la":3.0,"logSFR":-0.28,"logLCO":8.16,"Corr":2.00,"logMstarHI":8.84,"is_limit":False},
    {"name":"NGC 4569","logMstar":10.81,"r25":20.9,"Re":5.9,"la":4.3,"logSFR":0.12,"logLCO":8.81,"Corr":1.40,"logMstarHI":8.84,"is_limit":False},
    {"name":"NGC 4571","logMstar":10.10,"r25":7.7,"Re":3.8,"la":2.0,"logSFR":-0.54,"logLCO":7.88,"Corr":1.55,"logMstarHI":8.70,"is_limit":False},
    {"name":"NGC 4579","logMstar":11.15,"r25":15.3,"Re":5.4,"la":4.4,"logSFR":0.33,"logLCO":8.79,"Corr":1.38,"logMstarHI":9.02,"is_limit":False},
    {"name":"NGC 4596","logMstar":10.59,"r25":9.0,"Re":2.7,"la":3.8,"logSFR":-0.96,"logLCO":6.72,"Corr":1.83,"logMstarHI":None,"is_limit":False},
    {"name":"NGC 4654","logMstar":10.57,"r25":15.1,"Re":5.6,"la":4.0,"logSFR":0.58,"logLCO":8.84,"Corr":1.18,"logMstarHI":9.75,"is_limit":False},
    {"name":"NGC 4689","logMstar":10.24,"r25":8.3,"Re":4.7,"la":3.0,"logSFR":-0.39,"logLCO":8.22,"Corr":1.19,"logMstarHI":8.54,"is_limit":False},
    {"name":"NGC 4694","logMstar":9.90,"r25":4.6,"Re":1.9,"la":1.6,"logSFR":-0.81,"logLCO":7.41,"Corr":1.30,"logMstarHI":8.51,"is_limit":False},
    {"name":"NGC 4731","logMstar":9.50,"r25":12.2,"Re":7.3,"la":3.0,"logSFR":-0.22,"logLCO":7.29,"Corr":2.52,"logMstarHI":9.44,"is_limit":False},
    {"name":"NGC 4781","logMstar":9.64,"r25":6.1,"Re":2.0,"la":1.1,"logSFR":-0.32,"logLCO":7.82,"Corr":1.05,"logMstarHI":8.94,"is_limit":False},
    {"name":"NGC 4826","logMstar":10.24,"r25":6.7,"Re":1.5,"la":1.1,"logSFR":-0.69,"logLCO":7.79,"Corr":1.28,"logMstarHI":8.26,"is_limit":False},
    {"name":"NGC 4941","logMstar":10.18,"r25":7.3,"Re":3.4,"la":2.2,"logSFR":-0.35,"logLCO":7.80,"Corr":1.27,"logMstarHI":8.49,"is_limit":False},
    {"name":"NGC 4951","logMstar":9.79,"r25":6.9,"Re":1.9,"la":1.9,"logSFR":-0.46,"logLCO":7.65,"Corr":1.22,"logMstarHI":9.21,"is_limit":False},
    {"name":"NGC 4945","logMstar":10.36,"r25":11.8,"Re":4.5,"la":1.6,"logSFR":0.19,"logLCO":8.77,"Corr":0.97,"logMstarHI":8.92,"is_limit":False},
    {"name":"NGC 5042","logMstar":9.90,"r25":10.2,"Re":3.3,"la":2.4,"logSFR":-0.22,"logLCO":7.69,"Corr":1.84,"logMstarHI":9.29,"is_limit":False},
    {"name":"NGC 5068","logMstar":9.41,"r25":5.7,"Re":2.0,"la":1.3,"logSFR":-0.56,"logLCO":7.26,"Corr":1.38,"logMstarHI":8.82,"is_limit":False},
    {"name":"NGC 5134","logMstar":10.41,"r25":7.9,"Re":2.9,"la":2.1,"logSFR":-0.34,"logLCO":7.98,"Corr":1.14,"logMstarHI":8.92,"is_limit":False},
    {"name":"NGC 5128","logMstar":10.97,"r25":13.7,"Re":4.7,"la":4.1,"logSFR":0.09,"logLCO":8.40,"Corr":0.98,"logMstarHI":8.43,"is_limit":False},
    {"name":"NGC 5236","logMstar":10.53,"r25":9.7,"Re":3.5,"la":2.4,"logSFR":0.62,"logLCO":8.84,"Corr":1.14,"logMstarHI":9.98,"is_limit":False},
    {"name":"NGC 5248","logMstar":10.41,"r25":8.8,"Re":3.2,"la":2.0,"logSFR":0.36,"logLCO":8.77,"Corr":1.14,"logMstarHI":9.50,"is_limit":False},
    {"name":"ESO097-013","logMstar":10.53,"r25":5.3,"Re":1.9,"la":1.8,"logSFR":0.61,"logLCO":8.42,"Corr":1.40,"logMstarHI":9.81,"is_limit":False},
    {"name":"NGC 5530","logMstar":10.08,"r25":8.6,"Re":3.4,"la":1.7,"logSFR":-0.48,"logLCO":7.89,"Corr":1.34,"logMstarHI":9.11,"is_limit":False},
    {"name":"NGC 5643","logMstar":10.34,"r25":9.7,"Re":3.5,"la":1.6,"logSFR":0.41,"logLCO":8.56,"Corr":1.06,"logMstarHI":9.12,"is_limit":False},
    {"name":"NGC 6300","logMstar":10.47,"r25":9.0,"Re":3.6,"la":2.1,"logSFR":0.29,"logLCO":8.46,"Corr":1.12,"logMstarHI":9.13,"is_limit":False},
    {"name":"NGC 6744","logMstar":10.72,"r25":21.4,"Re":7.0,"la":4.8,"logSFR":0.38,"logLCO":8.27,"Corr":2.75,"logMstarHI":10.31,"is_limit":False},
    {"name":"IC 5273","logMstar":9.73,"r25":6.3,"Re":2.5,"la":1.3,"logSFR":-0.27,"logLCO":7.63,"Corr":1.14,"logMstarHI":8.95,"is_limit":False},
    {"name":"NGC 7456","logMstar":9.65,"r25":9.4,"Re":4.4,"la":2.9,"logSFR":-0.43,"logLCO":7.13,"Corr":2.02,"logMstarHI":9.28,"is_limit":False},
    {"name":"NGC 7496","logMstar":10.00,"r25":9.1,"Re":3.8,"la":1.5,"logSFR":0.35,"logLCO":8.33,"Corr":1.15,"logMstarHI":9.07,"is_limit":False},
    {"name":"IC 5332","logMstar":9.68,"r25":8.0,"Re":3.6,"la":2.8,"logSFR":-0.39,"logLCO":7.09,"Corr":2.26,"logMstarHI":9.30,"is_limit":False},
    {"name":"NGC 7743","logMstar":10.36,"r25":7.7,"Re":2.9,"la":1.9,"logSFR":-0.67,"logLCO":7.50,"Corr":2.65,"logMstarHI":8.50,"is_limit":False},
    {"name":"NGC 7793","logMstar":9.36,"r25":5.5,"Re":1.9,"la":1.1,"logSFR":-0.57,"logLCO":7.23,"Corr":1.34,"logMstarHI":8.70,"is_limit":False}
]

phangs_properties2 = {
    "ESO097-013X": {"vLSR": 430.3, "PA": 36.7, "i": 64.3, "Distance (Mpc)": 4.20},
    "IC 1954": {"vLSR": 1039.1, "PA": 63.4, "i": 57.1, "Distance (Mpc)": 12.80},
    "IC 5273": {"vLSR": 1286.0, "PA": 234.1, "i": 52.0, "Distance (Mpc)": 14.18},
    "IC 5332": {"vLSR": 699.3, "PA": 74.4, "i": 26.9, "Distance (Mpc)": 9.01},
    "NGC 0247X": {"vLSR": 148.8, "PA": 167.4, "i": 76.4, "Distance (Mpc)": 3.71},
    "NGC 0253X": {"vLSR": 235.4, "PA": 52.5, "i": 75.0, "Distance (Mpc)": 3.70},
    "NGC 0300X": {"vLSR": 155.5, "PA": 114.3, "i": 39.8, "Distance (Mpc)": 2.09},
    "NGC 0628": {"vLSR": 650.8, "PA": 20.7, "i": 8.9, "Distance (Mpc)": 9.84},
    "NGC 0685": {"vLSR": 1346.6, "PA": 100.9, "i": 23.0, "Distance (Mpc)": 19.94},
    "NGC 1068X": {"vLSR": 1130.1, "PA": 72.7, "i": 34.7, "Distance (Mpc)": 13.97},
    "NGC 1087": {"vLSR": 1501.5, "PA": 359.1, "i": 42.9, "Distance (Mpc)": 15.85},
    "NGC 1097": {"vLSR": 1257.5, "PA": 122.4, "i": 48.6, "Distance (Mpc)": 13.58},
    "NGC 1313X": {"vLSR": 451.2, "PA": 23.4, "i": 34.8, "Distance (Mpc)": 4.32},
    "NGC 1300": {"vLSR": 1545.4, "PA": 278.0, "i": 31.8, "Distance (Mpc)": 18.99},
    "NGC 1317": {"vLSR": 1930.5, "PA": 221.5, "i": 23.2, "Distance (Mpc)": 19.11},
    "NGC 1365": {"vLSR": 1613.3, "PA": 201.1, "i": 55.4, "Distance (Mpc)": 19.57},
    "NGC 1385": {"vLSR": 1476.8, "PA": 181.3, "i": 44.0, "Distance (Mpc)": 17.22},
    "NGC 1433": {"vLSR": 1057.4, "PA": 199.7, "i": 28.6, "Distance (Mpc)": 18.63},
    "NGC 1511": {"vLSR": 1331.0, "PA": 297.0, "i": 72.7, "Distance (Mpc)": 15.28},
    "NGC 1512": {"vLSR": 871.4, "PA": 261.9, "i": 42.5, "Distance (Mpc)": 18.83},
    "NGC 1546": {"vLSR": 1243.8, "PA": 147.8, "i": 70.3, "Distance (Mpc)": 17.69},
    "NGC 1559": {"vLSR": 1275.2, "PA": 244.5, "i": 65.4, "Distance (Mpc)": 19.44},
    "NGC 1566": {"vLSR": 1483.3, "PA": 214.7, "i": 29.5, "Distance (Mpc)": 17.69},
    "NGC 1637": {"vLSR": 698.9, "PA": 20.6, "i": 31.1, "Distance (Mpc)": 11.70},
    "NGC 1672": {"vLSR": 1318.3, "PA": 134.3, "i": 42.6, "Distance (Mpc)": 19.40},
    "NGC 1809": {"vLSR": 1290.4, "PA": 138.2, "i": 57.6, "Distance (Mpc)": 19.95},
    "NGC 1792": {"vLSR": 1175.9, "PA": 318.9, "i": 65.1, "Distance (Mpc)": 16.20},
    "NGC 2090": {"vLSR": 898.2, "PA": 192.5, "i": 64.5, "Distance (Mpc)": 11.75},
    "NGC 2283": {"vLSR": 821.9, "PA": -4.1, "i": 43.7, "Distance (Mpc)": 13.68},
    "NGC 2566": {"vLSR": 1609.6, "PA": 312.0, "i": 48.5, "Distance (Mpc)": 23.44},
    "NGC 2775": {"vLSR": 1339.2, "PA": 156.5, "i": 41.2, "Distance (Mpc)": 23.15},
    "NGC 2835": {"vLSR": 867.3, "PA": 1.0, "i": 41.3, "Distance (Mpc)": 12.22},
    "NGC 2903": {"vLSR": 547.0, "PA": 203.7, "i": 66.8, "Distance (Mpc)": 10.00},
    "NGC 2997": {"vLSR": 1076.9, "PA": 108.1, "i": 33.0, "Distance (Mpc)": 14.06},
    "NGC 3059": {"vLSR": 1236.5, "PA": -14.8, "i": 29.4, "Distance (Mpc)": 20.23},
    "NGC 3137": {"vLSR": 1086.6, "PA": -0.3, "i": 70.3, "Distance (Mpc)": 16.37},
    "NGC 3239": {"vLSR": 748.3, "PA": 72.9, "i": 60.3, "Distance (Mpc)": 10.86},
    "NGC 3351": {"vLSR": 774.7, "PA": 193.2, "i": 45.1, "Distance (Mpc)": 9.96},
    "NGC 3489X": {"vLSR": 692.1, "PA": 70.0, "i": 63.7, "Distance (Mpc)": 11.86},
    "NGC 3511": {"vLSR": 1096.7, "PA": 256.8, "i": 75.1, "Distance (Mpc)": 13.94},
    "NGC 3507": {"vLSR": 969.4, "PA": 55.8, "i": 21.7, "Distance (Mpc)": 23.55},
    "NGC 3521": {"vLSR": 798.0, "PA": 343.0, "i": 68.8, "Distance (Mpc)": 13.24},
    "NGC 3596": {"vLSR": 1187.9, "PA": 78.4, "i": 25.1, "Distance (Mpc)": 11.30},
    "NGC 3599X": {"vLSR": 836.8, "PA": 41.9, "i": 23.0, "Distance (Mpc)": 19.86},
    "NGC 3621": {"vLSR": 724.3, "PA": 343.8, "i": 65.8, "Distance (Mpc)": 7.06},
    "NGC 3626": {"vLSR": 1470.7, "PA": 165.2, "i": 46.6, "Distance (Mpc)": 20.05},
    "NGC 3627": {"vLSR": 715.4, "PA": 173.1, "i": 57.3, "Distance (Mpc)": 11.32},
    "NGC 4207": {"vLSR": 606.6, "PA": 121.9, "i": 64.5, "Distance (Mpc)": 15.78},
    "NGC 4254": {"vLSR": 2388.2, "PA": 68.1, "i": 34.4, "Distance (Mpc)": 13.10},
    "NGC 4293": {"vLSR": 926.2, "PA": 48.3, "i": 65.0, "Distance (Mpc)": 15.76},
    "NGC 4298": {"vLSR": 1138.1, "PA": 313.9, "i": 59.2, "Distance (Mpc)": 14.92},
    "NGC 4303": {"vLSR": 1559.8, "PA": 312.4, "i": 23.5, "Distance (Mpc)": 16.99},
    "NGC 4321": {"vLSR": 1572.3, "PA": 156.2, "i": 38.5, "Distance (Mpc)": 15.21},
    "NGC 4424": {"vLSR": 447.4, "PA": 88.3, "i": 58.2, "Distance (Mpc)": 16.20},
    "NGC 4457": {"vLSR": 886.0, "PA": 78.7, "i": 17.4, "Distance (Mpc)": 15.10},
    "NGC 4459X": {"vLSR": 1190.1, "PA": 108.8, "i": 47.0, "Distance (Mpc)": 15.85},
    "NGC 4476X": {"vLSR": 1962.7, "PA": 27.4, "i": 60.1, "Distance (Mpc)": 17.54},
    "NGC 4477X": {"vLSR": 1362.2, "PA": 25.7, "i": 33.5, "Distance (Mpc)": 15.76},
    "NGC 4496A": {"vLSR": 1721.8, "PA": 51.1, "i": 53.8, "Distance (Mpc)": 14.86},
    "NGC 4535": {"vLSR": 1953.6, "PA": 179.7, "i": 44.7, "Distance (Mpc)": 15.77},
    "NGC 4536": {"vLSR": 1794.6, "PA": 305.6, "i": 66.0, "Distance (Mpc)": 16.25},
    "NGC 4540": {"vLSR": 1286.5, "PA": 12.8, "i": 28.7, "Distance (Mpc)": 15.76},
    "NGC 4548": {"vLSR": 482.7, "PA": 138.0, "i": 38.3, "Distance (Mpc)": 16.22},
    "NGC 4569": {"vLSR": -225.6, "PA": 18.0, "i": 70.0, "Distance (Mpc)": 15.76},
    "NGC 4571": {"vLSR": 343.0, "PA": 217.5, "i": 32.7, "Distance (Mpc)": 14.90},
    "NGC 4579": {"vLSR": 1516.7, "PA": 91.3, "i": 40.2, "Distance (Mpc)": 21.00},
    "NGC 4596X": {"vLSR": 1883.3, "PA": 120.0, "i": 36.6, "Distance (Mpc)": 15.76},
    "NGC 4654": {"vLSR": 1051.5, "PA": 123.2, "i": 55.6, "Distance (Mpc)": 21.98},
    "NGC 4689": {"vLSR": 1614.2, "PA": 164.1, "i": 38.7, "Distance (Mpc)": 15.00},
    "NGC 4694": {"vLSR": 1168.4, "PA": 143.3, "i": 60.7, "Distance (Mpc)": 15.76},
    "NGC 4731": {"vLSR": 1483.6, "PA": 255.4, "i": 64.0, "Distance (Mpc)": 13.28},
    "NGC 4781": {"vLSR": 1248.3, "PA": 290.0, "i": 59.0, "Distance (Mpc)": 11.31},
    "NGC 4826": {"vLSR": 409.7, "PA": 293.6, "i": 59.1, "Distance (Mpc)": 4.41},
    "NGC 4941": {"vLSR": 1116.0, "PA": 202.2, "i": 53.4, "Distance (Mpc)": 15.00},
    "NGC 4951": {"vLSR": 1176.1, "PA": 91.2, "i": 70.2, "Distance (Mpc)": 15.00},
    "NGC 4945X": {"vLSR": 559.3, "PA": 43.8, "i": 90.0, "Distance (Mpc)": 3.47},
    "NGC 5042": {"vLSR": 1385.6, "PA": 190.6, "i": 49.4, "Distance (Mpc)": 16.78},
    "NGC 5068": {"vLSR": 667.2, "PA": 342.4, "i": 35.7, "Distance (Mpc)": 5.20},
    "NGC 5128": {"vLSR": 549.5, "PA": 32.2, "i": 45.3, "Distance (Mpc)": 3.69},
    "NGC 5134": {"vLSR": 1749.1, "PA": 311.6, "i": 22.7, "Distance (Mpc)": 19.92},
    "NGC 5236": {"vLSR": 509.4, "PA": 225.0, "i": 24.0, "Distance (Mpc)": 4.89},
    "NGC 5248": {"vLSR": 1163.0, "PA": 109.2, "i": 47.4, "Distance (Mpc)": 14.87},
    "NGC 5530": {"vLSR": 1183.2, "PA": 305.4, "i": 61.9, "Distance (Mpc)": 12.27},
    "NGC 5643": {"vLSR": 1191.3, "PA": 318.7, "i": 29.9, "Distance (Mpc)": 12.68},
    "NGC 6300": {"vLSR": 1102.1, "PA": 105.4, "i": 49.6, "Distance (Mpc)": 11.58},
    "NGC 6744": {"vLSR": 832.3, "PA": 14.0, "i": 52.7, "Distance (Mpc)": 9.39},
    "NGC 7456": {"vLSR": 1192.3, "PA": 16.0, "i": 67.3, "Distance (Mpc)": 15.70},
    "NGC 7496": {"vLSR": 1639.2, "PA": 193.7, "i": 35.9, "Distance (Mpc)": 18.72},
    "NGC 7743X": {"vLSR": 1687.3, "PA": 86.2, "i": 37.1, "Distance (Mpc)": 20.32},
    "NGC 7793X": {"vLSR": 222.1, "PA": 290.0, "i": 50.0, "Distance (Mpc)": 3.62}
}


# for t in result: 
#     if 'i' in t.colnames: 
#         print(f"Table: {t.meta.get('name', 'unknown')}, Columns: {t['i']}")

wis_H_phot = Table.read('/Users/administrator/Astro/LLAMA/wisdom_2mass_Hphotometry.fits', format='fits')
phangs_H_phot = Table.read('/Users/administrator/Astro/LLAMA/phangs_2mass_Hphotometry.fits', format='fits')
#wis_H_phot.show_in_browser()

def get_data(name):
    try:
        result = Vizier.query_object(name, catalog="VII/155/rc3")
        if len(result) == 0:
            result = Vizier.query_object(name, catalog="J/A+A/659/A188/ulx-xmm9")

        T_col = result[0]["T"]
        if isinstance(T_col, MaskedColumn):
            T_data = T_col.filled(np.nan)
        else:
            T_data = np.array(T_col)

        T_val = np.nanmedian(T_data)

        # # --- robust coordinate handling ---
        # if "RA2000" in result[0].colnames:
        #     RA_val = result[0]["RA2000"][0]
        #     DEC_val = result[0]["DE2000"][0]
        # else:
        #     RA_val = result[0]["RAJ2000"][0]
        #     DEC_val = result[0]["DEJ2000"][0]

        # c = SkyCoord(RA_val, DEC_val, unit=(u.hourangle, u.deg))

        # ra_deg = float(c.ra.deg)
        # dec_deg = float(c.dec.deg)

        # tab = Ned.query_object(name)
        # print(tab.colnames)
        # for col in tab.colnames:
        #     print(col, tab[col][0])
        # ra_deg = tab["RA(deg)"][0]
        # dec_deg = tab["DEC(deg)"][0]

        Simbad.add_votable_fields("ra", "dec")
        tab = Simbad.query_object(name)
        ra_deg = tab["ra"][0]
        dec_deg = tab["dec"][0]

        return T_val, ra_deg, dec_deg

    except Exception as e:
        print(f"[{name}] failed:", e)
        return np.nan, np.nan, np.nan

###### update wisdom table ######

for name, props in wis_properties.items():
    axis_ratio = props["axis_ratio"]
    i_rad =  np.arccos(np.clip(axis_ratio, -1.0, 1.0))
    props["i"] = np.degrees(i_rad)

print("Updating WISDOM table with Hubble T from Vizier...")
wis_properties = pd.DataFrame.from_dict(wis_properties, orient="index")

for name_str in wis_properties.index:

    max_retries = 3

    for attempt in range(max_retries):
        try:
            wis_properties.loc[name_str, "Hubble Stage"],wis_properties.loc[name_str, "RA"],wis_properties.loc[name_str, "DEC"] = get_data(name_str)

            break

        except (requests.exceptions.ConnectionError,
                RemoteServiceError,
                requests.exceptions.ReadTimeout) as e:

            print(f"⚠️ Vizier query failed for {name_str} (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("❌ All Vizier attempts failed.")


###### update phangs table ######
print("Updating PHANGS table with Hubble Stage from Vizier...")
if isinstance(phangs_properties, list):
    phangs_properties = pd.DataFrame(phangs_properties)

if "name" in phangs_properties.columns:
    phangs_properties = phangs_properties.set_index("name")

for name_str in phangs_properties.index:
    max_retries = 3
    hubble_T = None

    for attempt in range(max_retries):
        try:
            phangs_properties.loc[name_str, "Hubble Stage"],phangs_properties.loc[name_str, "RA"],phangs_properties.loc[name_str, "DEC"] = get_data(name_str)

            break

        except (requests.exceptions.ConnectionError,
                RemoteServiceError,
                requests.exceptions.ReadTimeout) as e:

            print(f"⚠️ Vizier query failed for {name_str} (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("❌ All Vizier attempts failed.")




wis_df = pd.DataFrame(wisdom)
wis_df['Name'] = normalize_name(wis_df['Name'])
wis_df['Name'] = wis_df['Name'].str.replace(" ", "", regex=False)   # remove all spaces
df_wis = wis_properties
df_wis['Name'] = df_wis.index
df_wis['Name'] = normalize_name(df_wis['Name'])
df_wis['Name'] = df_wis['Name'].str.replace(" ", "", regex=False)   # remove all spaces
wis_H_phot_df = wis_H_phot.to_pandas()
wis_H_phot_df['ID'] = normalize_name(wis_H_phot_df['ID'])
df_wis = df_wis.merge(
wis_H_phot_df,
left_on="Name",
right_on="ID",
how="left"
)
            

wis_df = pd.merge(wis_df, df_wis, left_on='Name', right_on='Name',how='left')
D_cm = pd.to_numeric(wis_df["Distance (Mpc)"], errors="coerce") * 3.0856776e24
H_flux = pd.to_numeric(wis_df["H flux"], errors="coerce") if "H flux" in wis_df.columns else pd.Series(np.nan, index=wis_df.index)
L = 4 * np.pi * D_cm**2 * (H_flux/0.21)*1.662 
# only take log10 where L is positive, otherwise set NaN
with np.errstate(invalid="ignore", divide="ignore"):
    wis_df["log LH (L⊙)"] = np.where(L > 0, np.log10(L / 3.828e33), np.nan)



######## save csv, comment out later ##########

out_dir = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "wis_df.csv")
wis_df.to_csv(out_path, index=False)




phangs_df = pd.DataFrame(phangs)
phangs_df['Name'] = normalize_name(phangs_df['Name'])
phangs_df['Name'] = phangs_df['Name'].str.replace(" ", "", regex=False)   # remove all spaces
df_phangs = pd.DataFrame(phangs_properties)
df_phangs2 = pd.DataFrame(phangs_properties2).T.reset_index()
df_phangs2 = df_phangs2.rename(columns={'index': 'Name'})
df_phangs = df_phangs.reset_index()  # moves index to column 'name'
if 'name' in df_phangs.columns[df_phangs.columns.duplicated()]:
    df_phangs = df_phangs.loc[:, ~df_phangs.columns.duplicated()]
df_phangs['name'] = df_phangs['name'].str.replace(" ", "", regex=False)   # remove all spaces
df_phangs2['name'] = normalize_name(df_phangs['name'])
df_phangs2['name'] = df_phangs['name'].str.replace(" ", "", regex=False)   # remove all spaces
phangs_H_phot_df = phangs_H_phot.to_pandas()
phangs_H_phot_df['ID'] = normalize_name(phangs_H_phot_df['ID'])
phangs_H_phot_df['ID'] = phangs_H_phot_df['ID'].str.replace(" ", "", regex=False)
# merge phangs H-photometry into df_phangs using normalized keys
df_phangs = df_phangs.merge(
    phangs_H_phot_df,
    left_on="name",
    right_on="ID",
    how="left"
)
# ensure df_phangs2 was created/renamed correctly and normalize its Name column
df_phangs2 = df_phangs2.rename(columns={'index': 'Name'})
df_phangs2['Name'] = normalize_name(df_phangs2['Name'])
df_phangs2['Name'] = df_phangs2['Name'].str.replace(" ", "", regex=False)

# also normalize df_phangs 'name' (remove spaces already done above but keep for safety)
df_phangs['name'] = normalize_name(df_phangs['name'])
df_phangs['name'] = df_phangs['name'].str.replace(" ", "", regex=False)
# merge additional properties from df_phangs2
df_phangs = df_phangs.merge(
    df_phangs2,
    left_on="name",
    right_on="Name",
    how="left"
)
# Ensure phangs_df 'Name' is in the same normalized form as df_phangs['Name'] before final merge
phangs_df['Name'] = normalize_name(phangs_df['Name'])
phangs_df['Name'] = phangs_df['Name'].str.replace(" ", "", regex=False)

phangs_df = pd.merge(phangs_df, df_phangs, left_on='Name', right_on='Name', how='left')

D_cm = pd.to_numeric(phangs_df["Distance (Mpc)"], errors="coerce") * 3.0856776e24
H_flux = pd.to_numeric(phangs_df["H flux"], errors="coerce") if "H flux" in phangs_df.columns else pd.Series(np.nan, index=phangs_df.index)

L = 4 * np.pi * D_cm**2 * (H_flux/0.21)*1.662 # convert from L H (multiplied by bandwidth) to lambdafnu H
        # only take log10 where L is positive, otherwise set NaN
with np.errstate(invalid="ignore", divide="ignore"):
    phangs_df["log LH (L⊙)"] = np.where(L > 0, np.log10(L / 3.828e33), np.nan)

    ######## save csv, comment out later ##########
out_dir = "/Users/administrator/Astro/LLAMA/ALMA/comp_samples"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "phangs_df.csv")
phangs_df.to_csv(out_path, index=False)
################################################
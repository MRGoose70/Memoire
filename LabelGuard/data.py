"""
Module de gestion des données.
Contient les fonctions pour générer le dataset synthétique et transformer les données en matrices utilisables.
"""

import random
import pandas as pd
import numpy as np

def create_large_training_dataset(n=4080, seed=42) -> pd.DataFrame:
    """
    Génère un DataFrame simulant n machines avec :
      - machine: adresse IP fictive,
      - ports_ouverts: liste aléatoire de ports ouverts,
      - labels: dictionnaire indiquant True/False pour plusieurs services.
    """
    random.seed(seed)
    possible_ports = {
        "Web": [80, 443, 8080, 8443],
        "BaseDeDonnees": [3306, 5432, 27017],
        "Messagerie": [25, 465, 587, 110, 143, 993],
        "Fichier": [21, 22, 445, 139],
        "DNS": [53],
        "Monitoring": [161, 162, 3000, 9090],
        "Proxy": [3128],
        "Odoo": [8069, 8071, 8072],
        "ERPNext": [8000, 8001, 8002],
        "Metabase": [3000, 3001, 3002],
        "Bob50": [6262],
        "HyperPlanning": [21200],
        "GitLab": [80, 443, 22],
    }
    all_possible_ports = {port for plist in possible_ports.values() for port in plist}

    def assign_labels(ports_list):
        return {label: any(p in ports_list for p in plist) for label, plist in possible_ports.items()}

    data = []
    for i in range(n):
        machine_ip = f"192.168.0.{i}"
        num_ports = random.randint(1, min(7, len(all_possible_ports)))
        ports_list = random.sample(list(all_possible_ports), num_ports)
        ports_list.sort()
        labels = assign_labels(ports_list)
        data.append({"machine": machine_ip, "ports_ouverts": ports_list, "labels": labels})
    return pd.DataFrame(data)

def ports_to_features(ports_list, all_ports_sorted) -> list:
    """
    Convertit la liste 'ports_list' en un vecteur binaire basé sur 'all_ports_sorted'.
    """
    return [1 if p in ports_list else 0 for p in all_ports_sorted]

def build_feature_matrix(df: pd.DataFrame, all_ports_sorted: list) -> np.ndarray:
    """
    Transforme la colonne 'ports_ouverts' du DataFrame en une matrice NumPy.
    """
    if df.empty:
        return np.empty((0, len(all_ports_sorted)))
    features_series = df['ports_ouverts'].apply(lambda pl: ports_to_features(pl, all_ports_sorted))
    return np.vstack(features_series.values)

def build_label_matrix(df: pd.DataFrame, label_columns: list):
    """
    Convertit la colonne 'labels' en un tableau NumPy de 0/1.
    """
    df_labels = pd.json_normalize(df['labels'])
    for col in label_columns:
        if col not in df_labels.columns:
            df_labels[col] = False
    df_labels = df_labels[label_columns]
    return df_labels.values.astype(int), label_columns

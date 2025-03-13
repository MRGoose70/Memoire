#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple de script Python combinant :
  - la création d'un grand dataset synthétique (multi-label) pour l'entraînement,
  - l'entraînement d'un Classifier Chain,
  - un scan Nmap sur une machine pour détecter les ports ouverts,
  - la prédiction des labels (processus métiers) en fonction des ports détectés.

Prérequis :
  - Avoir Nmap installé (ex : sudo apt-get install nmap).
  - Installer python-nmap (pip install python-nmap).
  - Installer scikit-learn, pandas, numpy (pip install scikit-learn pandas numpy).
  - Selon le type de scan, exécuter le script en sudo.

Auteurs / Sources :
 - python-nmap : https://pypi.org/project/python-nmap/
 - Scikit-learn Multi-label : https://scikit-learn.org/stable/modules/multiclass.html#multi-label-classification
 - Tsoumakas, G. & Katakis, I. (2007). "Multi-Label Classification: An Overview"
 - Read, J. et al. (2009). "Classifier chains for multi-label classification"
"""

import nmap
import pandas as pd
import numpy as np
import random

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1) DATASET : création d'un grand dataset synthétique (multi-label)
# -----------------------------------------------------------------------------
def create_large_training_dataset(n=2000, seed=42):
    """
    Génère un DataFrame simulant n machines, chacune avec :
      - machine : une adresse IP fictive
      - ports_ouverts : une liste aléatoire de ports ouverts
      - labels : dictionnaire indiquant True/False pour 7 rôles/services
         -> Labels : Web, BaseDeDonnees, Messagerie, Fichier, DNS, Monitoring, Proxy
    La génération est aléatoire à partir d'un dictionnaire de ports caractéristiques.
    Retourne un DataFrame.
    """
    random.seed(seed)
    
    # Dictionnaire des ports caractéristiques pour chaque label
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
        "Metabase": [3000, 3001, 3002]
    }
    
    # Création de l'ensemble de tous les ports possibles
    all_possible_ports = set()
    for plist in possible_ports.values():
        all_possible_ports.update(plist)
    all_possible_ports = list(all_possible_ports)
    
    def assign_labels(ports_list):
        """Assigne True pour un label si au moins un port caractéristique est présent."""
        labels = {}
        for label, plist in possible_ports.items():
            labels[label] = any(p in ports_list for p in plist)
        return labels
    
    data = []
    for i in range(n):
        machine_ip = f"192.168.0.{0 + i}"
        # Choisit aléatoirement entre 1 et min(7, len(all_possible_ports)) ports
        num_ports = random.randint(1, min(7, len(all_possible_ports)))
        ports_list = random.sample(all_possible_ports, num_ports)
        ports_list = sorted(ports_list)
        labels = assign_labels(ports_list)
        data.append({
            "machine": machine_ip,
            "ports_ouverts": ports_list,
            "labels": labels
        })
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 2) FONCTIONS UTILES : transformation des données (ports -> features, labels -> matrice)
# -----------------------------------------------------------------------------
def ports_to_features(ports_list, all_ports_sorted):
    """
    Convertit la liste 'ports_list' en vecteur binaire,
    en se basant sur la liste ordonnée 'all_ports_sorted'.
    """
    return [1 if p in ports_list else 0 for p in all_ports_sorted]

def build_feature_matrix(df, all_ports_sorted):
    """
    Transforme la colonne 'ports_ouverts' d'un DataFrame en une matrice NumPy.
    Chaque machine est représentée par un vecteur binaire indiquant l'ouverture de chaque port.
    """
    if df.empty:
        return np.empty((0, len(all_ports_sorted)))
    features_series = df['ports_ouverts'].apply(lambda pl: ports_to_features(pl, all_ports_sorted))
    X = np.vstack(features_series.values)
    return X

def build_label_matrix(df, label_columns):
    """
    Convertit la colonne 'labels' (dictionnaire) d'un DataFrame en un tableau NumPy de 0/1,
    avec les colonnes dans l'ordre défini par label_columns.
    """
    df_labels = pd.json_normalize(df['labels'])
    for col in label_columns:
        if col not in df_labels.columns:
            df_labels[col] = False
    df_labels = df_labels[label_columns]
    return df_labels.values.astype(int), label_columns

def train_classifier_chains(X_train, y_train):
    """
    Entraîne un modèle ClassifierChain à partir de X_train et y_train.
    Retourne le modèle entraîné.
    """
    base_estimator = LogisticRegression()
    model = ClassifierChain(base_estimator=base_estimator, order='random', random_state=42)
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------------------------------
# 3) NMAP : Fonctions pour lancer un scan et parser le résultat
# -----------------------------------------------------------------------------
def launch_nmap_scan(ip_address, port_range, options):
    """
    Lance un scan Nmap sur ip_address, pour la plage de ports port_range,
    en utilisant les options fournies.
    Retourne le dictionnaire renvoyé par python-nmap.
    
    Exemple :
       nm.scan(
         hosts='192.168.0.80',
         arguments='-sV -Pn -p 1-4096'
       )
    """
    nm = nmap.PortScanner()
    cmd_args = f"{options} -p {port_range}"
    print(f"[INFO] Lancement de nmap sur {ip_address} avec arguments: '{cmd_args}'")
    scan_result = nm.scan(hosts=ip_address, arguments=cmd_args)
    return scan_result

def parse_scan_result(scan_result):
    """
    Parcourt le résultat du scan Nmap et retourne un DataFrame avec :
      - machine (str)
      - ports_ouverts (list[int])
    """
    hosts_data = []
    for host in scan_result.get('scan', {}):
        host_status = scan_result['scan'][host]['status']['state']
        if host_status == 'up':
            ports_ouverts = []
            tcp_data = scan_result['scan'][host].get('tcp', {})
            for port, pdata in tcp_data.items():
                if pdata['state'] == 'open':
                    ports_ouverts.append(port)
            hosts_data.append({
                'machine': host,
                'ports_ouverts': ports_ouverts
            })
    df_scan = pd.DataFrame(hosts_data)
    return df_scan

# -----------------------------------------------------------------------------
# 4) MAIN : Enchaînement complet
# -----------------------------------------------------------------------------
def main():
    # (A) Création du dataset de training (grande échelle)
    df_train = create_large_training_dataset(n=200, seed=42)
    print("[INFO] Dataset d'entraînement généré :")
    print(df_train)

    # On récupère la liste de tous les ports présents dans le dataset
    all_ports_train = set()
    for ports_list in df_train['ports_ouverts']:
        all_ports_train.update(ports_list)
    all_ports_train = sorted(list(all_ports_train))
    
    print("\n[INFO] Ports possibles dans le training :", all_ports_train)

    # Construction de la matrice de features X_train
    X_train = build_feature_matrix(df_train, all_ports_train)

    # Définition des labels à prédire
    label_cols = ["Web", "BaseDeDonnees", "Messagerie", "Fichier", "DNS", "Monitoring", "Proxy", "Odoo", "ERPNext", "Metabase"]
    y_train, label_cols = build_label_matrix(df_train, label_cols)
    print(f"\n[INFO] X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")

    # (B) Entraînement du modèle Classifier Chain
    model_cc = train_classifier_chains(X_train, y_train)
    print("[INFO] Modèle Classifier Chain entraîné !")

    # (C) Lancer un scan Nmap sur la cible
    TARGET_IP = input("IP to scan : ")  # Par exemple, 192.168.0.80
    PORT_RANGE = input("Port range (e.g., 1-65535) : ")
    SCAN_OPTIONS = input("Scan options (e.g., -sV -Pn -sT) : ")
    scan_result = launch_nmap_scan(TARGET_IP, PORT_RANGE, SCAN_OPTIONS)

    # (D) Parser le résultat du scan
    df_scan = parse_scan_result(scan_result)
    if df_scan.empty:
        print("[WARNING] Aucune machine détectée 'up' ou aucun port ouvert !")
        print("[HINT] Vérifie l'IP, ajoute -Pn, exécute en sudo, etc.")
        return
    
    print("\n[INFO] Résultat du scan :")
    print(df_scan)

    # (E) Construire les features à partir des ports scannés
    # (Utilisation de la même liste 'all_ports_train' que pour l'entraînement)
    X_new = build_feature_matrix(df_scan, all_ports_train)
    print(f"[INFO] X_new shape = {X_new.shape} pour {len(df_scan)} machine(s) scannée(s).")

    # (F) Prédiction
    y_pred = model_cc.predict(X_new)

    # (G) Affichage des prédictions
    for i, row in df_scan.iterrows():
        machine = row['machine']
        pred_vector = y_pred[i]
        labels_dict = {label_cols[j]: bool(pred_vector[j]) for j in range(len(label_cols))}
        print(f"\n=== Machine: {machine} ===")
        print(f"Ports ouverts: {row['ports_ouverts']}")
        print(f"Labels prédits: {labels_dict}")

# -----------------------------------------------------------------------------
# Point d'entrée du script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

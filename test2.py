#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple de script Python combinant :
  - la création d'un dataset plus riche (multi-label) pour l'entraînement,
  - l'entraînement d'un Classifier Chain,
  - un scan Nmap sur une machine pour détecter les ports ouverts,
  - la prédiction des labels (processus métiers) en fonction des ports détectés.

Prérequis :
  - Avoir Nmap installé (apt-get install nmap).
  - Avoir python-nmap (pip install python-nmap).
  - Avoir scikit-learn, pandas, numpy (pip install scikit-learn pandas numpy).
  - Selon le type de scan, exécuter le script en sudo.

Auteurs / Sources :
 - python-nmap : https://pypi.org/project/python-nmap/
 - Multi-label scikit-learn : https://scikit-learn.org/stable/modules/multiclass.html#multi-label-classification
 - Tsoumakas, G. & Katakis, I. (2007). "Multi-Label Classification: An Overview"
 - Read, J. et al. (2009). "Classifier chains for multi-label classification"
"""

import nmap
import pandas as pd
import numpy as np

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1) DATASET : création d'un dataset plus riche (multi-label)
# -----------------------------------------------------------------------------
def create_rich_training_dataset():
    """
    Génère un DataFrame simulant ~15 machines, chacune avec :
      - machine (str)
      - ports_ouverts (list[int])
      - labels (dict[str, bool]) 
        -> 7 labels potentiels: Web, BaseDeDonnees, Messagerie,
                               Fichier, DNS, Monitoring, Proxy

    Retourne un DataFrame.
    """
    train_data = [
        {
            'machine': '192.168.0.101',
            'ports_ouverts': [80, 443],
            'labels': {
                'Web': True,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.102',
            'ports_ouverts': [3306, 22],
            'labels': {
                'Web': False,
                'BaseDeDonnees': True,
                'Messagerie': False,
                'Fichier': True,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.103',
            'ports_ouverts': [445, 139],
            'labels': {
                'Web': False,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': True,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.104',
            'ports_ouverts': [25, 110, 143],
            'labels': {
                'Web': False,
                'BaseDeDonnees': False,
                'Messagerie': True,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.105',
            'ports_ouverts': [80, 443, 3306],
            'labels': {
                'Web': True,
                'BaseDeDonnees': True,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.106',
            'ports_ouverts': [80, 25, 445, 139],
            'labels': {
                'Web': True,
                'BaseDeDonnees': False,
                'Messagerie': True,
                'Fichier': True,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.107',
            'ports_ouverts': [53],
            'labels': {
                'Web': False,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': False,
                'DNS': True,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.108',
            'ports_ouverts': [8080, 3128],
            'labels': {
                'Web': True,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': True
            }
        },
        {
            'machine': '192.168.0.109',
            'ports_ouverts': [161, 162],
            'labels': {
                'Web': False,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': True,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.110',
            'ports_ouverts': [3000, 22],
            'labels': {
                'Web': False,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': True,
                'DNS': False,
                'Monitoring': True,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.111',
            'ports_ouverts': [9090, 5432],
            'labels': {
                'Web': False,
                'BaseDeDonnees': True,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': True,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.112',
            'ports_ouverts': [27017, 80],
            'labels': {
                'Web': True,
                'BaseDeDonnees': True,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.113',
            'ports_ouverts': [8443, 53],
            'labels': {
                'Web': True,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': False,
                'DNS': True,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.114',
            'ports_ouverts': [143, 993, 587],
            'labels': {
                'Web': False,
                'BaseDeDonnees': False,
                'Messagerie': True,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': False
            }
        },
        {
            'machine': '192.168.0.115',
            'ports_ouverts': [8080, 8081, 3128, 443],
            'labels': {
                'Web': True,
                'BaseDeDonnees': False,
                'Messagerie': False,
                'Fichier': False,
                'DNS': False,
                'Monitoring': False,
                'Proxy': True
            }
        },
    ]

    df = pd.DataFrame(train_data)
    return df

# -----------------------------------------------------------------------------
# 2) FONCTIONS UTILES : transformation ports -> features, labels -> y, etc.
# -----------------------------------------------------------------------------
def ports_to_features(ports_list, all_ports_sorted):
    """
    Convertit la liste 'ports_list' en vecteur binaire, 
    en se basant sur la liste ordonnée 'all_ports_sorted'.
    """
    return [1 if p in ports_list else 0 for p in all_ports_sorted]

def build_feature_matrix(df, all_ports_sorted):
    """
    Applique ports_to_features à la colonne 'ports_ouverts' d'un DataFrame df.
    Retourne une matrice NumPy de dimension (nb_machines, nb_ports).
    """
    if df.empty:
        return np.empty((0, len(all_ports_sorted)))
    features_series = df['ports_ouverts'].apply(
        lambda pl: ports_to_features(pl, all_ports_sorted)
    )
    X = np.vstack(features_series.values)
    return X

def build_label_matrix(df, label_columns):
    """
    Convertit la colonne 'labels' (dictionnaire) d'un DataFrame 
    en colonnes distinctes, selon label_columns (liste d'étiquettes).
    Retourne (y, label_columns) où y est un numpy array 0/1.
    """
    df_labels = pd.json_normalize(df['labels'])
    for col in label_columns:
        if col not in df_labels.columns:
            df_labels[col] = False
    df_labels = df_labels[label_columns]
    return df_labels.values.astype(int), label_columns

def train_classifier_chains(X_train, y_train):
    """
    Entraîne un modèle ClassifierChain à partir de X_train, y_train.
    Retourne le modèle.
    """
    base_estimator = LogisticRegression()
    model = ClassifierChain(
        base_estimator=base_estimator, 
        order='random', 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------------------------------
# 3) NMAP : Fonctions pour lancer un scan et parser le résultat
# -----------------------------------------------------------------------------
def launch_nmap_scan(ip_address, port_range, options):
    """
    Lance un scan Nmap sur ip_address, pour la plage de ports port_range,
    en utilisant la chaîne 'options'.
    Retourne le dict de python-nmap.
    
    Ex:
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
    Parcourt le résultat python-nmap et retourne un DataFrame 
    avec colonnes:
      - machine
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
    # (A) Création / chargement du dataset de training
    df_train = create_rich_training_dataset()
    print("[INFO] Dataset d'entraînement créé :")
    print(df_train)

    # On liste tous les ports vus dans ce dataset
    all_ports_train = set()
    for ports_l in df_train['ports_ouverts']:
        all_ports_train.update(ports_l)
    all_ports_train = sorted(list(all_ports_train))
    
    print("\n[INFO] Ports possibles dans le training :", all_ports_train)

    # On construit X_train
    X_train = build_feature_matrix(df_train, all_ports_train)

    # On définit les labels qu'on veut prédire
    label_cols = ["Web", "BaseDeDonnees", "Messagerie", "Fichier", "DNS", "Monitoring", "Proxy"]
    y_train, label_cols = build_label_matrix(df_train, label_cols)

    print(f"[INFO] X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")

    # (B) Entraînement du modèle ClassifierChain
    model_cc = train_classifier_chains(X_train, y_train)
    print("[INFO] Modèle Classifier Chain entraîné !")

    # (C) Lance un scan Nmap sur la cible
    TARGET_IP = input("IP to scan : ")  # A adapter
    PORT_RANGE = input("Port range Ex : 1-65535 : ")
    SCAN_OPTIONS = input("Options Ex : -sV : ")    # -Pn pour ignorer ping, -sV pour détection de version
    scan_result = launch_nmap_scan(TARGET_IP, PORT_RANGE, SCAN_OPTIONS)

    # (D) Parse le résultat du scan
    df_scan = parse_scan_result(scan_result)
    if df_scan.empty:
        print("[WARNING] Aucune machine détectée 'up' ou aucun port ouvert !")
        print("[HINT] Vérifie l'IP, ajoute -Pn, exécute en sudo, etc.")
        return
    
    print("\n[INFO] Résultat du scan :")
    print(df_scan)

    # (E) On construit les features à partir des ports scannés
    #     (Important : on réutilise la même liste 'all_ports_train' qu'à l'entraînement)
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script amélioré pour :
  - Générer un dataset multi-label de grande taille (n=2000)
  - Entraîner un modèle Classifier Chain
  - Évaluer le modèle par validation croisée (5-fold) et via un split train/test
  - Lancer un scan Nmap et prédire les labels sur de nouvelles machines

Sources :
 - Code initial fourni dans LabelGuard.py :contentReference[oaicite:0]{index=0}
 - Documentation scikit-learn : https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import nmap
import pandas as pd
import numpy as np
import random

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1) Création du dataset synthétique multi-label
# -----------------------------------------------------------------------------
def create_large_training_dataset(n=2000, seed=42):
    """
    Génère un DataFrame simulant n machines avec :
      - machine : une adresse IP fictive,
      - ports_ouverts : une liste aléatoire de ports ouverts,
      - labels : dictionnaire indiquant True/False pour plusieurs services.
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
        "Metabase": [3000, 3001, 3002],
        "Bob50": [6262],
        "HyperPlanning": [21200]
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
        machine_ip = f"192.168.0.{i}"
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
# 2) Fonctions utilitaires pour la transformation des données
# -----------------------------------------------------------------------------
def ports_to_features(ports_list, all_ports_sorted):
    """
    Convertit la liste 'ports_list' en vecteur binaire,
    basé sur la liste ordonnée 'all_ports_sorted'.
    """
    return [1 if p in ports_list else 0 for p in all_ports_sorted]

def build_feature_matrix(df, all_ports_sorted):
    """
    Transforme la colonne 'ports_ouverts' en une matrice NumPy.
    Chaque machine est représentée par un vecteur binaire indiquant l'ouverture de chaque port.
    """
    if df.empty:
        return np.empty((0, len(all_ports_sorted)))
    features_series = df['ports_ouverts'].apply(lambda pl: ports_to_features(pl, all_ports_sorted))
    X = np.vstack(features_series.values)
    return X

def build_label_matrix(df, label_columns):
    """
    Convertit la colonne 'labels' (dictionnaire) en un tableau NumPy de 0/1,
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
    base_estimator = LogisticRegression(max_iter=1000)
    model = ClassifierChain(base_estimator=base_estimator, order='random', random_state=42)
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------------------------------
# 3) Fonctions Nmap pour lancer un scan et parser le résultat
# -----------------------------------------------------------------------------
def launch_nmap_scan(ip_address, port_range, options):
    """
    Lance un scan Nmap sur ip_address pour la plage de ports port_range,
    avec les options fournies.
    Retourne le dictionnaire de résultats.
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
    return pd.DataFrame(hosts_data)

# -----------------------------------------------------------------------------
# 4) Main : Enchaînement complet avec validation croisée et scan Nmap
# -----------------------------------------------------------------------------
def main():
    # (A) Création du dataset avec n=2000 échantillons
    df_train = create_large_training_dataset(n=2000, seed=42)
    print(f"[INFO] Dataset d'entraînement généré (taille = {df_train.shape[0]})")
    
    # Récupération de tous les ports présents dans le dataset
    all_ports_train = set()
    for ports_list in df_train['ports_ouverts']:
        all_ports_train.update(ports_list)
    all_ports_train = sorted(list(all_ports_train))
    print("\n[INFO] Ports possibles dans le dataset :", all_ports_train)
    
    # Construction de la matrice des features et des labels
    label_cols = ["Web", "BaseDeDonnees", "Messagerie", "Fichier", "DNS", "Monitoring",
                  "Proxy", "Odoo", "ERPNext", "Metabase", "Bob50", "HyperPlanning"]
    X = build_feature_matrix(df_train, all_ports_train)
    y, label_cols = build_label_matrix(df_train, label_cols)
    print(f"\n[INFO] X shape = {X.shape}, y shape = {y.shape}")

    # -----------------------------------------------------------------------------
    # Évaluation par validation croisée (5-fold)
    # -----------------------------------------------------------------------------
    model_cc = ClassifierChain(base_estimator=LogisticRegression(max_iter=1000), 
                                 order='random', random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_cc, X, y, cv=kf, scoring='accuracy')
    print(f"\n[INFO] Accuracy moyenne en validation croisée (5-fold) : {cv_scores.mean():.4f}")

    # Optionnel : séparation train/test pour afficher d'autres métriques
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_cc = train_classifier_chains(X_train, y_train)
    y_pred = model_cc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy sur jeu de test (20%) : {acc:.4f}")

    # Affichage des matrices de confusion par label
    print("\n[INFO] Matrices de confusion par label :")
    for idx, label in enumerate(label_cols):
        cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
        if cm.size == 1:  # Cas où une seule classe est présente
            print(f"{label} : {cm[0, 0]} (valeur unique)")
        else:
            tn, fp, fn, tp = cm.ravel()
            print(f"{label} : TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # -----------------------------------------------------------------------------
    # Partie scan Nmap et prédiction sur de nouvelles machines
    # -----------------------------------------------------------------------------
    TARGET_IP = input("IP à scanner : ")  # Ex : 192.168.0.183
    PORT_RANGE = input("Plage de ports (ex: 1-65535) : ")
    SCAN_OPTIONS = input("Options de scan (ex: -sV -Pn -sT) : ")
    scan_result = launch_nmap_scan(TARGET_IP, PORT_RANGE, SCAN_OPTIONS)

    df_scan = parse_scan_result(scan_result)
    if df_scan.empty:
        print("[WARNING] Aucune machine détectée 'up' ou aucun port ouvert !")
        print("[HINT] Vérifiez l'IP, ajoutez -Pn, exécutez en sudo, etc.")
        return

    print("\n[INFO] Résultat du scan :")
    print(df_scan)

    X_new = build_feature_matrix(df_scan, all_ports_train)
    print(f"[INFO] X_new shape = {X_new.shape} pour {len(df_scan)} machine(s) scannée(s).")

    y_pred_new = model_cc.predict(X_new)
    for i, row in df_scan.iterrows():
        machine = row['machine']
        pred_vector = y_pred_new[i]
        labels_dict = {label_cols[j]: bool(pred_vector[j]) for j in range(len(label_cols))}
        print(f"\n=== Machine: {machine} ===")
        print(f"Ports ouverts: {row['ports_ouverts']}")
        print(f"Labels prédits: {labels_dict}")

if __name__ == "__main__":
    main()

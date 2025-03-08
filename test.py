#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple de script Python utilisant Nmap pour scanner une machine,
puis exploitant un modèle Classifier Chains afin de déterminer les
processus métiers potentiels (multi-label).

Auteurs / Sources:
- python-nmap Docs: https://pypi.org/project/python-nmap/
- Scikit-learn Multi-label: https://scikit-learn.org/stable/modules/multiclass.html#multi-label-classification
- Tsoumakas, G. & Katakis, I. (2007). "Multi-Label Classification: An Overview"
- Read, J. et al. (2009). "Classifier chains for multi-label classification"

Note: Cet exemple est fictif. En pratique, on disposerait d'un dataset plus large
pour entraîner le modèle de manière plus robuste.
"""

import nmap
import pandas as pd
import numpy as np

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1) Variables et configuration
# -----------------------------------------------------------------------------
TARGET_IP = input("IP to scan : ")   # IP (ou hostname) de la cible à scanner
PORT_RANGE = input("Port range to scan Ex : 1-655365 : ")        # Plage de ports à scanner
SCAN_OPTIONS = input("Options Ex : -sV : ")     # Paramètres Nmap: -sV pour détection version

# -----------------------------------------------------------------------------
# 2) Lancement d'un scan Nmap via python-nmap
# -----------------------------------------------------------------------------
def launch_nmap_scan(ip_address, port_range, options):
    """
    Lance un scan Nmap sur l'adresse ip_address, pour la plage de ports port_range,
    avec les options 'options'. Retourne l'objet dict renvoyé par python-nmap.
    """
    nm = nmap.PortScanner()

    # -- Méthode recommandée : utiliser 'arguments' et éventuellement 'ports'
    # On combine options et -p <port_range>
    cmd_args = f"{options} -p {port_range}"
    
    print(f"[INFO] Lancement de nmap sur {ip_address} avec arguments: {cmd_args}")
    # Note: si tu souhaites vraiment spécifier `ports=port_range` séparément,
    #       fais nm.scan(hosts=ip_address, ports=port_range, arguments=options)
    #       Mais quand il y a '-Pn' ou d'autres flags, c'est souvent plus sûr
    #       de tout regrouper dans 'arguments'.
    
    scan_result = nm.scan(hosts=ip_address, arguments=cmd_args)
    return scan_result

# -----------------------------------------------------------------------------
# 3) Extraction des ports ouverts à partir du résultat Nmap
# -----------------------------------------------------------------------------
def parse_scan_result(scan_result):
    """
    Parcourt scan_result et retourne un DataFrame avec:
        - machine (str)
        - ports_ouverts (list d'int)
    pour chaque machine 'up' scannée.
    """
    hosts_data = []
    # 'scan' est un dict contenant chaque host scanné
    for host in scan_result.get('scan', {}):
        host_status = scan_result['scan'][host]['status']['state']
        if host_status == 'up':
            ports_ouverts = []
            # On récupère les infos sur le protocole TCP
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
# 4) Exemple de dataset d'entraînement fictif (multi-label)
# -----------------------------------------------------------------------------
def create_training_dataset():
    """
    Simule un petit dataset contenant:
        - machine : ip fictive
        - ports_ouverts : liste de ports
        - labels : dict indiquant Web, BaseDeDonnees, Messagerie, Fichier (bool)
    Retourne un DataFrame.
    """
    train_data = [
        {
            'machine': '192.168.0.101',
            'ports_ouverts': [80, 443],
            'labels': {'Web': True, 'BaseDeDonnees': False, 'Messagerie': False, 'Fichier': False}
        },
        {
            'machine': '192.168.0.102',
            'ports_ouverts': [3306, 22],
            'labels': {'Web': False, 'BaseDeDonnees': True, 'Messagerie': False, 'Fichier': False}
        },
        {
            'machine': '192.168.0.103',
            'ports_ouverts': [445],
            'labels': {'Web': False, 'BaseDeDonnees': False, 'Messagerie': False, 'Fichier': True}
        },
        {
            'machine': '192.168.0.104',
            'ports_ouverts': [25, 110],
            'labels': {'Web': False, 'BaseDeDonnees': False, 'Messagerie': True, 'Fichier': False}
        },
        {
            'machine': '192.168.0.105',
            'ports_ouverts': [80, 443, 3306],
            'labels': {'Web': True, 'BaseDeDonnees': True, 'Messagerie': False, 'Fichier': False}
        },
        {
            'machine': '192.168.0.106',
            'ports_ouverts': [80, 25, 445],
            'labels': {'Web': True, 'BaseDeDonnees': False, 'Messagerie': True, 'Fichier': True}
        }
    ]

    df_train = pd.DataFrame(train_data)
    return df_train

# -----------------------------------------------------------------------------
# 5) Transformation (ports -> vecteurs) et (labels -> vecteurs)
# -----------------------------------------------------------------------------
def ports_to_features(ports_list, all_ports_sorted):
    """
    Convertit la liste de ports 'ports_list' en un vecteur binaire,
    en se basant sur la liste triée all_ports_sorted.
    """
    return [1 if p in ports_list else 0 for p in all_ports_sorted]

def build_feature_matrix(df, all_ports_sorted):
    """
    Applique ports_to_features à toutes les lignes d'un DataFrame df
    qui doit contenir la colonne 'ports_ouverts'.
    Retourne une matrice NumPy.
    """
    if df.empty:
        return np.empty((0, len(all_ports_sorted)))

    features = df['ports_ouverts'].apply(lambda pl: ports_to_features(pl, all_ports_sorted))
    X = np.vstack(features.values)
    return X

def build_label_matrix(df, label_columns=None):
    """
    Convertit la colonne 'labels' (dictionnaire) d'un DataFrame en un
    DataFrame de colonnes séparées (si label_columns est fourni, on garantit l'ordre).
    Retourne la matrice NumPy de 0/1.
    """
    df_labels = pd.json_normalize(df['labels'])
    # S'il n'y a pas de label_columns imposé, on prend l'ordre par défaut
    if label_columns is None:
        label_columns = df_labels.columns.tolist()

    # Si certaines colonnes manquent, on les crée (remplies à False)
    for col in label_columns:
        if col not in df_labels.columns:
            df_labels[col] = False

    # On s'assure de l'ordre des colonnes
    df_labels = df_labels[label_columns]
    return df_labels.values.astype(int), label_columns

# -----------------------------------------------------------------------------
# 6) Entraîner un modèle Classifier Chains
# -----------------------------------------------------------------------------
def train_classifier_chains(X_train, y_train):
    """
    Entraîne un modèle ClassifierChain à partir des features X_train et
    des labels y_train (matrices NumPy).
    Retourne le modèle entraîné.
    """
    base_estimator = LogisticRegression()
    model = ClassifierChain(base_estimator=base_estimator, order='random', random_state=42)
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------------------------------
# 7) Programme principal
# -----------------------------------------------------------------------------
def main():
    # (A) Lancement du scan Nmap sur la cible
    scan_result = launch_nmap_scan(TARGET_IP, PORT_RANGE, SCAN_OPTIONS)
    
    # Petit debug: afficher le dict brut
    # print(scan_result)  # Décommente si besoin
    
    df_scan = parse_scan_result(scan_result)

    if df_scan.empty:
        print("[WARNING] Aucune machine 'up' détectée ou aucun port ouvert.")
        print("[HINT] Essaie d'ajouter '-Pn' dans SCAN_OPTIONS, ou exécuter sous sudo, ou vérifier IP.")
        return

    print("[INFO] DataFrame des machines scannées:")
    print(df_scan)

    # (B) Création du dataset d'entraînement fictif
    df_train = create_training_dataset()

    # On récupère la liste de tous les ports rencontrés dans df_train
    all_ports_train = set()
    for ports_l in df_train['ports_ouverts']:
        all_ports_train.update(ports_l)
    # On les trie pour garder un ordre cohérent
    all_ports_train = sorted(list(all_ports_train))

    # (C) Transformation en vecteurs (X_train, y_train)
    X_train = build_feature_matrix(df_train, all_ports_train)

    # On suppose qu'on a 4 labels: Web, BaseDeDonnees, Messagerie, Fichier (dans cet ordre)
    label_cols = ["Web", "BaseDeDonnees", "Messagerie", "Fichier"]
    y_train, label_cols = build_label_matrix(df_train, label_cols)

    # (D) Entraînement du modèle Classifier Chains
    model_cc = train_classifier_chains(X_train, y_train)
    print("[INFO] Modèle Classifier Chains entraîné.")

    # (E) Application du modèle à la machine scannée
    # On construit le vecteur X_new en se basant sur all_ports_train
    X_new = build_feature_matrix(df_scan, all_ports_train)

    y_pred = model_cc.predict(X_new)  # Matrice de prédiction

    # Affichage des résultats
    for i, row in df_scan.iterrows():
        machine = row['machine']
        pred_vector = y_pred[i]
        labels_dict = {label_cols[j]: bool(pred_vector[j]) for j in range(len(label_cols))}
        print(f"\n[RESULTAT] Machine: {machine}")
        print(f"Ports ouverts: {row['ports_ouverts']}")
        print(f"Labels prédits: {labels_dict}")

if __name__ == "__main__":
    main()

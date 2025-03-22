#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script amélioré pour :
  - Générer un dataset multi-label de machines et assigner un besoin métier en fonction des labels détectés.
  - Entraîner un modèle Classifier Chain sur le dataset.
  - Évaluer le modèle via validation croisée (5-fold) et via un split train/test.
  - Lancer un scan Nmap classique pour prédire les labels et assigner les besoins métiers.
  - Lancer un scan de sécurité Nmap utilisant un script NSE après l'assignation des besoins métiers,
    en utilisant un scan TCP connect (-sT) avec l'option -Pn pour désactiver la découverte d'hôte.
  - Afficher les résultats de manière jolie et lisible.

Sources :
 - Code initial fourni dans LabelGuard.py :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
 - Documentation sur la gestion des exceptions : https://docs.python.org/fr/3/tutorial/errors.html
 - PEP8 – Style Guide for Python Code : https://www.python.org/dev/peps/pep-0008/
 - Documentation Nmap NSE : https://nmap.org/book/nse.html
"""

import re
import nmap
import pandas as pd
import numpy as np
import random
import json

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------------------------------------------
# Fonctions utilitaires d'affichage et de validation
# ----------------------------------------------------------------------------
def pretty_print_dict(d, title=""):
    """
    Affiche le dictionnaire 'd' de façon lisible en utilisant un format JSON indenté.
    Optionnellement, affiche un titre au-dessus.
    """
    if title:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
    print(json.dumps(d, indent=2, ensure_ascii=False))


def validate_ip(ip):
    """
    Valide qu'une chaîne de caractères correspond bien à une adresse IPv4.
    """
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(pattern, ip):
        return all(0 <= int(octet) <= 255 for octet in ip.split('.'))
    return False


def get_ip_input(prompt):
    """
    Demande à l'utilisateur une adresse IP jusqu'à obtenir une valeur valide.
    """
    while True:
        ip = input(prompt)
        if validate_ip(ip):
            return ip
        else:
            print("[ERROR] Format d'IP invalide. Veuillez réessayer.")


def validate_port_range(port_range):
    """
    Valide que la plage de ports est au format 'start-end' avec des valeurs entre 1 et 65535.
    """
    pattern = r'^\d{1,5}-\d{1,5}$'
    if re.match(pattern, port_range):
        start, end = map(int, port_range.split('-'))
        if 1 <= start <= end <= 65535:
            return True
    return False


def get_port_range_input(prompt):
    """
    Demande à l'utilisateur une plage de ports jusqu'à obtenir une valeur valide.
    """
    while True:
        port_range = input(prompt)
        if validate_port_range(port_range):
            return port_range
        else:
            print("[ERROR] Format de plage de ports invalide. Utilisez le format 'start-end' avec des valeurs entre 1 et 65535.")


def get_scan_options_input(prompt):
    """
    Demande à l'utilisateur les options de scan jusqu'à obtenir une chaîne non vide.
    """
    while True:
        options = input(prompt)
        if options.strip():
            return options
        else:
            print("[ERROR] Les options de scan ne peuvent pas être vides.")


def get_nse_script_input(prompt):
    """
    Demande à l'utilisateur le nom du script NSE jusqu'à obtenir une chaîne non vide.
    """
    while True:
        script = input(prompt)
        if script.strip():
            return script
        else:
            print("[ERROR] Le nom du script NSE ne peut pas être vide.")


# ----------------------------------------------------------------------------
# Mapping entre labels et besoins métiers
# ----------------------------------------------------------------------------
label_to_need = {
    "Web": "Hébergement Web",
    "BaseDeDonnees": "Gestion de données",
    "Messagerie": "Service de messagerie",
    "Fichier": "Stockage de fichiers",
    "DNS": "Gestion DNS",
    "Monitoring": "Surveillance de l'infrastructure",
    "Proxy": "Sécurité et proxy",
    "Odoo": "ERP / Gestion d'entreprise",
    "ERPNext": "ERP / Gestion d'entreprise",
    "Metabase": "Business Intelligence",
    "Bob50": "Application métier Bob50",
    "HyperPlanning": "Gestion de planning"
}

def assign_business_need(labels: dict) -> list:
    """
    Assigne un ou plusieurs besoins métiers en fonction des labels fournis.
    Pour chaque label à True, retourne le besoin métier correspondant.
    Si aucun label n'est détecté, un besoin par défaut est assigné.
    """
    needs = []
    for label, is_present in labels.items():
        if is_present:
            need = label_to_need.get(label)
            if need and need not in needs:
                needs.append(need)
    if not needs:
        needs.append("Aucun besoin métier assigné")
    return needs

def add_business_need_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'besoin_metier' au DataFrame, basée sur la colonne 'labels'.
    """
    df = df.copy()
    df["besoin_metier"] = df["labels"].apply(assign_business_need)
    return df

# ----------------------------------------------------------------------------
# 1) Création du dataset synthétique multi-label
# ----------------------------------------------------------------------------
def create_large_training_dataset(n=2000, seed=42):
    """
    Génère un DataFrame simulant n machines avec :
      - machine : une adresse IP fictive,
      - ports_ouverts : une liste aléatoire de ports ouverts,
      - labels : dictionnaire indiquant True/False pour plusieurs services.
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
        "HyperPlanning": [21200]
    }
    all_possible_ports = set()
    for plist in possible_ports.values():
        all_possible_ports.update(plist)
    all_possible_ports = list(all_possible_ports)

    def assign_labels(ports_list):
        labels = {}
        for label, plist in possible_ports.items():
            labels[label] = any(p in ports_list for p in plist)
        return labels

    data = []
    for i in range(n):
        machine_ip = f"192.168.0.{i}"
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

# ----------------------------------------------------------------------------
# 2) Fonctions utilitaires pour la transformation des données
# ----------------------------------------------------------------------------
def ports_to_features(ports_list, all_ports_sorted):
    """
    Convertit la liste 'ports_list' en vecteur binaire basé sur 'all_ports_sorted'.
    """
    return [1 if p in ports_list else 0 for p in all_ports_sorted]

def build_feature_matrix(df, all_ports_sorted):
    """
    Transforme la colonne 'ports_ouverts' en une matrice NumPy.
    """
    if df.empty:
        return np.empty((0, len(all_ports_sorted)))
    features_series = df['ports_ouverts'].apply(lambda pl: ports_to_features(pl, all_ports_sorted))
    X = np.vstack(features_series.values)
    return X

def build_label_matrix(df, label_columns):
    """
    Convertit la colonne 'labels' en un tableau NumPy de 0/1.
    """
    df_labels = pd.json_normalize(df['labels'])
    for col in label_columns:
        if col not in df_labels.columns:
            df_labels[col] = False
    df_labels = df_labels[label_columns]
    return df_labels.values.astype(int), label_columns

def train_classifier_chains(X_train, y_train):
    """
    Entraîne un modèle ClassifierChain avec LogisticRegression.
    """
    base_estimator = LogisticRegression(max_iter=1000)
    model = ClassifierChain(base_estimator=base_estimator, order='random', random_state=42)
    model.fit(X_train, y_train)
    return model

# ----------------------------------------------------------------------------
# 3) Fonctions Nmap pour lancer un scan et parser le résultat
# ----------------------------------------------------------------------------
def launch_nmap_scan(ip_address, port_range, options):
    """
    Lance un scan Nmap sur l'IP et la plage de ports spécifiée.
    Gère les exceptions pour éviter tout crash.
    """
    try:
        nm = nmap.PortScanner()
        cmd_args = f"{options} -p {port_range}"
        print(f"\n[INFO] Lancement du scan Nmap sur {ip_address} avec : '{cmd_args}'")
        scan_result = nm.scan(hosts=ip_address, arguments=cmd_args)
        return scan_result
    except Exception as e:
        print(f"[ERROR] Erreur lors du scan Nmap : {e}")
        return {}  # Retourne un résultat vide en cas d'erreur


def parse_scan_result(scan_result):
    """
    Parse le résultat du scan Nmap et retourne un DataFrame.
    Vérifie l'existence des clés attendues pour éviter des KeyError.
    """
    hosts_data = []
    scan_data = scan_result.get('scan', {})
    for host, host_info in scan_data.items():
        status = host_info.get('status', {}).get('state', 'down')
        if status == 'up':
            ports_ouverts = []
            tcp_data = host_info.get('tcp', {})
            for port, pdata in tcp_data.items():
                if pdata.get('state') == 'open':
                    ports_ouverts.append(port)
            hosts_data.append({
                'machine': host,
                'ports_ouverts': ports_ouverts
            })
    return pd.DataFrame(hosts_data)

def launch_nmap_security_scan(ip_address, port_range, nse_script):
    """
    Lance un scan de sécurité Nmap en utilisant un script NSE.
    Utilise un scan TCP connect (-sT) avec -Pn pour désactiver la découverte d'hôte.
    Gère les exceptions pour éviter tout crash.
    """
    try:
        nm = nmap.PortScanner()
        cmd_args = f"-sT -sV -Pn --script={nse_script} -p {port_range}"
        print(f"\n[INFO] Lancement du scan de sécurité sur {ip_address} avec : '{cmd_args}'")
        scan_result = nm.scan(hosts=ip_address, arguments=cmd_args)
        return scan_result
    except Exception as e:
        print(f"[ERROR] Erreur lors du scan de sécurité Nmap : {e}")
        return {}  # Retourne un résultat vide en cas d'erreur


# ----------------------------------------------------------------------------
# 4) Main : Enchaînement complet avec validation croisée, scan Nmap et affichage soigné
# ----------------------------------------------------------------------------
def main():
    try:
        # Création du dataset
        df_train = create_large_training_dataset(n=2000, seed=42)
        print(f"\n[INFO] Dataset d'entraînement généré (taille = {df_train.shape[0]})")
    
        # Ajout de la colonne 'besoin_metier'
        df_train = add_business_need_column(df_train)
    
        # Récupération des ports présents dans le dataset
        all_ports_train = set()
        for ports_list in df_train['ports_ouverts']:
            all_ports_train.update(ports_list)
        all_ports_train = sorted(list(all_ports_train))
        print("\n[INFO] Ports possibles dans le dataset :", all_ports_train)
    
        # Construction des matrices de features et labels
        label_cols = ["Web", "BaseDeDonnees", "Messagerie", "Fichier", "DNS", "Monitoring",
                      "Proxy", "Odoo", "ERPNext", "Metabase", "Bob50", "HyperPlanning"]
        X = build_feature_matrix(df_train, all_ports_train)
        y, label_cols = build_label_matrix(df_train, label_cols)
        print(f"\n[INFO] X shape = {X.shape}, y shape = {y.shape}")
    
        # Validation croisée (5-fold)
        model_cc = ClassifierChain(base_estimator=LogisticRegression(max_iter=1000),
                                     order='random', random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model_cc, X, y, cv=kf, scoring='accuracy')
        print(f"\n[INFO] Accuracy moyenne en validation croisée (5-fold) : {cv_scores.mean():.4f}")
    
        # Séparation train/test et entraînement
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_cc = train_classifier_chains(X_train, y_train)
        y_pred = model_cc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n[INFO] Accuracy sur jeu de test (20%) : {acc:.4f}")
    
        # Affichage des matrices de confusion par label
        print("\n[INFO] Matrices de confusion par label :")
        for idx, label in enumerate(label_cols):
            cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
            if cm.size == 1:
                print(f"{label} : {cm[0, 0]} (valeur unique)")
            else:
                tn, fp, fn, tp = cm.ravel()
                print(f"{label} : TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
        # ----------------------------------------------------------------------------
        # Scan Nmap classique et prédiction sur de nouvelles machines
        # ----------------------------------------------------------------------------
        TARGET_IP = get_ip_input("\nIP à scanner : ")  # Exemple : 192.168.0.183
        PORT_RANGE = get_port_range_input("Plage de ports (ex: 1-65535) : ")
        SCAN_OPTIONS = get_scan_options_input("Options de scan (ex: -sV -Pn -sT) : ")
    
        scan_result = launch_nmap_scan(TARGET_IP, PORT_RANGE, SCAN_OPTIONS)
        df_scan = parse_scan_result(scan_result)
    
        if df_scan.empty:
            print("\n[WARNING] Aucune machine détectée 'up' ou aucun port ouvert !")
            print("[HINT] Vérifiez l'IP, ajoutez -Pn, exécutez en sudo, etc.")
            return
    
        print("\n[INFO] Résultat du scan classique :")
        print(df_scan)
    
        X_new = build_feature_matrix(df_scan, all_ports_train)
        print(f"\n[INFO] X_new shape = {X_new.shape} pour {len(df_scan)} machine(s) scannée(s).")
    
        y_pred_new = model_cc.predict(X_new)
        for i, row in df_scan.iterrows():
            machine = row['machine']
            pred_vector = y_pred_new[i]
            labels_dict = {label_cols[j]: bool(pred_vector[j]) for j in range(len(label_cols))}
            business_needs = assign_business_need(labels_dict)
            print("\n" + "=" * 50)
            print(f"Machine : {machine}")
            print("=" * 50)
            print(f"Ports ouverts           : {row['ports_ouverts']}")
            print(f"Labels prédits          : {labels_dict}")
            print(f"Besoins métiers assignés: {business_needs}")
    
        # ----------------------------------------------------------------------------
        # Scan de sécurité Nmap avec script NSE
        # ----------------------------------------------------------------------------
        nse_script = get_nse_script_input("\nNom du script NSE à utiliser pour le scan de sécurité (ex: vuln, http-enum) : ")
        port_range_sec = get_port_range_input("Plage de ports pour le scan de sécurité (ex: 1-65535) : ")
    
        for i, row in df_scan.iterrows():
            machine = row['machine']
            sec_scan_result = launch_nmap_security_scan(machine, port_range_sec, nse_script)
            print("\n" + "=" * 50)
            print(f"Scan de sécurité pour la machine : {machine}")
            print("=" * 50)
            pretty_print_dict(sec_scan_result, title="Résultat du scan de sécurité")
    
    except Exception as e:
        print(f"[FATAL ERROR] Une erreur inattendue s'est produite : {e}")


if __name__ == "__main__":
    main()

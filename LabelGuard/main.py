#!/usr/bin/env python3
"""
Main entry point for the LabelGuard application.
Coordinates dataset generation, model training/evaluation, and network scanning.
Modified to log output both to console and to `output.txt` for readability.
Made with the help of OpenAI's ChatGPT.
"""

import logging
import sys
from data import create_large_training_dataset, build_feature_matrix, build_label_matrix
from model import train_classifier_chain, evaluate_model, cross_validate_model
from scanner import launch_nmap_scan, parse_scan_result, launch_nmap_security_scan
from utils import get_ip_input, get_port_range_input, get_scan_options_input, get_nse_script_input
from business import add_business_need_column, assign_business_need
from sklearn.model_selection import train_test_split
from datetime import datetime


def main():
    # ----------------------------------------------------------------
    # Configure logging to output to both console and a file
    # ----------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create formatter with timestamp, level, and message
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler (writes to output.txt)
    fh = logging.FileHandler(f"Rapport_du_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    # ----------------------------------------------------------------
    # Start of application logic
    # ----------------------------------------------------------------
    # Génération du dataset synthétique
    df_train = create_large_training_dataset(n=4082, seed=42)
    logger.info("Dataset d'entraînement généré avec %d enregistrements", df_train.shape[0])

    # Ajout de la colonne 'besoin_metier'
    df_train = add_business_need_column(df_train)

    # Détermination de tous les ports présents dans le dataset
    all_ports_train = sorted({port for ports in df_train['ports_ouverts'] for port in ports})
    logger.info("Ports possibles dans le dataset: %s", all_ports_train)

    # Construction des matrices de features et labels
    label_cols = ["Web", "BaseDeDonnees", "Messagerie", "Fichier", "DNS", "Monitoring",
                  "Proxy", "Odoo", "ERPNext", "Metabase", "Bob50", "HyperPlanning", "GitLab"]
    X = build_feature_matrix(df_train, all_ports_train)
    y, label_cols = build_label_matrix(df_train, label_cols)
    logger.info("Matrice de features: %s, Matrice de labels: %s", X.shape, y.shape)

    # Validation croisée et évaluation du modèle
    cv_f1 = cross_validate_model(X, y)
    logger.info("F1-macro moyenne en validation croisée (5-fold): %.4f", cv_f1)

    # Séparation train/test et entraînement du modèle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_classifier_chain(X_train, y_train)
    (
        test_accuracy,
        test_hloss,
        test_f1,
        (ci_low, ci_high),
        y_pred,
    ) = evaluate_model(model, X_test, y_test, label_cols)

    logger.info(
        "Test set — Acc: %.4f | Hamming: %.4f | F1-macro: %.4f "
        "(95%% CI [%.4f – %.4f])",
        test_accuracy,
        test_hloss,
        test_f1,
        ci_low,
        ci_high,
    )


    # Scan Nmap classique et prédiction sur de nouvelles machines
    target_ip = get_ip_input("Entrez l'IP à scanner : ")
    port_range = get_port_range_input("Entrez la plage de ports (ex: 1-65535) : ")
    scan_options = get_scan_options_input("Entrez les options de scan (ex: -sV -Pn -sT) : ")
    scan_result = launch_nmap_scan(target_ip, port_range, scan_options)
    df_scan = parse_scan_result(scan_result)
    if df_scan.empty:
        logger.warning("Aucune machine détectée 'up' ou aucun port ouvert trouvé.")
        sys.exit(1)

    logger.info("Résultat du scan classique:\n%s", df_scan)
    X_new = build_feature_matrix(df_scan, all_ports_train)
    y_pred_new = model.predict(X_new)
    for i, row in df_scan.iterrows():
        machine = row['machine']
        pred_vector = y_pred_new[i]
        labels_dict = {label_cols[j]: bool(pred_vector[j]) for j in range(len(label_cols))}
        business_needs = assign_business_need(labels_dict)
        logger.info("Machine: %s\nPorts ouverts: %s\nLabels prédits: %s\nBesoins métiers assignés: %s",
                    machine, row['ports_ouverts'], labels_dict, business_needs)

    # Scan de sécurité Nmap avec script NSE
    nse_script = get_nse_script_input("Entrez le nom du script NSE (ex: vuln, http-enum) : ")
    port_range_sec = get_port_range_input("Entrez la plage de ports pour le scan de sécurité (ex: 1-65535) : ")
    for i, row in df_scan.iterrows():
        machine = row['machine']
        sec_scan_result = launch_nmap_security_scan(machine, port_range_sec, nse_script)
        logger.info("Résultat du scan de sécurité pour la machine %s:\n%s", machine, sec_scan_result)

if __name__ == "__main__":
    main()

"""
Module de mapping des labels aux besoins métiers.
"""

def assign_business_need(labels: dict) -> list:
    """
    Assigne des besoins métiers en fonction des labels fournis.
    Pour chaque label à True, retourne le besoin métier correspondant.
    """
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
    needs = []
    for label, is_present in labels.items():
        if is_present:
            need = label_to_need.get(label)
            if need and need not in needs:
                needs.append(need)
    if not needs:
        needs.append("Aucun besoin métier assigné")
    return needs

def add_business_need_column(df):
    """
    Ajoute une colonne 'besoin_metier' au DataFrame en se basant sur la colonne 'labels'.
    """
    df = df.copy()
    df["besoin_metier"] = df["labels"].apply(assign_business_need)
    return df

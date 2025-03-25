"""
Module d'utilitaires.
Contient des fonctions pour la validation des IP, des plages de ports, et la collecte d'entrées utilisateur.
"""

import re

def pretty_print_dict(d: dict, title: str = "") -> None:
    """
    Affiche le dictionnaire 'd' en format JSON lisible, avec un titre optionnel.
    """
    import json
    if title:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
    print(json.dumps(d, indent=2, ensure_ascii=False))

def validate_ip(ip: str) -> bool:
    """
    Valide qu'une chaîne est une adresse IPv4 correcte.
    """
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(pattern, ip):
        return all(0 <= int(octet) <= 255 for octet in ip.split('.'))
    return False

def get_ip_input(prompt: str) -> str:
    """
    Demande à l'utilisateur une adresse IP jusqu'à ce qu'une adresse valide soit fournie.
    """
    while True:
        ip = input(prompt)
        if validate_ip(ip):
            return ip
        else:
            print("[ERROR] Format d'IP invalide. Veuillez réessayer.")

def validate_port_range(port_range: str) -> bool:
    """
    Valide que la plage de ports est au format 'start-end' avec des valeurs valides.
    """
    pattern = r'^\d{1,5}-\d{1,5}$'
    if re.match(pattern, port_range):
        start, end = map(int, port_range.split('-'))
        if 1 <= start <= end <= 65535:
            return True
    return False

def get_port_range_input(prompt: str) -> str:
    """
    Demande à l'utilisateur une plage de ports jusqu'à obtenir une valeur valide.
    """
    while True:
        port_range = input(prompt)
        if validate_port_range(port_range):
            return port_range
        else:
            print("[ERROR] Format de plage de ports invalide. Utilisez le format 'start-end' avec des valeurs entre 1 et 65535.")

def get_scan_options_input(prompt: str) -> str:
    """
    Demande à l'utilisateur les options de scan jusqu'à obtenir une chaîne non vide.
    """
    while True:
        options = input(prompt)
        if options.strip():
            return options
        else:
            print("[ERROR] Les options de scan ne peuvent pas être vides.")

def get_nse_script_input(prompt: str) -> str:
    """
    Demande à l'utilisateur le nom d'un script NSE jusqu'à obtenir une chaîne non vide.
    """
    while True:
        script = input(prompt)
        if script.strip():
            return script
        else:
            print("[ERROR] Le nom du script NSE ne peut pas être vide.")

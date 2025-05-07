"""
Module de scanning.
Contient les fonctions pour lancer des scans Nmap classiques et de sécurité, puis pour parser les résultats.
"""

import nmap
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def launch_nmap_scan(ip_address: str, port_range: str, options: str) -> dict:
    """
    Lance un scan Nmap standard sur l'IP et la plage de ports spécifiés.
    """
    try:
        nm = nmap.PortScanner()
        cmd_args = f"{options} -p {port_range}"
        logger.info("Lancement du scan Nmap sur %s avec : %s", ip_address, cmd_args)
        scan_result = nm.scan(hosts=ip_address, arguments=cmd_args)
        return scan_result
    except Exception as e:
        logger.error("Erreur lors du scan Nmap: %s", e)
        return {}

def parse_scan_result(scan_result: dict) -> pd.DataFrame:
    """
    Parse le résultat du scan Nmap et retourne un DataFrame avec les adresses IP et les ports ouverts.
    """
    hosts_data = []
    scan_data = scan_result.get('scan', {})
    for host, host_info in scan_data.items():
        status = host_info.get('status', {}).get('state', 'down')
        if status == 'up':
            ports_open = []
            tcp_data = host_info.get('tcp', {})
            for port, pdata in tcp_data.items():
                if pdata.get('state') == 'open':
                    ports_open.append(port)
            hosts_data.append({'machine': host, 'ports_ouverts': ports_open})
    return pd.DataFrame(hosts_data)

def launch_nmap_security_scan(ip_address: str, port_range: str, nse_script: str) -> dict:
    """
    Lance un scan de sécurité Nmap utilisant un script NSE.
    """
    try:
        nm = nmap.PortScanner()
        cmd_args = f"-sT -sV -Pn --script={nse_script} -p {port_range}"
        logger.info("Lancement du scan de sécurité sur %s avec : %s", ip_address, cmd_args)
        scan_result = nm.scan(hosts=ip_address, arguments=cmd_args)
        return scan_result
    except Exception as e:
        logger.error("Erreur lors du scan de sécurité Nmap: %s", e)
        return {}

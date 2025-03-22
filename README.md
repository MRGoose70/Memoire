# LabelGuard

LabelGuard est un outil Python conçu pour :

- Générer un dataset multi-label de machines et assigner des besoins métiers en fonction des labels détectés.
- Entraîner un modèle de classification multi-label basé sur des chaînes de classificateurs (Classifier Chains).
- Évaluer le modèle via validation croisée (5-fold) et un split train/test.
- Lancer un scan Nmap classique pour prédire les labels et assigner les besoins métiers.
- Effectuer un scan de sécurité Nmap en utilisant des scripts NSE.

---

## Fonctionnalités

### 1. Génération de datasets synthétiques

- Création de datasets multi-label simulant des machines avec des ports ouverts et des labels associés.
- Assignation automatique de besoins métiers en fonction des labels détectés.

### 2. Entraînement et évaluation de modèles

- Entraînement d'un modèle de classification multi-label basé sur des chaînes de classificateurs.
- Validation croisée (5-fold) pour évaluer les performances du modèle.
- Séparation train/test pour mesurer l'accuracy sur un jeu de test.

### 3. Scan Nmap classique

- Lancement d'un scan Nmap pour détecter les ports ouverts sur une machine cible.
- Prédiction des labels et assignation des besoins métiers en fonction des résultats du scan.

### 4. Scan de sécurité Nmap

- Utilisation de scripts NSE pour effectuer des scans de sécurité approfondis sur une machine cible.

---

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances nécessaires. Vous pouvez les installer via le fichier `requirements.txt`.

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**Liste des dépendances :**

- `python-nmap` : Interface Python pour Nmap.
- `pandas` : Manipulation et analyse de données.
- `numpy` : Calcul scientifique.
- `scikit-learn` : Outils de machine learning.

---

## Utilisation

### Lancer le script

Pour exécuter le script principal, utilisez la commande suivante :

```bash
python main.py
```

### Étapes principales

#### Génération du dataset

- Le script génère un dataset synthétique de machines avec des ports ouverts et des labels associés.

#### Entraînement et évaluation du modèle

- Le modèle est entraîné sur le dataset généré.
- Une validation croisée (5-fold) est effectuée pour évaluer les performances.

#### Scan Nmap classique

- Fournissez une adresse IP, une plage de ports et des options de scan.
- Le script effectue un scan Nmap et prédit les labels associés à la machine scannée.

#### Scan de sécurité Nmap

- Fournissez un script NSE et une plage de ports.
- Le script effectue un scan de sécurité approfondi et affiche les résultats.

---

## Exemple d'exécution

```bash
python main.py --scan-classic --ip 192.168.1.1 --ports 1-1000
```

---

## Structure du projet

- **`main.py`** : Script principal.
- **`dataset_generator.py`** : Génération de datasets synthétiques.
- **`model_training.py`** : Entraînement et évaluation du modèle.
- **`nmap_scanner.py`** : Scan Nmap classique et de sécurité.
- **`requirements.txt`** : Liste des dépendances.

---

## Fonctionnalités détaillées

### Génération de datasets

Le script génère un dataset synthétique contenant :

- **`machine`** : Adresse IP fictive.
- **`ports_ouverts`** : Liste de ports ouverts.
- **`labels`** : Dictionnaire indiquant les services détectés (Web, BaseDeDonnees, etc.).

### Modèle de classification

Le modèle utilise des chaînes de classificateurs (`ClassifierChain`) basées sur une régression logistique (`LogisticRegression`) pour prédire les labels multi-label.

### Scan Nmap

Le script utilise la bibliothèque `python-nmap` pour effectuer des scans Nmap. Les résultats sont analysés pour extraire les ports ouverts et prédire les labels associés.

### Scan de sécurité

Le script permet d'exécuter des scripts NSE (Nmap Scripting Engine) pour effectuer des analyses de sécurité approfondies.

---

## Résultats

Les résultats des scans et des prédictions sont affichés de manière lisible, avec :

- Les ports ouverts détectés.
- Les labels prédits.
- Les besoins métiers assignés.

---

## Contributeurs

- **Auteur principal** : Arts Loïck

---

## Sources

- [python-nmap Documentation](https://pypi.org/project/python-nmap/)
- [Scikit-learn Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html)
- [Nmap NSE Documentation](https://nmap.org/book/nse.html)

---

## Licence

Ce projet est sous licence MIT. Vous êtes libre de l'utiliser, de le modifier et de le distribuer.

---

## Remarques

- Assurez-vous d'avoir les permissions nécessaires pour exécuter des scans Nmap sur les machines cibles.
- Certains scans nécessitent des privilèges administratifs (`sudo`).

---

## Contact

Pour toute question ou suggestion, veuillez contacter MRGoose70
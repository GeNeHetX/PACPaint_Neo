# 3. Utilisation du modèle sur une nouvelle image
On utilise le script `pacpaint_tb\process_single_WSI\process_wsi.py`. Il faut compléter les chemins `PATH_TEMP_DIR`, `PATH_WSI`, `PATH_NEO`, `PATH_COMP`, qui sont respectivement le chemin vers le dossier où l'on va stocker les résultats, le chemin vers l'image scannée, le chemin vers le dossier contenant les poids du modèle Neo, le chemin vers le dossier contenant les poids du modèle Comp.

Une fois les calculs réalisés on peut analyser les résultats dans le notebook `infer_wsi.ipynb` en remplissant à nouveau les bons chemins.
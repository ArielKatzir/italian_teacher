"""
Italian gender exceptions.

Words that don't follow standard patterns (e.g., -o = masc, -a = fem).
"""

from typing import Dict

GENDER_EXCEPTIONS: Dict[str, str] = {
    # Masculine words ending in -a
    "problema": "Masc",
    "tema": "Masc",
    "sistema": "Masc",
    "programma": "Masc",
    "clima": "Masc",
    "poeta": "Masc",
    "pianeta": "Masc",
    "cinema": "Masc",
    "dramma": "Masc",
    "diploma": "Masc",
    "panorama": "Masc",
    "pilota": "Masc",
    "atleta": "Masc",
    "turista": "Masc",  # Can be both, but often masc
    # Feminine words ending in -o
    "mano": "Fem",
    "radio": "Fem",
    "foto": "Fem",
    "moto": "Fem",
    "auto": "Fem",
    "eco": "Fem",
    # Irregular -e endings (masculine)
    "cane": "Masc",
    "pane": "Masc",
    "mare": "Masc",
    "sole": "Masc",
    "sale": "Masc",
    "pesce": "Masc",
    "padre": "Masc",
    "nome": "Masc",
    "fiore": "Masc",
    "colore": "Masc",
    "amore": "Masc",
    "errore": "Masc",
    # Irregular -e endings (feminine)
    "notte": "Fem",
    "madre": "Fem",
    "classe": "Fem",
    "chiave": "Fem",
    "nave": "Fem",
    "gente": "Fem",
    "mente": "Fem",
    "legge": "Fem",
    "voce": "Fem",
    "luce": "Fem",
    # Animals with fixed gender regardless of biological sex
    "aquila": "Fem",  # eagle (even male eagles)
    "volpe": "Fem",  # fox
    "tigre": "Fem",  # tiger
    "giraffa": "Fem",  # giraffe
    "balena": "Fem",  # whale
    "farfalla": "Fem",  # butterfly
    "mosca": "Fem",  # fly
    "rana": "Fem",  # frog
    "topo": "Masc",  # mouse
    "ragno": "Masc",  # spider
    "corvo": "Masc",  # crow
    "leopardo": "Masc",  # leopard
    "serpente": "Masc",  # snake
    "elefante": "Masc",  # elephant
}

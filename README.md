# Apprentissage par renforcement profond Master


Le but de ce travail est de se familiariser avec les algorithmes de Deep Reinforcement Learning.

### Execution

- `cartpole_agent.py` permet de faire apprendre une agent et de sauvegarder le modèle obtenu quelque part.
- `DQN_cartpole.py` contient l'architecture du réseau de neurones.
- `ReplayMemory.py` contient le code du buffer circulaire de mémoire
- `test_agent_cartpole.py` contient le code qui permet de tester un modèle et de sauvegarder ou non la vidéo.

**Entrainement**

Pour entrainer le modèle, ouvrez le fichier `cartpole_agent.py` et changer les hyperparamètre souhaité (dans la fonction `__init__`) puis executez le dans votre IDE ou alors avec la ligne de commande `python3 cartpole_agent.py`

Pour tester votre modèle il faut obligatoirement le faire en ligne de commande. Faite la commande suivante :

`python3 test_agent_cartpole.py [path] [epsilon] [record] [wrapped] [render]`

avec :
 - `[path]` le chemin vers le modèle (pas d'espace)
 - `[epsilon]` le pourcentage d'exploration souhaité (< 1).
 - `[record]` True ou False, si vous souhaiter enregistrer la video vers 'videos/new_video.mp4'
 - `[wrapped]` True ou False, si vous voulez wrapper l'environnement gym. Si True le nombre max d'étape est 500.
 - `[render]` True ou False, si vous voulez voir la video du Cartpole (prend plus de temps à s'executer pour un modèle infini)

 Pour voir le modèle maximal executez donc 

 `python3 test_agent_cartpole.py models/immortal_boy 0 False False True`.

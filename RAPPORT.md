# Rapport DQN Gout Numa

## Introduction

Dans ce TP nous allons implémenter l'algorithme de *Deep Q-Network* grâce à la bibliothèque python d'environnement *Gym*. Puis sur un environnement plus compliqué s'inspirant du jeu *Minecraft* de Mojang.

## Partie I : Familiarisation avec *Gym*

Gym propose une large catégorie d'environnements pour expérimenter des algorithmes de RL. Leur but est de fournir une grande gamme d'environemments variés afin d'améliorer les progrès dans l'apprentissage par renforcement.

*Gym* propose donc beaucoup d'environemments, à chaque étape l'agent effectue une action, l'observation de l'environnement change et une récompense nous est donnée. Nous utiliserons l'environnement CartPole-v1 dans la première partie du TP. *Gym* est fourni avec une méthode de rendu d'image et un *VideoRecorder* qui va nous permettre d'enregistrer les actions de notre agent entrainé.

## Partie II : Algorithme DQN

### II.1 ReplayMemory

L'algorithme fonctionne avec une mémoire circulaire, quand la mémoire est remplie les premières observations sont ecrasées. Il y est stocké le tuple `(state, action, next_state, reward)`.

Notre implémentation se trouve dans le fichier `ReplayMemory.py`. Nous implémentons 3 méthodes dans notre classe. 
- `push` qui ajoute une étape et écrase les précédentes si nécéssaire.
- `sample` qui renvoie un échantillon de la mémoire de la taille `batch_size`
- `len` qui renvoie la longueur de la mémoire actuelle. Servira pour le début de l'apprentissage car au début la mémoire est vide, on ne peut donc pas extraire un échantillon.

### II.2 Premier réseau de neurone



# Vocabulaire

- modèle+ = modèle pré-entrainé sur TOUT coco
- dataset+ = tout COCO
- modèle_set = modèle préentrainé sur juste le set de classes
- dataset_set :
    - dataset ne contenant que des images contenants une classe dans le set
    - enlever les objets qui ne sont pas dans le se de class



# TODO


## STAN :
- DETR Part

## MERYEM
- YoloV6 Part
- crée dataset_animaux




## TOUS LES DEUX :
- Télécharger COCO 2017
- récupérer modèle + et faire inférence sur dataset_animaux, puis filtrer les objets que l'on veut pas
- entrainer (si possible) modèle_animaux
- voire si on peut connecter ca avec fyftyone
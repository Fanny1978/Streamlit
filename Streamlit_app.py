import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import joblib
import os
from sklearn.preprocessing import LabelEncoder

@st.cache_data(ttl=6*3600)
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Fichier de modèle non trouvé : {model_path}")
        return None
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du modèle : {e}")
        return None    

model = load_model('lgbm_best_model.joblib') 



print(lgb.__version__)
# Charger les données de test 
# Si vous n'en avez pas, vous pourrez utiliser des entrées manuelles

@st.cache_data(ttl=6*3600)
def load_test_data(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Fichier de test non trouvé : {data_path}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"Le fichier de test {data_path} est vide.")
        return None
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du fichier de test: {e}")
        return None

# Données de test 
test_data = load_test_data('accidents_fictifs_numeriques_reorganise.csv')


@st.cache_data(ttl=1*3600)
def charger_donnees(chemin_fichier):
    """Charge un fichier CSV et le retourne dans un DataFrame."""
    try:
        return pd.read_csv(chemin_fichier)
    except FileNotFoundError:
        st.error(f"Fichier non trouvé : {chemin_fichier}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"Le fichier {chemin_fichier} est vide.")
        return None
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement de {chemin_fichier}: {e}")
        return None

# Chemins des fichiers CSV
chemin_caracteristiques=" https://drive.google.com/uc?export=download&id=1yHKtsMHsHnKSr-qOMuw8YflJXl14qI5g "
chemin_vehicules=" https://drive.google.com/uc?export=download&id=1j0GqVV-SDoZ4wujmJuiM8UFSgMUOKqnh "
chemin_lieux=" https://drive.google.com/uc?export=download&id=1rwkNwarEe0LeEYfh2n6RmXH3GeQ20Gym "
chemin_usagers=" https://drive.google.com/uc?export=download&id=1TRaBSrx57-ZNexeq7VOdKReKtm2qgifd"
chemin_avant_machine_learning =" https://drive.google.com/uc?export=download&id=1e25ERNtmEXrIB_ZYB1ELHNBXVJWjCIPR "



# Chargement des DataFrames avec mise en cache
df_caracteristiques = charger_donnees(chemin_caracteristiques)
df_vehicules = charger_donnees(chemin_vehicules)
df_lieux = charger_donnees(chemin_lieux)
df_usagers = charger_donnees(chemin_usagers)
df_avant_machine_learning = charger_donnees(chemin_avant_machine_learning)
st.title("Projet Accidents Routiers en France ")


# Définir la largeur de la barre latérale
sidebar_width = 400  # Largeur en pixels

# Ajouter du code CSS personnalisé
st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] {{
            width: {sidebar_width}px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Charger l'image 
image = Image.open("logo-2021.png")
st.sidebar.image(image, width=250)
st.sidebar.title("Sommaire")
pages=["Introduction", "Description des données", "Nettoyage et enrichissement des données", "Exploration", "DataVisualisation", "Fusion des 4 datasets et Pre-processing avant Machine Learning", "Modèle de prédiction de la gravité (4 classes)", "Modèle de prédiction binaire de la gravité","Modèle de série temporelle SARIMA", "Conclusion"]
page=st.sidebar.radio(" ", pages)
#st.sidebar.markdown("---")  # Ajoute une ligne de séparation
st.sidebar.markdown("**Auteurs :**")

participants = {
    "Bassira Ali Alhadji": "https://www.linkedin.com/in/bassira-ali-alhadji-33b778179/",
    "Fanny Le Jaouen": "https://www.linkedin.com/in/fanny-le-jaouen-b0018b296/",
    "Christophe Varlet":"https://www.linkedin.com/in/christophevarlet/"
}

for nom, url in participants.items():
    st.sidebar.markdown(f"- <a href='{url}' target='_blank'>{nom}</a>", unsafe_allow_html=True)


st.sidebar.markdown("Promotion Data Analyst : Janvier 2025")

if page == pages[0] : 
  st.image("accident 3.jpg", width=500)
  st.write("Ce projet a été réalisé dans le cadre de notre formation de Data Analyst chez DataScientest.")
  st.write("### :material/target: Objectifs du projet ")
  st.write(":heavy_check_mark: Analyser les tendances concernant les accidents routiers")
  st.write(''':heavy_check_mark: Déterminer une liste de caractéristiques qui déterminent la gravité d'un accident''')
  st.write(''':heavy_check_mark: Prédire la gravité d'un accident''')
  st.write(" ### :material/folder: Sources des données")
  st.write(' :heavy_check_mark: Fichier national des accidents corporels de la circulation dit "Fichier BAAC" (Open data)')
  st.write(''':heavy_check_mark: Période : 2005-2023 pour l'exploration puis période 2019-2023 pour la suite du projet car cette période est mieux renseignée ''')
 
  
if page == pages[1] : 

  st.write(" ### :material/database: Description des données")
  st.write(':heavy_check_mark: 4 Datasets : 4 fichiers csv par année (soit 72 fichiers csv de 2005 à 2023)')
  st.write(' :heavy_check_mark: Chaque dataset apporte des informations complémentaires sur les accidents')
  st.write(''':material/description: Dataset **Caractéristiques** : décrit les circonstances générales de l'accident (types de collisions, luminosité, date)''')
  st.write(''' :material/description: Dataset **Lieux** : décrit le lieu principal de  l'accident (catégorie route, nb de voies, régime de circulation, surface de la chaussée,…)''')
  st.write(''':material/description: Dataset **Véhicules** : décrit les véhicules impliqués dans l'accident (n° plaque immatriculations, type de véhicule, localisation du choc, manœuvre,…)''')
  st.write(''' :material/description: Dataset **Usagers** : décrit les usagers impliqués dans l'accident (place de l'usager dans le véhicule, gravité, trajet de l'usager,…)''')
  st.write(" ##### Jointure des 4 datasets")
  st.image("jointures des tables.jpg", caption="Les 4 datasets avec leurs liaisons", width=600)
  

if page == pages[2] : 
  st.write("### :material/cleaning_services: Nettoyage des Données")
  st.write(":heavy_check_mark: Gestion des Valeurs Manquantes")
  st.write(":heavy_check_mark: Suppression des doublons")
  st.write(":heavy_check_mark: Gestion de la cohérence du format des données")
  st.write("### :material/label_important: Enrichissement des Données")
  st.write(":heavy_check_mark: Mapping des données")
  st.write(":heavy_check_mark: Ajout de nouvelles variables")
  

if page == pages[3] :
  st.write("### :material/search: Exploration")
  st.write('Les datasets présentés concernent la période 2019-2023 car sur la période 2005-2023 la volumétrie engendrait une saturation de Streamlit')
  st.write(' ')
  st.write('Dataset Caractéristiques (2019-2023)')
  st.dataframe(df_caracteristiques.head(10))
  st.write(df_caracteristiques.shape)

  st.write('Dataset Véhicules (2019-2023)')
  st.dataframe(df_vehicules.head(10))
  st.write(df_vehicules.shape)
 

  st.write('Dataset Lieux (2019-2023)')
  st.dataframe(df_lieux.head(10))
  st.write(df_lieux.shape)
 
  st.write('Dataset Usagers (2019-2023)')
  st.dataframe(df_usagers.head(10))
  st.write(df_usagers.shape)

if page == pages[4] : 
  st.write("### DataVisualisation")
  tab1, tab2, tab3, tab4 = st.tabs(["Analyse Générale", "Analyse temporelle", "Zoom sur le gravité", "Scoring géographique"])

  with tab1:
    st.header("Data visualisation réalisée sous PowerBI")
    st.image("Analyse Générale 1.jpg", width=1000)
    st.write(' :earth_americas: ***Analyse géographique*** : On remarque que les accidentés se concentrent sur les grandes agglomérations : Ile de France, Marseille, Lyon et Bordeaux. ')
    st.write('''Les 2 zones qui concentrent le plus de d'accidents sont l'Ile de France et le Sud-Est.''')
    st.write('''La majorité des accidents ont lieu en agglomération avec 62 % contre 38 % hors agglomération. ''')
    st.image("distribution âge.jpg", width=600)
    st.write('''On observe un pic d'accidents autour de la vingtaine. Ce pic concerne les jeunes conducteurs qui en plus de leur manque d'expérience adoptent parfois des comportements à risque. La recherche de sensations fortes peuvent les pousser à adopter des comportements dangereux, comme la vitesse excessive, les dépassements risqués ou la conduite sous l'influence de l'alcool. De plus, les jeunes adultes ont tendance à conduire plus tard et pendant les week-ends, ce qui augmente leur exposition aux risques. ''')
    st.image("distribution vitesse.jpg", width=600)
    st.write('''La majorité des accidents a lieu sur les routes où la vitesse est limitée à 50km/h, c'est à dire princpalement en agglomération ''')
    
    

  with tab2:
    st.header("Data visualisation réalisée sous PowerBI")
    st.image("Analyse Temporelle par année.jpg", caption="Nombre d'accidents par an", width=700)
    st.write(''' :date: ***Analyse par an*** : On observe un légère baisse des accidents de 2019 à 2023. La baisse est plus évidente ci-desous sur la période 2005-2023. En 2020 la baisse du nombre d'accidents est dûe à la reduction des déplacement en raison du confinement. ''')
    st.image("Nb_accidents_par annee_2005_2023.jpg", caption="Nombre d'accidents par an de 2005 à 2023 - Histogramme réalisé sous Python avec Seaborn", width=700)
    st.write(''':date: De 2005 à 2023 on observe clairement une baisse du nombre d'accidents annuels. La baisse est importante de 2005 à 2012 puis s'infléchit un peu. Les politiques de prévention routière et les évolutions technologiques ont porté leurs fruits.''')
    st.image("Analyse Temporelle par mois.jpg", caption="Nombre d'accidents par mois", width=700)
    st.write(':date: ***Analyse par mois*** : On observe un phénomène saisonnier. Le nombre d accidents augmentent au cours des mois de Mai, Juin et Juillet puis en Septembre et Octobre. Ces mois sont favorables aux départs en WE et en vacances avec une météo clémente. ')
    st.image("Analyse Temporelle par jour.jpg", caption="Nombre d'accidents par jour", width=700)
    st.write(''':date: ***Analyse sur la semaine*** : On observe un pic d'accidents le vendredi en raison des départs en week-end et des sorties festives. Le dimanche est le jour qui connaît le moins d'accidents.''')
    st.image("Analyse Temporelle par créneau horaire.jpg", caption="Nombre d'accidents par créneau horaire", width=700)
    st.write(''':date: ***Analyse sur la journée*** : On obverve un petit pic d'accidents entre 8h et 10h à l'heure de pointe du matin et un pic plus important entre 16h et 20h à l'heure de point du soir.''')

  with tab3:
    st.image("Répartition des accidentés par type de gravité.jpg", caption="Zoom sur la gravité",  width=500)
    st.write('''On observe sur ce Pie Chart, la répartition des accidents par gravité : 42 % d’indemnes, 40 % de blessés légers, 15 % de blessés hospitalisés(+24h) et 2,6% de tués (sur le coup ou dans les 30 jours qui suivent l’accident) ''')
    st.image("répartition des accidents graves selon genre  agglo et type vehicule.jpg", caption="Zoom sur la gravité",  width=900)
    st.write('''***Ici on considère uniquement les accidents graves (blessés hospitalisés et tués)*** :''')
    st.write('''Sur le 1er Pie-Chart, on observe que les accidents graves sont répartis de la façon suivant selon le genre : 78 % concernent des hommes et 22 % des femmes. ''')
    st.write('''Sur le 2e Pie-Chart, on remarque que les accidents graves ont lieu à 55 % hors agglomération contre 45 % en agglomération. ''')
    st.write('''Sur le 3e Pie-Chart, on voit que 51 % des accidents graves concernent des voitures et 42 % concernent des 2 roues.''')
    st.image("Top 10 départements avec le plus d'accidents graves.jpg", caption="Zoom sur la gravité",  width=900)
    st.write('''***Ici on considère toujours uniquement les accidents graves (blessés hospitalisés et tués)*** :''')
    st.write('''Les 10 départements où ont lieu le plus d'accidents graves se situent en Ile de France, dans le Nord de la France, dans la région de Lyon-Grenoble, dans la région de Bordeaux et sur la côte méditerranéenne ''')

  with tab4:
    st.image("Zones géo accidents Grave et Conditions Atmosphérique.jpg", caption="Répartition géographique des accidents graves en conditions météorologiques défavorables",  width=900)
    st.write('''Ici c'est le scoring des zones avec un grand nombre d’accidents graves en conditions atmosphériques dégradées qui est représenté. C’est le Top20 des départements pour ces conditions. 
Il s’agit notamment de l’Ile de France, de la Bretagne, du Nord et de la région de Lyon. Il s’agit des zones où il faudrait investiguer et déterminer quelles améliorations apporter pour baisser le nombre d’accidents : travaux sur les routes, sur les intersections, au niveau du revêtement des routes ou des actions sur les limitations de vitesse.
''')
   
    # Ajoutez ici d'autres visualisations ou du texte pour l'onglet 2

if page == pages[5] : 
  st.write("### Fusion des 4 datasets ")
  st.write(':material/info: La fusion a été faite sur les données de 2019 à 2023 de France Métropolitaine car les données antérieures comptaient beaucoup de valeurs manquantes et certaines variables n étaient pas remplies de la même façon avant 2019. Et les données des DOM comportaient un fort taux de valeurs manquantes pour plusieurs variables.')
  st.write(' :heavy_check_mark: 1ère étape : Fusion des datasets **Véhicules** et **Usagers** sur la clé **id_vehicule**')
  st.write(':heavy_check_mark: 2e étape : Fusion avec les 2 autres datasets **Caracteristiques** et **Lieux** sur la clé **Num_acc**')
 
  st.write("### Pre-processing avant Machine Learning ")
  st.write(':heavy_check_mark: Suppression des Labels')
  st.write(':heavy_check_mark: Simplification par réduction du nombre de catégories de certaines variables')
  st.write(':heavy_check_mark: Les variables sont des variables catégorielles numériques')
  st.write('Dataset fusionné et pre-processé avant Machine Learning (2019-2023)')
  st.dataframe(df_avant_machine_learning.head(10))
  st.write(df_avant_machine_learning.shape)
if page == pages[6] : 
  st.write("### :gear: Modèle de Prédiction de la gravité d'un accident ")
  st.write('''***Objectif : réaliser un modèle pour prédire la gravité d'un accident à partir d'une liste de caractéristiques de l'accident***''')
  st.write(''':heavy_check_mark: La ***variable cible*** est la ***gravité*** qui prend 4 valeurs selon le niveau de gravité : ***indemne, blessé léger, blessé hospitalisé et tué*** ''')
  st.write(''':heavy_check_mark: Il s'agit d'un ***problème de classification multiclasse supervisée*** ''')
  st.image("Les 4 classes grav.jpg", caption="Variable cible", width=300) 
  
  st.write('''''')
  st.write(''':heavy_check_mark: Utilisation des méthodes de ***Features importance*** pour déterminer les ***variables explicatives*** ''')
  st.image("Features Importance avec LightGBM 2.png",caption="Histogramme des Features Importance du modèle LightGBM", width=700) 
  st.write(''':heavy_check_mark: Liste des 22 variables explicatives sélectionnées ''')
  st.image("22 variables.jpg", width=300) 

  st.write(''':heavy_check_mark: Sélection de 4 modèles d’arbre de classification que nous avons testés en parallèle  ''')
  st.image("modèles_sélectionnés.jpg", caption="Modèles d'arbre de classification sélectionnés", width=300) 
  st.write(''':heavy_check_mark: Déséquilibre de la variable cible ''')
  st.image("desequilibre classe variable cible.jpg", caption="Variable cible déséquilibrée", width=500) 
  st.write(''':heavy_check_mark: Pour gérer le déséquilibre de la variable cible nous avons testé : ''')
  st.write(''' - Le ***suréchantillonnage avec SMOTE*** : les classes minoritaires n'ont pas été mieux prises en compte''')
  st.write(''' - La ***pondération de classe*** : les classes minoritaires ont été mieux prises en compte notamment avec le modèle LightGBM''')
  st.write(''' - La ***pondération de classe*** et le ***SMOTE*** conjointement : les performances et la prise en compte des classes minoritaires ont été moins bonnes qu'avec la pondération seule.''')
  st.image("comparaison des modèles.jpg", caption="Comparaison des modèles lightGBM avec SMOTE + ponderation et avec pondération seule.", width=500) 
  st.write(''':heavy_check_mark: Le modèle LightGBM avec pondération seule a été sélectionné car il a permis d'obtenir les meilleures performances et la meilleure prise en compte des classes minoritaires.  ''')
  st.write(''':heavy_check_mark: L'optimisation du modèle LightGBM avec pondération a été réalisée avec GridSearchCV et RandomizedSearchCV pour trouver les meilleurs hyperparamètres. Cela a permis d'améliorer les performances générales du modèle.  ''')
  st.image("matrice confusion_4_classes.jpg", caption="Matrice de confusion du modèle LightGBM avec les meilleurs hyperparamètres et avec pondération.", width=500) 
  st.write('''Les métriques ***rappel de la classe 2*** et ***rappel de la classe 3*** ainsi que le ***rappel global*** ont été particulièrement suivis en raison du déséquilibre de la variable cible. D'autant plus que les classes déséquilibrées les ***tués*** et les ***blessés hospitalisés*** sont des classes importantes à bien prédire.''')
  st.image("Performances Light GBM_4_classes.jpg", caption="Performances du modèle LightGBM avec les meilleurs hyperparamètres et avec pondération de classe.", width=300) 
  st.write(''' :heavy_check_mark: Le  ***rappel de la classe 2*** est passé de 0,1 à 0,55 en utilisant la pondération de classe, ce qui est une belle amélioration. ''')
  st.write(''':heavy_check_mark: La performance du modèle sélectionné reste cependant modeste avec un ***rappel de 0.63*** et un ***F1 Score de 0,64*** ''')
  st.write('''Pour améliorer le modèle, il pourrait être intéressant de ''')
  st.write('''- Tester d'autres méthodes de rééquilibrage des classes comme le RandomOverSampler par exemple.''')
  st.write('''- D’augmenter le volume des données en ajoutant les données de 2024 pour tenter d’augmenter les performances du modèle.''')
  st.write('''- Mettre en place un modèle de prédiction de la gravité des accidents plus simple avec une variable cible binaire : accident grave (tué ou blessé grave) et accident non grave (indemne ou blessé léger).''')
  
  st.write(''' ### :curly_loop: Test du modèle LightGBM pour prédire la gravité d'accidents fictifs ''')
  if model is not None:
    st.write("Modèle chargé avec succès !")
    # Le reste de votre code
  else:
    st.stop()


  
  # Sélectionner les caractéristiques
  X_fictif = test_data[['age', 'secu1', 'catu', 'obsm', 'catv', 'col', 'secu2', 'zone_dep_num', 'sexe', 'catr',
                     'vma_categorie_numerique', 'agg', 'choc', 'trajet', 'horaire', 'circ', 'manv', 'mois', 'int',
                     'lum', 'nbv_categorie_numerique', 'surf']]

 
  # 3. Effectuer des prédictions
  y_pred_fictif = model.predict(X_fictif)

  # 4. Décodage manuel des prédictions
  def manuel_decode(y_pred_encoded):
    return [val + 1 for val in y_pred_encoded]

  y_pred_fictif_decoded = manuel_decode(y_pred_fictif)

  # 5. Ajouter les prédictions au DataFrame
  test_data['gravité_prédite'] = y_pred_fictif_decoded

  # 6. Mapper les niveaux de gravité aux labels
  gravite_mapping = {
    1: "indemne",
    2: "tué",
    3: "blessé hospitalisé",
    4: "blessé léger"
  }

# 6. Ajouter les labels au DataFrame
  test_data['gravité_prédite_label'] = [gravite_mapping.get(gravite, "inconnu") for gravite in y_pred_fictif_decoded]

# 7. Afficher les résultats
  st.write('#### :information_source: Caractéristiques de 6 accidents fictifs')

# 8. Afficher les prédictions
  st.write(test_data)

# 9. Mapper les niveaux de gravité aux labels et aux styles
  gravite_mapping = {
    1: ("indemne", "green"),
    2: ("tué", "red"),
    3: ("blessé hospitalisé", "maroon"),
    4: ("blessé léger", "palevioletred")
  }
# 10. Ajouter les labels et les styles au DataFrame
  test_data['gravité_prédite_label'] = [gravite_mapping.get(gravite, ("inconnu", "black"))[0] for gravite in y_pred_fictif_decoded]
  test_data['style_texte'] = [gravite_mapping.get(gravite, ("inconnu", "black"))[1] for gravite in y_pred_fictif_decoded]

# 11. Afficher les résultats sous forme de phrases avec les styles
  st.write('#### :gear: Prédiction pour les 6 accidents fictifs')

  for index, row in test_data.iterrows():
    accident_id = index + 1
    gravite_predite = row['gravité_prédite_label']
    style_texte = row['style_texte']
    st.markdown(f"Pour l'accident {accident_id}, le modèle a prédit : <span style='color:{style_texte}; font-weight: bold;'>{gravite_predite}</span>", unsafe_allow_html=True)
  st.write(''' #### :heavy_check_mark: Processus métiers auquels ce modèle prédictif pourrait s’appliquer  : ''')
  st.write(''' Ce modèle de prédiction pourrait intéresser : ''')
  st.write('''- Les ***autorités locales et nationales*** :L’analyse des variables qui permettent de prédire la gravité des accidents peut guider les autorités vers des nouvelles campagnes de sécurité routière. ''')
  st.write('''- Les ***compagnies d'assurance*** pourraient utiliser les prédictions pour affiner leurs modèles d'évaluation des risques et ajuster les primes d'assurance en fonction de la probabilité d'accidents graves. ''')
  st.write('''- Les ***constructeurs automobiles*** : le modèle pourrait être intégré aux systèmes de sécurité des véhicules. Le système ajusterait les paramètres de sécurité active (freinage d'urgence, contrôle de stabilité, etc.) en fonction de la prédiction de gravité.''')

 
if page == pages[7] : 
  
  st.write("###  :gear:   Modèle de prédiction binaire de la Gravité")
  st.write('''***Objectif : réaliser un modèle ***binaire*** pour prédire la gravité d'un accident à partir d'une liste de caractéristiques de l'accident***''')
  st.write(''':heavy_check_mark: La ***variable cible*** est la ***gravité*** qui prend 2 valeurs selon le niveau de gravité : ***grave*** (blessé hospitalisé et tué) et ***non grave*** (indemne, blessé léger) ''')
  st.write(''':heavy_check_mark: Il s'agit d'un ***problème de classification binaire supervisée*** ''')
  st.image("classes binaire Light GBM.jpg", caption="Variable cible binaire", width=300) 
  st.write(''':heavy_check_mark: Variable cible binaire déséquilibrée ''')
  st.image("desequilibre classe binaire.jpg", caption="Déséquilibre variable cible binaire", width=400)
  st.write(''':heavy_check_mark: Même liste de 22 variables explicatives que dans le modèle à 4 classes ''')
  st.image("22 variables.jpg", width=300) 
  st.write(''':heavy_check_mark: Choix du modèle LightGBM ''')
  
 
  st.write(''':heavy_check_mark: Application de la ***pondération de classe*** : la classe 1 (accident grave) est minoritaire.''')
  st.image("Matrice confusion Variable cible binaire2.jpg", caption="Matrice de confusion du modèle LightGBM binaire avec les meilleurs hyperparamètres et avec pondération.", width=500) 
  st.write(''':heavy_check_mark: Les métriques ***rappel des classes*** et  ***rappel global*** ont été particulièrement suivis en raison du déséquilibre de la variable cible.''')
  st.write('''Le  ***rappel global*** est de **0,79** (contre 0,63 pour le modèle à 4 classes) et le F1 Score est de **0,70** (contre 0,64 pour le modèle à 4 classes). ''')
  st.write('''**Les performances du modèle sont nettement meilleures que celles du modèle à 4 classes.**  ''')

if page == pages[8] : 
  
  st.write("### :hourglass:  Modèle de serie temporelle SARIMA")
  st.write('Prévision du nombre d accidents par type de gravité pour les 6 mois à venir à partir des 6 derniers mois.')
  st.image("prevision SARIMA.jpg", width=800)
  st.write('''Ces prévisions pourraient intéresser : ''')
  st.write('''- Les ***autorités locales et nationales*** pour évaluer l'efficacité des politiques de sécurité routière existantes, identifier les périodes où des mesures supplémentaires pouraient être nécessaires. Il s'agirait sur ces périodes de renforcer les contrôles de polices ou lancer des campagnes de sensibilisations ciblées. ''')
  st.write('''- Les ***hôpitaux*** pour anticiper, gérer et planifier les ressources sur les périodes de fortes volumétries d'accidents entrainant des hospitalisation. ''')
  st.write('''- Les ***compagnies d'assurance*** pour anticiper les fluctuations du nombre de sinistres et ajuster leurs provisions en conséquence. Cela leur permettrait de mieux gérer les risques finaciers liés aux accidents de la route.''')

if page == pages[9] : 
  st.image("accident1.jpg", width=400)
  st.write("### :heavy_check_mark: Conclusion")
  
  st.write('''#### Bilan du projet ''')
  st.write('''Nous avons mis en place un modèle pour **prédire la gravité d’un accident routier** en combinant **un traitement avancé des données** et **un modèle optimisé de Machine Learning**.''')
  st.write('''En mettant en œuvre un **pré-traitement rigoureux**, comprenant **la gestion des valeurs manquantes** et **l'enrichissement des données**, nous avons optimiser la qualité du jeu de données et améliorer leur **impact sur la prédiction de la gravité**. ''')
  st.write('''**Les résultats des modèles** : ''')
  st.write("<b><u>Modèle prédictif avec une variable cible à 4 classes</u> </b>", unsafe_allow_html=True)
  st.write('''-  **Rappel de 0.63** ''')
  st.write('''-  **F1 Score de 0,64** ''')
  st.write('''Les performances de ce modèle avec une **variable cible à 4 classes** restent modestes avec un ***rappel de 0.63*** et un ***F1 Score de 0,64***. Avec plus de temps nous aurions pu tester d'autres méthodes de rééquilibrage des classes et continuer l'optimisation du modèle pour obtenir de meilleures performances.''')
  st.write("<b><u>Modèle prédictif avec une variable cible binaire</u> </b>", unsafe_allow_html=True)
  st.write('''-  **Rappel de 0.79** ''')
  st.write('''-  **F1 Score de 0,70** ''')
  st.write('''Les performances de ce modèle avec **variable cible binaire** sont nettement meilleures avec un ***rappel de 0.79*** et un ***F1 Score de 0,70***. ''')
  st.write('''La prise en compte des données de 2024 qui vont bientôt être disponibles pourraient participer à augmenter les performances des modèles.''')
  st.write('''#### Ce que ce projet nous a apporté ''')
  st.write('''Ce projet de **Data Analyse** sur les accidents routiers en France a été une expérience enrichissante et formatrice dans le cadre de la **formation de Data Analyst** chez **DataScientest**. Il a été l’occasion de mettre en pratique nos connaissances en Python, de maîtriser les techniques de nettoyage et de prétraitement des données, de Data Visualisation et de se familiariser avec les algorithmes de Machine Learning.
Au-delà des compétences techniques acquises, ce projet a révélé le potentiel de l'analyse de données pour apporter des solutions concrètes à des problèmes sociétaux importants, comme la sécurité routière.
Ce projet a été une combinaison réussie de travail individuel et collectif, où chaque membre de l'équipe a pu exprimer son potentiel tout en contribuant à l'objectif commun. ''')
  st.write('''Ce projet a été une étape clé dans notre parcours d'équipe de Data Analysts, et nous sommes tous enthousiastes à l'idée d'appliquer les connaissances et les compétences acquises ici à de futurs défis en entreprise.''')
from flask import Flask, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# --- Questions et réponses initiales ---
question_initiales = [
    "Vous recevez une tâche complexe dans un projet d'équipe. Quelle étape vous semble la plus engageante ?",
    "Face à une panne critique dans un système informatique, quelle action vous semble la plus naturelle ?",
    "Imaginez que vous dirigez un atelier pour former vos collègues. Quel sujet choisiriez-vous ?"
]

reponse_initiales = [
    [
        "Configurer des systèmes et résoudre des problèmes techniques pour assurer le bon fonctionnement.",
        "Analyser et interpréter des données pour proposer une solution basée sur des faits.",
        "Identifier les failles potentielles et protéger les systèmes contre les menaces.",
        "Concevoir et coder une fonctionnalité qui améliore le produit ou le service."
    ],
    [
        "Écrire ou ajuster un programme pour corriger ou prévenir le problème.",
        "Vérifier si une intrusion ou une menace de sécurité est en cause, et renforcer la protection.",
        "Diagnostiquer le système et intervenir rapidement pour rétablir son fonctionnement.",
        "Collecter des données pour comprendre l'origine du problème et identifier des tendances."
    ],
    [
        "Apprendre à reconnaître et contrer les cyberattaques ou vulnérabilités.",
        "Former sur les meilleures pratiques pour gérer des infrastructures complexes.",
        "Démontrer comment créer une application performante ou automatiser un processus.",
        "Enseigner comment analyser des données pour résoudre des problèmes métier."
    ]
]

# Génération du questionnaire initial avec les réponses
questions_with_responses = [
    {"question": question_initiales[i], "responses": reponse_initiales[i]}
    for i in range(len(question_initiales))
]
# Questionnaire pour Software Development
questionnaire1 = [
    "Quelle est votre expérience avec le développement logiciel ?",
    "Quelles technologies utilisez-vous le plus fréquemment ?",
    "Quel aspect du développement logiciel préférez-vous ?",
    "Avez-vous de l'expérience avec les méthodologies de développement Agile ?",
    "À quel niveau êtes-vous familier avec les pratiques de test de logiciels ?",
    "Travaillez-vous souvent sur des projets en équipe ?",
    "Quelle est votre expérience avec les architectures modernes (microservices, cloud-native) ?"
]

reponses1 = [
    [
        "Aucune expérience",
        "Débutant (moins de 2 ans)",
        "Intermédiaire (2-5 ans)",
        "Avancé (plus de 5 ans)"
    ],
    [
        "Frameworks front-end (React, Angular)",
        "Langages back-end (Java, C#, Python)",
        "Développement mobile (Swift, Kotlin)",
        "DevOps et automatisation (Docker, Kubernetes)"
    ],
    [
        "Conception de l'interface utilisateur (UI/UX)",
        "Implémentation de la logique métier",
        "Optimisation des performances et sécurité",
        "Déploiement et intégration continue"
    ],
    [
        "Oui, Scrum",
        "Oui, Kanban",
        "Oui, d'autres méthodologies",
        "Non, aucune expérience"
    ],
    [
        "Tests unitaires",
        "Tests d'intégration",
        "Tests de performance",
        "Aucun type de test"
    ],
    [
        "Oui, avec des outils de collaboration (Git, Jira)",
        "Oui, mais rarement avec des outils de collaboration",
        "Non, surtout des projets individuels",
        "Parfois, cela dépend du projet"
    ],
    [
        "Avancé, je travaille régulièrement avec ces architectures",
        "Intermédiaire, j'ai participé à quelques projets",
        "Débutant, je commence à les apprendre",
        "Aucune expérience"
    ]
]

# Questionnaire pour Data Science
questionnaire2 = [
    "Avez-vous une préférence pour l'analyse de données structurées (tableaux) ou non structurées (images, texte) ?",
    "Quelle est votre expérience avec les bibliothèques de machine learning (comme TensorFlow, scikit-learn) ?",
    "Préférez-vous travailler sur la visualisation de données ou sur l'entraînement de modèles prédictifs ?",
    "À quel point êtes-vous à l'aise avec les langages de programmation pour les données (Python, R) ?",
    "Avez-vous de l'expérience dans le nettoyage et la préparation des jeux de données ?",
    "Utilisez-vous souvent les techniques de traitement de données massives (Big Data) ?",
    "À quel point êtes-vous familier avec les outils de visualisation de données (Matplotlib, Tableau) ?"
]

reponses2 = [
    ["Données structurées uniquement", 
     "Principalement des données structurées, mais je suis ouvert aux données non structurées", 
     "Principalement des données non structurées, mais je suis ouvert aux données structurées", 
     "Données non structurées uniquement", 
     "Pas de préférence, je suis à l'aise avec les deux"],

    ["Aucune expérience, je n'ai jamais utilisé ces bibliothèques", 
     "Débutant, j'ai utilisé quelques bibliothèques pour des projets simples", 
     "Intermédiaire, j'ai travaillé sur plusieurs projets incluant du machine learning", 
     "Avancé, je les utilise régulièrement pour des projets complexes"],

    ["Je préfère uniquement la visualisation de données", 
     "Principalement la visualisation, mais je m'intéresse aux modèles prédictifs", 
     "Principalement l'entraînement de modèles prédictifs, mais je fais parfois de la visualisation", 
     "J'aime les deux aspects de manière égale"],

    ["Pas du tout à l'aise, je n'ai jamais utilisé ces langages", 
     "Débutant, je connais les bases", 
     "Intermédiaire, j'ai une certaine expérience en développement avec ces langages", 
     "Avancé, je maîtrise bien ces langages et les utilise régulièrement"],

    ["Non, je n'ai jamais travaillé sur le nettoyage de données", 
     "J'ai une expérience limitée dans le nettoyage des données", 
     "J'ai souvent travaillé sur la préparation de données pour mes projets", 
     "Je suis très expérimenté, c'est une partie régulière de mon travail"],

    ["Non, je n'ai jamais travaillé avec des techniques de Big Data", 
     "J'ai une connaissance de base, mais je ne les utilise pas régulièrement", 
     "J'utilise occasionnellement des techniques de Big Data pour certains projets", 
     "J'utilise fréquemment les techniques de Big Data dans mes projets"],

    ["Pas familier du tout, je n'ai jamais utilisé ces outils", 
     "Débutant, j'ai utilisé ces outils pour des visualisations simples", 
     "Intermédiaire, je les utilise régulièrement pour des projets divers", 
     "Avancé, je maîtrise bien ces outils et je les utilise pour créer des visualisations"]
]

# Questionnaire pour System Administration
questionnaire3 = [
    "Préférez-vous travailler avec les systèmes d'exploitation Linux ou Windows ?",
    "Avez-vous de l'expérience avec les outils de virtualisation (comme VMware, VirtualBox) ?",
    "À quel point êtes-vous à l'aise avec l'administration de serveurs web (Apache, Nginx) ?",
    "Avez-vous déjà configuré des systèmes de sauvegarde et de restauration ?",
    "Avez-vous de l'expérience avec les scripts d'automatisation (Bash, PowerShell) ?",
    "Avez-vous travaillé avec des systèmes de surveillance et de journalisation (Nagios, ELK) ?",
    "Quelle est votre expérience avec les environnements cloud (AWS, Azure) ?"
]

reponses3 = [
    ["Principalement Linux, je le trouve plus flexible", 
     "Principalement Windows, j'ai plus d'expérience avec ce système", 
     "J'aime les deux, cela dépend du projet", 
     "Aucune préférence, je suis à l'aise avec les deux"],

    ["Oui, je les utilise régulièrement", 
     "Oui, mais seulement pour des tests occasionnels", 
     "J'ai une connaissance de base, mais je ne les utilise pas souvent", 
     "Non, je n'ai aucune expérience avec ces outils"],

    ["Très à l'aise, je les configure et les administre régulièrement", 
     "Assez à l'aise, j'ai configuré quelques serveurs web", 
     "Je suis débutant, j'ai seulement une expérience limitée", 
     "Pas du tout à l'aise, je n'ai jamais administré de serveurs web"],

    ["Oui, je les configure et les gère régulièrement", 
     "Oui, mais uniquement pour des projets spécifiques", 
     "J'ai une connaissance de base, mais je n'ai pas beaucoup d'expérience pratique", 
     "Non, je n'ai jamais configuré de systèmes de sauvegarde"],

    ["Oui, j'écris des scripts d'automatisation régulièrement", 
     "Oui, mais uniquement pour des tâches occasionnelles", 
     "J'ai écrit quelques scripts simples, mais je ne suis pas très expérimenté", 
     "Non, je n'ai aucune expérience avec les scripts d'automatisation"],

    ["Oui, je les utilise régulièrement pour surveiller les systèmes", 
     "Oui, mais uniquement pour quelques projets spécifiques", 
     "J'ai une connaissance de base, mais je ne les utilise pas souvent", 
     "Non, je n'ai jamais travaillé avec ces outils"],

    ["Avancé, je travaille régulièrement sur ces plateformes", 
     "Intermédiaire, j'ai travaillé sur quelques projets cloud", 
     "Débutant, j'ai seulement une connaissance de base", 
     "Aucune expérience, je n'ai jamais travaillé avec le cloud"]
]

# Questionnaire pour Cybersecurity
questionnaire4 = [
    "Quelle est votre expérience en matière d'évaluation des vulnérabilités et de tests d'intrusion ?",
    "À quel point êtes-vous familier avec les principes de cryptographie ?",
    "Avez-vous déjà configuré des pare-feu ou d'autres dispositifs de sécurité réseau ?",
    "Avez-vous de l'expérience avec les audits de sécurité et la conformité (ISO 27001, GDPR) ?",
    "Avez-vous utilisé des outils de détection d'intrusion (comme Snort) ou d'analyse de malwares ?",
    "Quelle est votre connaissance des attaques courantes (phishing, injection SQL) et des mesures de prévention ?",
    "Avez-vous déjà travaillé sur la sécurité des applications (OWASP, sécurité des API) ?"
]

reponses4 = [
    ["Avancé, je réalise régulièrement des tests d'intrusion professionnels", 
     "Intermédiaire, j'ai effectué quelques évaluations de vulnérabilités", 
     "Débutant, j'ai seulement une expérience théorique ou limitée", 
     "Aucune expérience, je n'ai jamais effectué de tests d'intrusion"],

    ["Très familier, je les utilise régulièrement dans mon travail", 
     "Assez familier, j'ai une bonne compréhension théorique et pratique", 
     "Débutant, j'ai une connaissance de base des concepts", 
     "Pas du tout familier, je n'ai jamais travaillé avec la cryptographie"],

    ["Oui, je les configure et les administre régulièrement", 
     "Oui, mais uniquement pour des projets spécifiques", 
     "J'ai une connaissance de base et j'ai configuré quelques dispositifs", 
     "Non, je n'ai jamais configuré de dispositifs de sécurité réseau"],

    ["Oui, j'ai mené plusieurs audits de sécurité et conformité", 
     "Oui, j'ai participé à quelques audits, mais je n'en suis pas le responsable principal", 
     "J'ai une connaissance de base des normes, mais peu d'expérience pratique", 
     "Non, je n'ai aucune expérience avec les audits de sécurité"],

    ["Oui, je les utilise régulièrement dans mes tâches de sécurité", 
     "Oui, mais uniquement à des fins d'apprentissage ou pour des projets limités", 
     "J'ai une connaissance de base, mais je ne les utilise pas fréquemment", 
     "Non, je n'ai jamais utilisé ces outils"],

    ["Avancé, je suis capable de les identifier et de mettre en œuvre des mesures de prévention", 
     "Intermédiaire, j'ai une bonne compréhension théorique et pratique de ces attaques", 
     "Débutant, je connais les concepts de base mais peu de techniques de prévention", 
     "Aucune connaissance, je ne suis pas familier avec ces attaques"],

    ["Oui, j'ai une expérience approfondie dans l'application des pratiques OWASP", 
     "Oui, j'ai travaillé sur quelques projets de sécurité des applications", 
     "J'ai une connaissance de base des concepts, mais peu d'expérience pratique", 
     "Non, je n'ai jamais travaillé sur la sécurité des applications"]
]



# --- Fonction de génération du questionnaire ---
def generate_questionnaire(domain):
    if domain == "Software Development":
        return questionnaire1, reponses1
    elif domain == "Data Science":
        return questionnaire2, reponses2
    elif domain == "System Administration":
        return questionnaire3, reponses3
    elif domain == "Cybersecurity":
        return questionnaire4, reponses4
    else:
        return [], []  # Si un domaine invalide est fourni

# --- Classification pour prédire le domaine d'intérêt ---
# Données d'entraînement
X = [
    [3, 0, 2],  # Exemple pour "Software Development"
    [1, 3, 3],  # Exemple pour "Data Science"
    [0, 2, 1],  # Exemple pour "System Administration"
    [2, 1, 0],  # Exemple pour "Cybersecurity"
]
y = [0, 1, 2, 3]  # Labels pour les domaines d'intérêt
domains = ["Software Development", "Data Science", "System Administration", "Cybersecurity"]

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèle k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
#envoyer le formulaire de 3question 
# --- Routes API ---
@app.route('/api/questions', methods=['GET'])
def get_questions():
    """Retourne les questions initiales sous forme de JSON."""
    return jsonify(questions_with_responses)
#bach yjini mel front end resultat de reponses
user_responses = [3, 3, 1]

#retourner les reponses de questionnaire 
def predict_domain( user_responses ):
    """
    Prévoit le domaine d'intérêt basé sur les réponses de l'utilisateur.
    Exemple de réponse utilisateur : [3, 3, 1].
    """
    # Exemple de réponses utilisateur (à recevoir dans la requête POST)

    user_responses_scaled = scaler.transform([user_responses])
    predicted_domain_index = knn.predict(user_responses_scaled)[0]
    predicted_domain = domains[predicted_domain_index]

    return predicted_domain

# --- Route pour lancer le questionnaire ---
@app.route('/start_questionnaire', methods=['POST'])
def start_questionnaire():
    # Exemple de données d'entrée, ces données seraient fournies par l'utilisateur
    #user_responses = request.json.get('user_answers', [])  # Récupérer les réponses
    predicted_domain = predict_domain(user_responses)  # Prédire le domaine

    # Générer les questions et réponses en fonction du domaine prédit
    questions, responses = generate_questionnaire(predicted_domain)

    return jsonify({
        "predicted_domain": predicted_domain,
        "questions": questions,
        "responses": responses
    })
# --- Lancement de l'application ---
if __name__ == '__main__':
    app.run(debug=True)

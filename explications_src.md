# Explications de l'architecture source (`src/`)

Ce document détaille le rôle de chaque classe et de chaque fonction présentes dans le répertoire `src/` constituant le cœur de l'algorithme GP-MOE (Gaussian Process Mixture-of-Experts) implémenté pour l'apprentissage en ligne.

---

## 1. `src/data_stream.py`
Ce fichier gère la simulation d'un flux de données arrivant de manière séquentielle (une par une), essentiel pour tester des algorithmes d'apprentissage en ligne.

* **Classe `DataStreamer`** : Transforme un jeu de données statique complet en un générateur séquentiel.
    * `__init__(self, X, y)` : Initialise le flux avec l'ensemble des caractéristiques `X` et la cible `y`. Instancie également les paramètres pour la standardisation.
    * `standardize(self)` : Calcule la moyenne et l'écart-type originaux de l'ensemble des données, puis centre et réduit (moyenne 0, variance 1) les jeux `X` et `y`. Les projections futures utiliseront ces métriques.
    * `__iter__(self)` : Rend la classe itérable pour pouvoir utiliser une trame `for (x, y) in streamer:`.
    * `__next__(self)` : Retourne la prochaine observation $(x_i, y_i)$ disponible. Si les données ont été standardisées, retourne les versions transformées. Lève une erreur `StopIteration` quand toutes les données ont été lues.
    * `reset(self)` : Remet l'index du flux à zéro pour recommencer la lecture depuis le début.

---

## 2. `src/kernel.py`
Ce module est responsable des opérations de covariance mathématique fondamentales pour les Processus Gaussiens.

* `rbf_kernel(X1, X2, theta)` : Code la fonction mathématique du **noyau RBF** (Radial Basis Function). Calcule de manière vectorisée et performante la matrice de covariance basée sur la distance Euclidienne au carré entre tous les points de `X1` et `X2`, mis à l'échelle par le vecteur des longueurs d'échelle `theta`.
* `compute_covariance(X, theta, sigma_sq)` : Calcule la matrice de covariance complète et robuste utilisée pour l'entraînement d'un cluster. Elle appelle `rbf_kernel(X, X, theta)` puis ajoute le bruit blanc observationnel $\sigma^2 I$ (via `sigma_sq * np.eye(N)`) sur la diagonale principale.

---

## 3. `src/gp_expert.py`
Ce fichier définit un expert local. Dans le cadre de ce modèle de mélange, un expert équivaut à un et un seul "cluster" de données modélisé par un Processus Gaussien indépendant.

* **Classe `GPExpert`**
    * `__init__(self, expert_id, D, prior_mean, prior_cov)` : Crée un nouveau cluster vide. Il tire immédiatement une configuration initiale aléatoire pour ses hyperparamètres ($\theta$, $\sigma^2$) depuis une distribution `log-normale` basée sur les priors moyens et covariances a priori donnés en argument.
    * `add_observation(self, x_i, y_i)` : Stocke en mémoire une nouvelle observation assignée à ce cluster particulier.
    * `get_X(self)` / `get_y(self)` : Retournent les observations de ce cluster sous forme de tableau `Numpy`.
    * `N` (Propriété dynamique) : Raccourci retournant le nombre d'observations actuellement affectées à cet expert.
    * `marginal_likelihood(self, theta, sigma_sq, B=None)` : Fonction mathématiquement lourde qui calcule la **log-vraisemblance marginale complète** (ou son approximation stochastique par sous-échantillonnage (minibatch) si le point limite `B` est renseigné). Permet de dire "à quel point ces paramètres expliquent bien les données du cluster".
    * `update_hyperparameters(self, B=None)` : Exécute l'algorithme MCMC d'**Elliptical Slice Sampling (ESS)**. Au lieu d'accepter aveuglément ou de rejeter comme Metropolis-Hastings (MH), le processus *slice* l'espace elliptique continu à la recherche d'hyperparamètres qui améliorent la `marginal_likelihood` actuelle. L'état interne `theta` et `sigma_sq` de l'expert est alors mis à jour à l'issue de cet appel.

---

## 4. `src/crp.py`
S'occupe des statistiques Bayésiennes non paramétriques pour affecter (router) les nouveaux points de données aux divers experts.

* `multivariate_t_distribution(x, mu, Sigma, nu)` : Calcule numériquement (et de manière stable via `log_pdf`) la valeur de la densité de probabilité d'un point `x` sous la distribution T-Multivariée de Student (qui résulte de la marginalisation du prior Normal-Inverse-Wishart).
* **Classe `CRP` (Chinese Restaurant Process)** : Le CRP stipule qu'un nouveau client (point) a plus de chances de s'attabler à une grande table (cluster prédominant) et une petite chance constante (paramètre concentration $\alpha$) d'ouvrir une nouvelle table (nouvel expert).
    * `__init__(self, alpha, mu_0, kappa_0, nu_0, Psi_0)` : Paramètre les hyperparamètres du processus et la mesure de base.
    * `_compute_posterior_predictive(self, X_k)` : Méthode privée (interne). Étant donné les points actuellement autour d'une table `X_k`, met à jour les paramètres de Student-t marginaux (moyenne $\mu_N$, position $\Psi_N$, degrés de liberté $df$) résumant leurs géométries dans l'espace.
    * `compute_assignment_probabilities(self, x_i, clusters_X)` : Donne un nouveau point spatial `x_i` et vérifie son adéquation avec tous les clusters existants (probabilité qu'il soit généré par eux via les calculs prédictifs) ainsi qu'avec un éventuel tout nouveau cluster généré par le prior vide pondéré par la concentration $\alpha$. Elle retourne ce vecteur de scores pour l'échantillonnage proportionnel.

---

## 5. `src/smc_sampler.py`
Le noyau de la modélisation conditionnelle séquentielle. Il opère des dizaines de "mondes parallèles" ou "hypohèses" d'où l'aspect "Multi-Particules".

* **Classe `SMCParticle`** : Représente UN chemin possible d'exploration.
    * `__init__(...)` : Crée une particule avec son propre dictionnaire privé d'experts (`self.experts`) et son propre traceur de processus de restaurant `crp`.
    * `_get_clusters_X(self)` : Extracteur listant les positions attributaires assignées actuellement dans tous les experts de cette particule.
    * `process_observation(self, x_i, y_i, ...)` : C'est le cœur itératif pour une hypothèse. 1) Obtient les probabilités du `CRP` pour ce point. 2) Tire un jeton aléatoire basé sur ces biais (`np.random.choice(...)`) pour l'assigner à l'expert $z_i$. 3) S'il tombe sur le cluster fantôme, instancie un nouveau `GPExpert` vierge. 4) Demande à ce cluster d'intégrer le point, puis déclenche son auto-optimisation RBF `update_hyperparameters()`. 5) Ajuste le poids $w_i^{(j)}$ de confiance en cette particule par rapport aux autres en évaluant un ratio de vraisemblances marginales (Avant/Après l'insertion du point).
* **Classe `SMCSampler`** : Le chef d'orchestre des particules.
    * `__init__(self, J, ...)` : Initialise une volière de $J$ particules indépendantes ayant chacune un poids initial équitable.
    * `normalize_weights(self)` : Applique l'exponentiation (Astuce de soustraction de la probabilité Max pour ne pas diviser par zéro) pour transformer la somme des sous-scores logorithmiques des particules afin que la somme de tous les poids des particules crées vaille $1.0$ exactement.
    * `compute_N_eff(self)` : Calcule la taille d'échantillon active (Effective Sample Size). Si beaucoup de particules ont un poids quasi nul parce qu'elles ont fait de mauvais choix d'assignations, cet algorithme retournera un chiffre très bas.
    * `resample(self)` : Évite la Dégénérescence Particulaire. Si $N_{eff} < \frac{J}{2}$, cette méthode tue les particules aux scores les plus bas en clonant et copiant à l'identique les particules dominantes ayant fait les meilleurs choix de routage (via tirage de loi Multinomiale des indices), puis réinitialise les poids de manière uniforme.
    * `update(self, x_i, y_i, ...)` : Diffuse l'information du nouveau point à toutes ses particules puis applique les phases de normalisation et d'évaluation au ré-échantillonnage d'office.

---

## 6. `src/gp_moe.py`
La façade (API Publique de haut niveau). Un utilisateur final du modèle depuis le dossier d'expérimentation n'interagit de fait qu'avec ce fichier masquant l'orchestration interne.

* **Classe `GPMoE`**
    * `__init__(self, D, J=20, ...)` : Permet de configurer les variables globales et crée le `SMCSampler` interne prêt à recevoir le flux. Par défaut, pré-configure convenablement les distributions a priori et prend en compte la limite de batch `B` optionnelle.
    * `update(self, x_i, y_i, update_hyperparams=True)` : Méthode d'ingestion séquentielle (Streaming Ingestion). Pousse simplement la donnée descendante vers le `smc.update(...)` sous-jacent.
    * `predict(self, x_test)` : Calcule les prévisions d'un nouveau point inconnu `x_test`. L'algorithme opère une boucle complexe :
        1. Pour chaque particule (pondérée par sa pertinence globale `w`),
        2. Évalue à quelle probabilité fractionnelle `x_test` appartiendrait à n'importe quel de ses $K$ experts.
        3. Fait calculer par ces experts les équations stochastiques d'**inférences Gaussiennes** exactes (calcul du vecteur $k^*$ et matrice $K^{-1}$) pour la moyenne et l'intervalle de variance prédictive sur ce point dans leur région.
        4. Mélange les valeurs d'espérance selon le théorème des Lois Totales Pondérées par Mélange fini au sein d'une particule, puis mélange macroscopiquement ces résultats entre toutes les particules du filtre SMC pour fournir le duo final pondéré ultime et robuste `(moyenne_pred, variance_pred)`.

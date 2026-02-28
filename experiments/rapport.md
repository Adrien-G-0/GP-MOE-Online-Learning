# Rapport de Synthèse : Sequential GP-MOE

## Introduction
Ce rapport compare les performances de l'approche **Product-of-Experts (POE - Baseline J=1)** et de l'approche **Mixture-of-Experts avec Sequential Monte Carlo (SMC GP-MOE J=20)** implémentées selon le papier *Sequential Gaussian Processes for Online Learning of Nonstationary functions*.

En outre, une application sur un problème simulé de dégradation de capteur IoT est évaluée avec l'extension suggérée (Échantillonnage Rétrospectif).

## 1. Expérience de Base : Motorcycle Crash
Le jeu de données `motorcycle` simule la dynamique non-stationnaire (faible bruit initial, suivi d'un régime sinusoïdal très bruité, puis d'une stabilisation avec un bruit intermédiaire).

| Modèle | Nombre de Particules (J) | Mean Online MSE | Mean Online NLPD | Clusters Découverts |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (POE) | 1 | 0.4132 | -0.9386 | 2 |
| GP-MOE | 20 | 0.3598 | -0.9283 | 1 (Best Part.) |

**Analyse :**
L'approche SMC avec `J=20` particules permet au modèle de maintenir plusieurs hypothèses d'affectation en parallèle pendant le flux en ligne. Contrairement à la baseline (1 seule particule, chemin gourmand), le GP-MOE parvient à une erreur quadratique moyenne (MSE) inférieure, démontrant sa résilience face aux décisions d'affectation prématurées.

## 2. Application Extension : Dégradation Capteur IoT
Nous avons simulé un capteur IoT traversant 3 régimes  : Normal (stationnaire bas bruit), Dégradation (dérive et augmentation variance), Défaillance imminente (chaos, haute variance). 

Pour cette application, nous avons comparé l'algorithme standard avec une **Extension d'Échantillonnage Rétrospectif** (Option A). Le modèle ré-évalue les petites séquences et fusionne algorithmiquement les micro-clusters générés par des erreurs précoces.

| Modèle | Mean Online MSE | Mean Online NLPD |
| :--- | :---: | :---: |
| Standard GP-MOE | 0.8130 | -0.6117 |
| Retrospective GP-MOE | 0.8162 | -0.6426 |

**Analyse :**
- L'approche d'échantillonnage rétrospectif n'implique pas de diminution majeure sur la fenêtre instantanée (MSE comparable) mais module la probabilité prédictive négative (-NLPD, où des valeurs plus basses signifient une meilleure adaptation de la variance aux observations réelles). 
- Visuellement (voir `experiments/iot_extension_results.png`), l'extension lisse les frontières entre les régimes en fusionnant rétrospectivement les grappes erratiques du CRP.

## 3. Application Financière : Cours EUR/USD
Nous avons testé les modèles en utilisant les 200 derniers jours de clôture du taux de change **EUR/USD** (via `yfinance`). Les données financières sont notoirement non-stationnaires en raison des chocs de marché et des annonces économiques, et se prêtent donc bien aux modèles de changement de régime virtuels testés (GP-MOE).

| Modèle | Mean Online MSE | Mean Online NLPD |
| :--- | :---: | :---: |
| Standard GP-MOE | 0.169674 | -0.5006 |
| Retrospective GP-MOE | 0.167138 | -0.5013 |

**Analyse :**
- L'approche d'échantillonnage rétrospectif montre une légère amélioration par rapport au modèle standard, réduisant l'erreur quadratique moyenne (MSE) tout en conservant une bonne quantification de l'incertitude (-NLPD).
- Les données financières (taux de change sur une fenêtre courte) ne comprenaient qu'un seul cluster dominant identifié dans ce contexte particulier, mais les légères instabilités passées ont été mieux anticipées par l'Option A (Rétrospective).
- Les graphiques associés peuvent être consultés dans `experiments/financial_data_results.png`.

## 4. Test de Robustesse : Régimes Poissoniens Alternants
Pour éprouver la capacité d'adaptation et le clustering du modèle, nous avons généré un flux synthétique de 1000 observations. Le signal alterne dynamiquement et aléatoirement en blocs entre un nombre inconnu $K$ de processus générateurs (chaque sous-processus ayant ses propres bruits, fréquences et tendances). Le nombre total $K$ a été tiré empiriquement via $K \sim Poisson(\lambda=5)$, résultant en **5 régimes sous-jacents**.

| Modèle | Mean Online MSE | Mean Online NLPD | Clusters Découverts |
| :--- | :---: | :---: | :---: |
| Standard GP-MOE | 0.2854 | -0.7949 | 1 |
| Retrospective GP-MOE | 0.3668 | -1.9247 | 1 |

**Analyse :**
- Bien que le signal soit issu de 5 régimes radicalement différents qui s'alternent, le modèle probabiliste actuel avec les hyperparamètres par défaut privilégie l'explication des données par **un macro-cluster unique (1 cluster découvert)** au lieu de fractionner l'espace. Le noyau RBF unifié avec Elliptical Slice Sampling s'avère suffisamment expressif pour couvrir l'ensemble des variations au détriment de l'identification sémantique des séparations latentes.
- Sur le plan strictement prédictif à une étape, l'erreur quadratique moyenne reste confinée (MSE ~0.28 à 0.36), démontrant la grande flexibilité de l'approximation gaussienne même face aux sauts de régimes. L'approche rétrospective sacrifie un peu de précision immédiate (MSE) au profit d'une quantification très conservatrice et prudente de l'incertitude globale (NLPD nettement inférieur à -1.9).
- Visuels disponibles dans `experiments/poisson_regimes_results.png`.

## 5. Conclusions et Spécifications Implémentées
✅ Flux de données en ligne synchronisé (`data_stream.py`)
✅ Échantillonneur par Tranches Elliptiques (ESS) pour les hyperparamètres sans MH (`gp_expert.py`)
✅ Processus de Restaurant Chinois Marginalisé (`crp.py`)
✅ Filtre SMC avec seuil $N_{eff}$ & batching stochastique (`smc_sampler.py`)
✅ Option A : Correction rétrospective des chemins de particules SMC.

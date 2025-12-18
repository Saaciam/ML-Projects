# Churn Prediction (Telco) — Classification binaire end-to-end

Ce projet construit un modèle de Machine Learning capable de **prédire le churn client** (résiliation) afin d’aider une entreprise à **anticiper les départs** et déclencher des actions de rétention (offres, appels, support). :contentReference[oaicite:3]{index=3}

> 🎯 Idée clé : il est souvent **plus coûteux** de perdre un client que de contacter un client fidèle “pour rien”.
> Dans ce projet, on priorise donc la réduction des **faux négatifs (FN)**, c’est-à-dire les churners ratés. :contentReference[oaicite:4]{index=4}

---

## 🧠 Problème ML

- Type : **apprentissage supervisé**
- Tâche : **classification binaire**
- Cible :
  - `0` = No churn
  - `1` = Churn :contentReference[oaicite:5]{index=5}

---

## 🔧 Pipeline du projet (bonnes pratiques)

1. **Chargement & exploration**
2. **Préparation de la cible** (`Churn` → 0/1) + suppression de l’identifiant (`customerID`)
3. **Nettoyage**
   - conversion `TotalCharges` en numérique
   - suppression/gestion des valeurs manquantes
4. **Split Train / Validation / Test (70/15/15) stratifié**
   - objectif : évaluation honnête + mêmes proportions de churn dans chaque sous-ensemble :contentReference[oaicite:6]{index=6}
5. **Prétraitement**
   - numériques : `StandardScaler`
   - catégorielles : `OneHotEncoder(handle_unknown="ignore")`
6. **Modèle baseline**
   - Régression Logistique (`class_weight="balanced"`)
7. **Évaluation (validation) + ajustement du seuil**
8. **Évaluation finale (test)**
   - le test n’est utilisé qu’une fois, à la fin :contentReference[oaicite:7]{index=7}

---

## 📏 Évaluation : pourquoi ces métriques ?

### Matrice de confusion (TN, FP, FN, TP)
Elle montre *où le modèle se trompe* :
- **FN** : churners ratés (erreur la plus grave ici)
- **FP** : clients contactés inutilement (coût marketing) :contentReference[oaicite:8]{index=8}

### Recall vs Precision
- **Recall(1) = TP / (TP + FN)**  
  → “Parmi les churners réels, combien j’en détecte ?”  
  ✅ métrique prioritaire ici (FN coûteux) :contentReference[oaicite:9]{index=9}

- **Precision(1) = TP / (TP + FP)**  
  → “Parmi ceux que je cible, combien churnent vraiment ?”  
  utile pour contrôler le volume d’actions marketing :contentReference[oaicite:10]{index=10}

### Seuil de décision
Le modèle renvoie une probabilité ; le **seuil** transforme cette proba en classe (0/1).  
Baisser le seuil → **recall ↑** (FN ↓) mais **FP ↑** (precision ↓). :contentReference[oaicite:11]{index=11}

---

## ✅ Résultats (seuil sélectionné = 0.4)

Matrice de confusion sur le jeu de test :

|              | Prédit 0 | Prédit 1 |
|--------------|----------|----------|
| Réel 0       | TN=474   | FP=301   |
| Réel 1       | FN=34    | TP=246   |

Interprétation (impact métier) :
- on accepte davantage de FP pour **rater moins de churners**,
- ici : **34 churners ratés** seulement (FN), au prix de 301 faux positifs (FP). :contentReference[oaicite:12]{index=12}

---

## ▶️ Installation & exécution

1) Cloner le repo
```bash
git clone <URL_DU_REPO>
cd churn-prediction-telco
```
2) Créer un environnement + installer les dépendances
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

3) Lancer
python main.py



### les imports
import pandas as pd
from sklearn.model_selection import train_test_split
import os

### Chargement des données
PATH = "D:\portfolio-projects\churn\data\Telco-Customer-Churn.csv"
df = pd.read_csv(PATH)

print(df.shape) ## permet de savoir combien de lignes et de colonnes
print(df.columns) ## permet de savoir les variables disponibles
df.head()

### Vérifier la repartition de la variable cible
df["Churn"].value_counts()
print(df["Churn"].value_counts(normalize=True)) ## permet de voir si les classes sont équilibrées

### préparation des données - nettoyage - splitting
#### préparation des données
df = df.copy()

y = df["Churn"].map({"No": 0, "Yes": 1})  ## transformation de la variable cible en binaire

X = df.drop(columns=["customerID", "Churn"])  ## suppression des colonnes inutiles

#### nettoyage des données
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")  ## transformation en numérique
print("NaN TotalCharges:", X["TotalCharges"].isna().sum())  ## compter les NaN

#### suppression des lignes avec des NaN
mask = X["TotalCharges"].notna()
X = X.loc[mask]
y = y.loc[mask]


#### splitting ( train/val/test - 70/15/15 )

from sklearn.model_selection import train_test_split
RANDOM_STATE = 42

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)


val_size = 0.15 /0.85  ## ajustement de la taille de validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_size, stratify= y_trainval ,random_state=RANDOM_STATE)

print("Train:", X_train.shape, y_train.mean())
print("Val:", X_val.shape, y_val.mean())
print("Test:", X_test.shape, y_test.mean())

### identification des colonnes catégorielles et numériques
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

print("Nb colonnes catégorielles:", len(cat_cols))
print("Nb colonnes numériques:", len(num_cols))
print("Exemples catégorielles:", cat_cols[:5])
print("Exemples numériques:", num_cols[:5])


### Prétraitement des données
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

### Entraînement d'un modèle de base - régression logistique
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model",LogisticRegression(max_iter=1000, class_weight="balanced") )])

clf.fit(X_train, y_train)

### Evaluation du modèle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Probabilités de churn (classe 1)
y_val_proba = clf.predict_proba(X_val)[:, 1]

# Prédiction avec seuil 0.5 (baseline)
y_val_pred = (y_val_proba >= 0.5).astype(int)

cm = confusion_matrix(y_val, y_val_pred)
print("Matrice de confusion (validation) :\n", cm)

print("\nRapport (validation) :\n", classification_report(y_val, y_val_pred, digits=3))
print("ROC-AUC (validation):", roc_auc_score(y_val, y_val_proba))


from sklearn.metrics import confusion_matrix, precision_score, recall_score

thresholds = [0.5, 0.4, 0.3]
for t in thresholds:
    y_pred_t = (y_val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_t).ravel()
    prec = precision_score(y_val, y_pred_t)
    rec = recall_score(y_val, y_pred_t)
    print(f"Seuil={t} | TN={tn} FP={fp} FN={fn} TP={tp} | precision={prec:.3f} recall={rec:.3f}")


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

THRESH = 0.4

y_test_proba = clf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= THRESH).astype(int)

cm_test = confusion_matrix(y_test, y_test_pred)
print("Matrice de confusion (TEST) :\n", cm_test)

print("\nRapport (TEST) :\n", classification_report(y_test, y_test_pred, digits=3))
print("ROC-AUC (TEST):", roc_auc_score(y_test, y_test_proba))


import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/churn_model.joblib")

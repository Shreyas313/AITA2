import scipy.io
import numpy as np
import joblib
import os
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- project / data / model paths (changed) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
mat_path = os.path.join(data_dir, "url.mat")
svm_folder = os.path.join(data_dir, "url_svmlight")
models_dir = os.path.join(project_root, "models")
model_path = os.path.join(models_dir, "model.pkl")

# ---------------------------------------------------
# 1. LOAD DATA FROM url.mat
# ---------------------------------------------------
print(f"Loading MAT file: {mat_path} ...")
mat = scipy.io.loadmat(mat_path)

# Change these names ONLY if your MAT variables differ
X_mat = mat["X"]             # features matrix
y_mat = mat["y"].ravel()     # labels vector

print("Loaded MAT data:")
print(" → X_mat shape:", X_mat.shape)
print(" → y_mat shape:", y_mat.shape)

# ---------------------------------------------------
# 2. LOAD EVERY SVM-Light FILE IN url_svmlight/
# ---------------------------------------------------
print("\nLoading SVM-light files from folder: url_svmlight/ ...")

svm_folder = os.path.join(data_dir, "url_svmlight")
X_svm_list = []
y_svm_list = []

if not os.path.exists(svm_folder):
    raise FileNotFoundError(f"Folder not found: {svm_folder}")

for filename in os.listdir(svm_folder):
    if filename.endswith(".txt") or filename.endswith(".svm") or filename.endswith(".dat"):
        path = os.path.join(svm_folder, filename)
        print(" → Loading:", filename)

        X_svm, y_svm = load_svmlight_file(path)
        X_svm = X_svm.toarray()       # convert to dense
        y_svm = y_svm.ravel()

        X_svm_list.append(X_svm)
        y_svm_list.append(y_svm)

# Merge all SVM-light datasets
if len(X_svm_list) > 0:
    X_svm_all = np.vstack(X_svm_list)
    y_svm_all = np.hstack(y_svm_list)

    print("Loaded SVM-light data:")
    print(" → X_svm_all shape:", X_svm_all.shape)
    print(" → y_svm_all shape:", y_svm_all.shape)
else:
    print("WARNING: No SVM-light files found. Only MAT data will be used.")
    X_svm_all = np.empty((0, X_mat.shape[1]))
    y_svm_all = np.array([])

# ---------------------------------------------------
# 3. COMBINE MAT + SVM-LIGHT DATASETS
# ---------------------------------------------------
print("\nCombining MAT + SVM-light datasets...")

X_total = np.vstack([X_mat, X_svm_all])
y_total = np.hstack([y_mat, y_svm_all])

print("Final dataset:")
print(" → X_total shape:", X_total.shape)
print(" → y_total shape:", y_total.shape)

# ---------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_total, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 5. TRAIN SVM MODEL
# ---------------------------------------------------
print("\nTraining SVM model...")
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# ---------------------------------------------------
# 6. EVALUATE MODEL
# ---------------------------------------------------
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"SVM Accuracy: {acc * 100:.2f}%")

# ---------------------------------------------------
# 7. SAVE MODEL (changed)
# ---------------------------------------------------
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, model_path)
print(f"\nTraining complete! model saved to: {model_path}")
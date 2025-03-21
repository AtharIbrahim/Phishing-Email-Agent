from pyexpat import model

from train import X_test


accuracy = model.score(X_test, X_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
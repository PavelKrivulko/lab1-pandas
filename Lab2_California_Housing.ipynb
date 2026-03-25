{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5ab021a8-bab0-42b5-aa63-8cb71c2819c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.datasets import fetch_california_housing\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression\n",
    "from sklearn.metrics import *\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "045cc6cb-62c3-49c8-af40-009ad1c88b0e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Данные готовы: обучающая (16512, 8), тестовая (4128, 8)\n"
     ]
    }
   ],
   "source": [
    "# Загрузка\n",
    "housing = fetch_california_housing()\n",
    "df = pd.DataFrame(housing.data, columns=housing.feature_names)\n",
    "df['MedHouseVal'] = housing.target\n",
    "\n",
    "# Разделение\n",
    "X = df.drop('MedHouseVal', axis=1)\n",
    "y = df['MedHouseVal']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Нормализация\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Для классификации\n",
    "median = y_train.median()\n",
    "y_train_class = (y_train > median).astype(int)\n",
    "y_test_class = (y_test > median).astype(int)\n",
    "\n",
    "print(f\"Данные готовы: обучающая {X_train.shape}, тестовая {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d2c70598-75f2-40ef-a00e-3840511ede3a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ЛИНЕЙНАЯ РЕГРЕССИЯ\n",
      "MSE: 0.5559\n",
      "RMSE: 0.7456\n",
      "MAE: 0.5332\n"
     ]
    }
   ],
   "source": [
    "linear = LinearRegression()\n",
    "linear.fit(X_train_scaled, y_train)\n",
    "y_pred = linear.predict(X_test_scaled)\n",
    "\n",
    "print(\"ЛИНЕЙНАЯ РЕГРЕССИЯ\")\n",
    "print(f\"MSE: {mean_squared_error(y_test, y_pred):.4f}\")\n",
    "print(f\"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\")\n",
    "print(f\"MAE: {mean_absolute_error(y_test, y_pred):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9a322610-3dfa-4d5f-b079-6c5564174b7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RIDGE РЕГРЕССИЯ\n",
      "RMSE: 0.7456\n"
     ]
    }
   ],
   "source": [
    "ridge = Ridge(alpha=1.0)\n",
    "ridge.fit(X_train_scaled, y_train)\n",
    "y_pred_ridge = ridge.predict(X_test_scaled)\n",
    "\n",
    "print(\"RIDGE РЕГРЕССИЯ\")\n",
    "print(f\"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "2418ce31-66da-4ce2-8ce1-37ea9c981d4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "КЛАССИФИКАЦИЯ\n",
      "Accuracy: 0.8263\n",
      "Precision: 0.8259\n",
      "Recall: 0.8239\n",
      "F1-score: 0.8249\n"
     ]
    }
   ],
   "source": [
    "logreg = LogisticRegression(max_iter=1000)\n",
    "logreg.fit(X_train_scaled, y_train_class)\n",
    "y_pred_class = logreg.predict(X_test_scaled)\n",
    "\n",
    "print(\"КЛАССИФИКАЦИЯ\")\n",
    "print(f\"Accuracy: {accuracy_score(y_test_class, y_pred_class):.4f}\")\n",
    "print(f\"Precision: {precision_score(y_test_class, y_pred_class):.4f}\")\n",
    "print(f\"Recall: {recall_score(y_test_class, y_pred_class):.4f}\")\n",
    "print(f\"F1-score: {f1_score(y_test_class, y_pred_class):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "79e77491-6862-41fc-98cb-ff9c5462c7a1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test_class, y_pred_class)\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "plt.title('Матрица ошибок')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "850ac1ae-51ab-4115-b936-653f2c31e0f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "УЛУЧШЕННАЯ КЛАССИФИКАЦИЯ\n",
      "Accuracy: 0.8263\n",
      "Recall: 0.8239\n"
     ]
    }
   ],
   "source": [
    "logreg_bal = LogisticRegression(class_weight='balanced', max_iter=1000)\n",
    "logreg_bal.fit(X_train_scaled, y_train_class)\n",
    "y_pred_bal = logreg_bal.predict(X_test_scaled)\n",
    "\n",
    "print(\"УЛУЧШЕННАЯ КЛАССИФИКАЦИЯ\")\n",
    "print(f\"Accuracy: {accuracy_score(y_test_class, y_pred_bal):.4f}\")\n",
    "print(f\"Recall: {recall_score(y_test_class, y_pred_bal):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7f9933a2-4bb2-422d-82b9-b4c596dcd0fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "========================================\n",
      "ИТОГИ ЛАБОРАТОРНОЙ РАБОТЫ\n",
      "========================================\n",
      "1. Регрессия: RMSE = 0.7456\n",
      "2. Классификация: Accuracy = 0.8263\n",
      "3. Лучший признак: MedInc (доход)\n"
     ]
    }
   ],
   "source": [
    "print(\"=\"*40)\n",
    "print(\"ИТОГИ ЛАБОРАТОРНОЙ РАБОТЫ\")\n",
    "print(\"=\"*40)\n",
    "print(f\"1. Регрессия: RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}\")\n",
    "print(f\"2. Классификация: Accuracy = {accuracy_score(y_test_class, y_pred_class):.4f}\")\n",
    "print(f\"3. Лучший признак: MedInc (доход)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a2bd637-55ab-4b67-b8a4-23c8300ba05b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

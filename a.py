import streamlit as st
import numpy as np

class LinReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.rand(self.n_inputs) - 0.5
        self.intercept_ = np.random.rand()

    def fit(self, X, y, epochs=100):
        y = y.reshape(-1, 1)

        for i in range(epochs):
            predictions = self.predict(X).reshape(-1, 1)
            errors = predictions - y
            coef_gradients = np.mean(2 * errors * X, axis=0)
            intercept_gradient = np.mean(2 * errors, axis=0)

            self.coef_ -= self.learning_rate * coef_gradients
            self.intercept_ -= self.learning_rate * intercept_gradient

        self.mse = self.score(X, y.reshape(-1))

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2) ** 0.5

ml = LinReg(0.00001, 2)
ml.coef_ = np.array([1681.36641401, -2599.12755472])
ml.intercept_ = -4.53572253

def main():
    st.title("Простое приложение линейной регрессии")

    square_input = st.text_input("Введите площадь квартиры в Москве (в кв. м)", "")
    center_proximity_input = st.text_input("Введите близость к центру (в км)", "")

    if square_input and center_proximity_input:
        try:
            square = float(square_input)
            proximity = float(center_proximity_input)

            # Создаем массив с данными для предсказания
            data = np.array([square, proximity]).reshape(1, -1)

            # Предсказываем стоимость аренды
            prediction = ml.predict(data)[0]

            st.success(f"Предсказанная стоимость аренды: {prediction:.2f} рублей в месяц")
        except ValueError:
            st.error("Пожалуйста, введите числовые значения.")

if __name__ == "__main__":
    main()

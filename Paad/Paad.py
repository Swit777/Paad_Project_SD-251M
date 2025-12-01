import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(page_title="PAAD Data Analysis App", layout="wide")


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

def get_df():
    return st.session_state.get("df", None)


def save_df(df):
    st.session_state["df"] = df


# ---------- САЙДБАР / НАВИГАЦИЯ ----------

st.sidebar.title("Навигация")

page = st.sidebar.radio(
    "Раздел",
    [
        "1. Загрузка данных",
        "2. Описательный анализ",
        "3. Трансформации",
        "4. Визуализация",
        "5. Модели регрессии",
        "6. Временные ряды и ARIMA",
        "7. Предсказание на новых данных",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Проект PAAD – CRISP-DM**")


# ---------- 1. ЗАГРУЗКА ДАННЫХ ----------

if page == "1. Загрузка данных":
    st.title("1. Загрузка данных")

    uploaded_file = st.file_uploader(
        "Загрузите файл с данными (CSV или Excel)",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,            # авто-определение разделителя
                    engine="python",
                    on_bad_lines="skip"  # пропуск битых строк
                )
            else:
                df = pd.read_excel(uploaded_file)

            save_df(df)

            st.success("Данные успешно загружены!")
            st.write("Первые 10 строк:")
            st.dataframe(df.head(10))

            st.write("Информация о столбцах:")
            info = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str),
                "num_missing": df.isna().sum()
            })
            st.dataframe(info)

        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")

    else:
        st.info("Пока файл не загружен. Загрузите CSV или Excel.")


# ---------- 2. ОПИСАТЕЛЬНЫЙ АНАЛИЗ ----------

elif page == "2. Описательный анализ":
    st.title("2. Описательный анализ (Descriptive statistics)")

    df = get_df()
    if df is None:
        st.warning("Сначала загрузите датасет.")
    else:
        numeric_df = df.select_dtypes(include=[np.number])

        st.subheader("2.1. Описательная статистика")
        st.dataframe(numeric_df.describe().T)

        st.subheader("2.2. Матрица корреляций")
        corr = numeric_df.corr()
        st.dataframe(corr)

        st.subheader("Тепловая карта корреляций")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("2.3. Матрица ковариации")
        st.dataframe(numeric_df.cov())


# ---------- 3. ТРАНСФОРМАЦИИ ----------

elif page == "3. Трансформации":
    st.title("3. Трансформация данных")

    df = get_df()
    if df is None:
        st.warning("Сначала загрузите датасет.")
    else:
        st.write("Столбцы:")
        st.write(list(df.columns))

        col = st.selectbox("Столбец для трансформации", df.columns)
        transform_type = st.selectbox(
            "Тип трансформации",
            [
                "Никакая",
                "Логарифмическая (log)",
                "Биннинг",
                "Категориальный → dummy",
                "Нормализация (0-1)",
                "Стандартизация (z-score)",
            ],
        )

        if transform_type == "Никакая":
            st.info("Выберите тип трансформации.")
        elif st.button("Применить"):
            df2 = df.copy()

            if transform_type == "Логарифмическая (log)":
                df2[col + "_log"] = np.log1p(df2[col])

            elif transform_type == "Биннинг":
                bins = st.number_input("Количество бинов", 2, 100, 5)
                df2[col + "_bin"] = pd.cut(df2[col], bins=bins, labels=False)

            elif transform_type == "Категориальный → dummy":
                dummies = pd.get_dummies(df2[col], prefix=col)
                df2 = pd.concat([df2.drop(columns=[col]), dummies], axis=1)

            elif transform_type == "Нормализация (0-1)":
                x = df2[col].astype(float)
                df2[col + "_norm"] = (x - x.min()) / (x.max() - x.min())

            elif transform_type == "Стандартизация (z-score)":
                x = df2[col].astype(float)
                df2[col + "_z"] = (x - x.mean()) / x.std(ddof=0)

            save_df(df2)
            st.success("Готово! Новые данные:")
            st.dataframe(df2.head())


# ---------- 4. ВИЗУАЛИЗАЦИЯ ----------

elif page == "4. Визуализация":
    st.title("4. Визуализация данных")

    df = get_df()
    if df is None:
        st.warning("Сначала загрузите данные.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        plot_type = st.selectbox(
            "Тип графика",
            ["Histogram", "Box plot", "Scatter plot", "Time series"],
        )

        if plot_type == "Histogram":
            col = st.selectbox("Столбец", numeric)
            if st.button("Построить"):
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=30)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

        elif plot_type == "Box plot":
            col = st.selectbox("Столбец", numeric)
            if st.button("Построить"):
                fig, ax = plt.subplots()
                ax.boxplot(df[col].dropna())
                ax.set_xticklabels([col])
                ax.set_title(f"Box plot of {col}")
                st.pyplot(fig)

        elif plot_type == "Scatter plot":
            x = st.selectbox("X", numeric, key="scatter_x")
            y = st.selectbox("Y", numeric, key="scatter_y")
            if st.button("Построить"):
                fig, ax = plt.subplots()
                ax.scatter(df[x], df[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"{x} vs {y}")
                st.pyplot(fig)

        elif plot_type == "Time series":
            time_col = st.selectbox("Дата/время", df.columns)
            val_col = st.selectbox("Значение", numeric)
            if st.button("Показать"):
                ts = df[[time_col, val_col]].dropna()
                ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
                ts = ts.dropna().sort_values(time_col).set_index(time_col)

                fig, ax = plt.subplots()
                ts[val_col].plot(ax=ax)
                ax.set_title(f"{val_col} по {time_col}")
                st.pyplot(fig)


# ---------- 5. МОДЕЛИ РЕГРЕССИИ ----------

elif page == "5. Модели регрессии":
    st.title("5. Регрессия")

    df = get_df()
    if df is None:
        st.warning("Сначала загрузите данные.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric) < 2:
            st.error("Нужно минимум 2 числовых столбца.")
        else:
            y = st.selectbox("Целевая переменная (y)", numeric)
            X = st.multiselect("Признаки (X)", [c for c in numeric if c != y])

            model_name = st.selectbox(
                "Модель",
                ["Linear Regression", "Gradient Boosting", "kNN", "Decision Tree"],
            )

            if st.button("Обучить"):
                data = df[[y] + X].dropna()
                if data.empty:
                    st.error("После удаления пропусков данных не осталось.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        data[X], data[y], test_size=0.2, random_state=42
                    )

                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(random_state=42)
                    elif model_name == "kNN":
                        model = KNeighborsRegressor()
                    else:
                        model = DecisionTreeRegressor(random_state=42)

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    mse = mean_squared_error(y_test, preds)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, preds)

                    st.write("MSE:", mse)
                    st.write("RMSE:", rmse)
                    st.write("R²:", r2)

                    st.session_state["trained_model"] = model
                    st.session_state["model_features"] = X
                    st.session_state["model_target"] = y

                    fig, ax = plt.subplots()
                    ax.scatter(y_test, preds)
                    ax.plot(
                        [y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        "r--"
                    )
                    ax.set_xlabel("Истинные значения")
                    ax.set_ylabel("Предсказанные значения")
                    ax.set_title("Predicted vs True")
                    st.pyplot(fig)


# ---------- 6. ВРЕМЕННЫЕ РЯДЫ ----------

elif page == "6. Временные ряды и ARIMA":
    st.title("6. Временные ряды + ARIMA")

    df = get_df()
    if df is None:
        st.warning("Сначала загрузите данные.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        time_col = st.selectbox("Столбец даты/времени", df.columns)
        value_col = st.selectbox("Значение ряда", numeric)

        period = st.number_input("Период сезонности", 2, 365, 12)
        steps = st.number_input("Шагов прогноза", 1, 200, 10)

        if st.button("Выполнить анализ"):
            ts = df[[time_col, value_col]].dropna()
            ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
            ts = ts.dropna().sort_values(time_col).set_index(time_col)

            series = ts[value_col]

            # --- Декомпозиция
            st.subheader("Декомпозиция временного ряда")
            try:
                decomp = seasonal_decompose(series, model="additive", period=int(period))
                fig = decomp.plot()
                fig.set_size_inches(10, 8)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка декомпозиции: {e}")

            # --- ARIMA прогноз
            st.subheader("ARIMA прогноз")
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fit = model.fit()
                forecast = fit.forecast(steps=int(steps))

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(series, label="История")
                ax2.plot(forecast, label="Прогноз")
                ax2.legend()
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Ошибка ARIMA: {e}")


# ---------- 7. ПРЕДСКАЗАНИЕ НА НОВЫХ ДАННЫХ ----------

elif page == "7. Предсказание на новых данных":
    st.title("7. Предсказание на новых данных")

    model = st.session_state.get("trained_model", None)
    features = st.session_state.get("model_features", None)
    target = st.session_state.get("model_target", "target")

    if model is None or not features:
        st.warning("Сначала обучите модель во вкладке 5.")
    else:
        st.write(f"Модель обучена для y = **{target}**")
        st.write("Ожидаемые признаки:", features)

        new_file = st.file_uploader("Загрузите новый CSV или Excel", ["csv", "xlsx"])

        if new_file is not None:
            if new_file.name.endswith(".csv"):
                new_df = pd.read_csv(
                    new_file,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip"
                )
            else:
                new_df = pd.read_excel(new_file)

            st.write("Первые строки новых данных:")
            st.dataframe(new_df.head())

            missing = [f for f in features if f not in new_df.columns]
            if missing:
                st.error("Не хватает признаков: " + ", ".join(missing))
            else:
                if st.button("Сделать предсказание"):
                    X_new = new_df[features].dropna()
                    preds = model.predict(X_new)

                    result = new_df.copy()
                    # выравниваем длину, если были NaN, которые выкинули
                    result = result.loc[X_new.index]
                    result[target + "_pred"] = preds

                    st.success("Готово!")
                    st.write("Результаты (первые 20 строк):")
                    st.dataframe(result.head(20))

                    csv = result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Скачать результат",
                        csv,
                        "predictions.csv",
                        "text/csv",
                    )

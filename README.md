# TS HSE Project: Seasonality Features on M4 Monthly

Небольшой исследовательский проект по гипотезе:

> Какие способы моделирования сезонности дают наибольший прирост качества для глобальной модели на месячных рядах?

В проекте используются:

- датасет `M4 Monthly`
- бейзлайны `Naive`, `SeasonalNaive`, `auto.theta`, `auto.ets`
- одна глобальная модель `CatBoostRegressor`
- сравнение нескольких наборов seasonal-признаков

## Структура

```text
.
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── figures/
│   └── tables/
└── src/
```

## Как запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_experiment --config configs/seasonality_m4_monthly.json
```

После запуска результаты появятся в `results/tables` и `results/figures`.

## Как получаются данные

При первом запуске код автоматически скачивает файлы `M4 Monthly` в `data/raw`.

Используются три файла:

- `Monthly-train.csv`
- `Monthly-test.csv`
- `M4-info.csv`

Источник: датасет `M4` из репозитория `M4-methods`.

Для эксперимента используется фиксированная подвыборка из `120` monthly-рядов с `random_state = 17`.

## Что делает код

1. скачивает `M4 Monthly` в `data/raw`, если файлов еще нет
2. приводит данные к длинному формату
3. берет фиксированную подвыборку рядов
4. считает бейзлайны
5. обучает `CatBoost` на нескольких вариантах признаков
6. сохраняет таблицы метрик и график качества по горизонту

## Основные метрики

- `sMAPE`
- `MASE`

Обе метрики считаются на исходной шкале.

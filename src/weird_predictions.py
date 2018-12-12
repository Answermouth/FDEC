from src.utils import *
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split

COLUMN_TO_PREDICT = 'FD_Avg'

def create_prediction_datasets(data_averaged, t_minus):
    print('Creating prediction data')
    for i in range(1, t_minus):
        for col in COLUMNS:
            new_col_name = col+str(i)
            data_averaged[new_col_name] = data_averaged[col]
            data_averaged[new_col_name] = data_averaged[new_col_name].shift(i)

    prediction = data_averaged[COLUMN_TO_PREDICT].shift(-1)[t_minus:-1]
    attributes = data_averaged[t_minus:-1]

    return prediction, attributes


def random_forest(attributes, prediction):
    x_train, x_test, y_train, y_test = train_test_split(attributes, prediction, random_state=0)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    opt = BayesSearchCV(
        rf,
        {
            'n_estimators': Integer(200, 2000),
            'max_features': Categorical(['auto', 'sqrt']),
            'max_depth': Integer(10, 110),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'bootstrap': Categorical([True, False]),
        },
        n_iter=32,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    opt.fit(x_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x_test, y_test))

    params_from_bayes = opt.best_params_

    bayes_rf = RandomForestRegressor(**params_from_bayes)

    scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']

    bayes_scores = cross_validate(bayes_rf, attributes, prediction, scoring=scoring, cv=10)

    print(simplify_scores(bayes_scores))


def evaluate_model(model):
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    residuals.plot(kind='kde')
    print(residuals.describe())
    # plt.show()


def simplify_scores(scores):
    simplified_scores = {}
    for k, v in scores.items():
        simplified_scores[k] = np.mean(v)
    return simplified_scores


def main():
    print("Start")
    for dataset in DATASETS:
        avg_data = import_dataset(dataset, True)
        prediction, attributes = create_prediction_datasets(avg_data, 5)
        random_forest(attributes.drop(columns='Timestamp'), prediction)


if __name__ == "__main__":
    main()

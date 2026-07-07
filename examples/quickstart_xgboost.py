"""Development demo for the preliminary Boulevard XGBoost wrapper."""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from boulevard.estimators.xgboost import XGBRegressor


def main() -> None:
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        noise=5.0,
        random_state=0,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=0,
    )

    X_calib, X_test, y_calib, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=0,
    )

    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=0,
        verbosity=0,
    )

    model.fit(X_train, y_train)
    model.calibrate(X_calib, y_calib, alpha=0.1)

    pred = model.predict(X_test)
    lower, upper = model.predict_interval(X_test)

    print("First 5 predictions:")
    print(pred[:5])

    print("\nFirst 5 intervals:")
    for lo, hi in zip(lower[:5], upper[:5], strict=False):
        print(f"[{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()

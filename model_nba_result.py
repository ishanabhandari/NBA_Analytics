import os
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

NBA_DATA = 'nba_data'
STANDINGS = os.path.join(NBA_DATA, 'standings')
SCORES = os.path.join(NBA_DATA, 'scores')
RESULTS = os.path.join(NBA_DATA, 'results')


def add_target_column(team):
    """
    Add a target column to the dataframe which has the result of the next game for the team
    :param team:
    :return:
    """
    team['target'] = team["won"].shift(-1)
    return team


def find_team_averages(team, lookback=10):
    """
    Find the team performance over a lookback of 10 games to get better accuracy for the model
    :param lookback: Look back period for the team performance
    :param team: Team dataframe
    :return: Team averages dataframe
    """
    team_averages = team.rolling(lookback).mean()
    return team_averages


def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col


def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))


def model_backtesting(data, model, predictors, start=2, step=1):
    """

    :param data:
    :param model:
    :param predictors:
    :param start:
    :param step:
    :return:
    """
    all_predictions = []

    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        prediction = model.predict(test[predictors])
        prediction = pd.Series(prediction, index=test.index)
        combined = pd.concat([test["target"], prediction], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)


def model_fitting(df):
    """
    Fit the model to the dataframe
    :param df:
    :return:
    """
    # Using a linear model - Ridge Classifier - to fit the model
    rr = RidgeClassifier(alpha=1)

    # Using a time series split to split the data into training and testing
    split = TimeSeriesSplit(n_splits=3)

    # Using a sequential feature selector to select the 30 best features for the model
    # There are 142 features in the dataframe and we want to select the best 30
    sfs = SequentialFeatureSelector(rr,
                                    n_features_to_select=30,
                                    direction="forward",
                                    cv=split,
                                    n_jobs=1
                                    )

    # We need to scale the data to set values between 0 and 1.
    # Before we scale the data, we need to remove the columns that are not needed for scaling
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    # Scale the data using MinMaxScaler to set all values between 0 and 1
    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    # We can optimize the data using a lookback of 10 games and getting a mean of the team performance over those games
    df_enhanced = df[list(selected_columns) + ["won", "team", "season"]]

    # Create dataframes using the team averages for season using the lookback period
    df_enhanced = df_enhanced.groupby(['team', 'season'], group_keys=False).apply(find_team_averages)

    # Rename columns for the lookback period
    enhanced_cols = [f"{col}_10" for col in df_enhanced.columns]

    # Add the enhanced columns to the enhanced dataframe
    df_enhanced.columns = enhanced_cols

    # Concatenate the original dataframe with the enhanced dataframe
    df = pd.concat([df, df_enhanced], axis=1)

    # We have NaNs in the dataframe for the first 10 games of the season because of our lookback period
    # We need to remove those rows
    df = df.dropna()

    # Teams have a better win/loss record at home than away
    # We can use this information to enhance the model
    # We can add a column to the dataframe to indicate if the team is playing the next game at home or away
    # We can add who the team is playing next and the date of the next game

    df["home_next"] = add_col(df, "home")
    df["team_opp_next"] = add_col(df, "team_opp")
    df["date_next"] = add_col(df, "date")

    sanitized_df = df.merge(df[enhanced_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"],
                            right_on=["team_opp_next", "date_next"])

    removed_columns = list(sanitized_df.columns[sanitized_df.dtypes == "object"]) + removed_columns

    selected_columns = sanitized_df.columns[~sanitized_df.columns.isin(removed_columns)]

    # Fit the model to the dataframe
    sfs.fit(sanitized_df[selected_columns], sanitized_df["target"])

    # Get the best predictors from the model fitting
    predictors = list(selected_columns[sfs.get_support()])

    predictions = model_backtesting(df, rr, predictors)

    score = accuracy_score(predictions["actual"], predictions["prediction"])

    return score


def main():
    """
    Main function to run the NBA prediction from the model
    :return:
    """
    # Load the data into a dataframe
    df = pd.read_csv(os.path.join(RESULTS, "nba_games.csv"), index_col=0)

    # Sort the dataframe by date
    df = df.sort_values('date')

    # Reset the index of the dataframe to account for the sorting
    df = df.reset_index(drop=True)

    # Delete unused columns from the dataframe
    del df["mp.1"]
    del df["mp_opp.1"]
    del df["index_opp"]

    # Split the dataframe by team (there are 30 teams in the NBA)
    df = df.groupby('team', group_keys=False).apply(add_target_column)

    # Find any NaN values for the target column and replace it with the integer 2 for not known
    # This is to handle the last game of the season after which there is not other game
    df["target"][pd.isnull(df["target"])] = 2

    # Convert the target column to an integer so we have 1 for a win and 0 for a loss
    df["target"] = df["target"].astype(int, errors="ignore")

    # NaNs and Nulls are not desired in a machine learning model so removing those
    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    valid_columns = df.columns[~df.columns.isin(nulls.index)]

    # Create a new dataframe with only the valid columns
    df = df[valid_columns].copy()

    score = model_fitting(df)
    print(f"Accuracy score for the model is: {score}")


if __name__ == '__main__':
    main()

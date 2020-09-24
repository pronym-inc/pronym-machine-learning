import pickle
from typing import Dict, List, Union, TypeVar, Generic, Any, Tuple

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from pronym_machine_learning.columns import Column


ModelT = TypeVar("ModelT", bound=Union[RandomForestRegressor, LinearRegression, Ridge])


class EstimationModel(Generic[ModelT]):
    input_columns: Dict[str, Column]
    output_columns: Dict[str, Column]
    model: ModelT

    def __init__(
            self,
            input_columns: Dict[str, Column],
            output_columns: Dict[str, Column],
            model: ModelT
    ):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.model = model

    def cross_validate(self, input_df: pd.DataFrame, output_df: pd.DataFrame, num_partitions: int = 5) -> List[float]:
        cv_sets = self._get_cross_validation_sets(input_df, output_df, num_partitions)
        results = []
        for training_set, validation_set in cv_sets:
            training_input, training_output = training_set
            validation_input, validation_output = validation_set
            self.fit(training_input, training_output)
            predictions = self.predict(validation_input)
            error = mean_absolute_error(predictions, validation_output)
            results.append(error)
        return results

    def fit(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> ModelT:
        cleaned_input_df = self._clean_input_df(input_df)
        cleaned_output_df = self._clean_output_df(output_df)
        output_col = list(self.output_columns.values())[0]
        return self.model.fit(cleaned_input_df, cleaned_output_df[output_col.name])

    def predict(self, input_df: pd.DataFrame) -> Any:
        cleaned_input_df = self._clean_input_df(input_df, calibrate=False)
        encoded_prediction = self.model.predict(cleaned_input_df)
        # Assumes just 1 output column
        output_column = list(self.output_columns.values())[0]
        return output_column.untransform_value(encoded_prediction)

    def serialize(self) -> str:
        return pickle.loads(self.model)

    def _clean_df(self, in_df: pd.DataFrame, columns: Dict[str, Column], calibrate: bool = True) -> pd.DataFrame:
        df = in_df.copy()
        for column in columns.values():
            if calibrate:
                df = column.calibrate_and_get_transformed_df(df)
            else:
                df = column.transform_df(df)
        return df

    def _clean_input_df(self, df: pd.DataFrame, calibrate: bool = True) -> pd.DataFrame:
        return self._clean_df(df, self.input_columns, calibrate)

    def _clean_output_df(self, df: pd.DataFrame, calibrate: bool = True) -> pd.DataFrame:
        return self._clean_df(df, self.output_columns, calibrate)

    def _get_cross_validation_sets(
            self,
            input_df: pd.DataFrame,
            output_df: pd.DataFrame,
            num_partitions: int = 5
    ) -> List[Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]]:
        output = []
        for i in range(num_partitions):
            training_input,\
                validation_input,\
                training_output,\
                validation_output = train_test_split(input_df, output_df)
            output.append((
                (training_input, training_output),
                (validation_input, validation_output)
            ))
        return output

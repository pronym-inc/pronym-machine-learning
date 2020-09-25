from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Dict, Optional

import pandas as pd


DataT = TypeVar("DataT")


class Column(Generic[DataT], ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calibrate(self, df: pd.DataFrame) -> None:
        ...

    def calibrate_and_get_transformed_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calibrate(df)
        return self.transform_df(df)

    def transform_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        new_columns = self.get_transformed_columns(df)
        # Delete our column name - we may replace it with our new columns.
        del df[self.name]
        return pd.concat([df, new_columns], axis=1, sort=False)

    def untransform_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        new_columns = self.get_untransformed_columns(df)
        for column_name in self.get_transformed_column_names():
            del df[column_name]
        return pd.concat([df, new_columns], axis=1, sort=False)

    def get_columns_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.name]

    def get_transformed_column_names(self) -> List[str]:
        return [self.name]

    def get_transformed_columns(self, df) -> pd.DataFrame:
        return pd.DataFrame({
            self.name: self
                .get_columns_from_df(df)
                .apply(self.transform_value)
                .fillna(0)
        })

    def get_untransformed_columns(self, df) -> pd.DataFrame:
        return pd.DataFrame({
            self.name: df[self.name].apply(self.untransform_value)
        })

    def transform_value(self, value: Optional[DataT]) -> Optional[DataT]:
        return value

    def untransform_value(self, transformed_value: Optional[DataT]) -> Optional[DataT]:
        return transformed_value


class CategoricalColumn(Generic[DataT], Column[DataT]):
    onehot_column_name_to_value: Dict[str, DataT]

    def __init__(self, name):
        Column.__init__(self, name)
        self.onehot_column_name_to_value = {}

    def calibrate(self, df: pd.DataFrame) -> None:
        self.onehot_column_name_to_value = {}
        for idx, value in enumerate(df[self.name].unique()):
            key = "{0}__{1}".format(self.name, idx)
            self.onehot_column_name_to_value[key] = value

    def get_columns_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.name].fillna(0)

    def get_transformed_column_names(self) -> List[str]:
        return list(self.onehot_column_name_to_value.keys())

    def get_transformed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_columns = {}
        for column_name, category in self.onehot_column_name_to_value.items():
            column_values = df[self.name].apply(
                lambda x: (1 if x == category else 0))
            transformed_columns[column_name] = pd.Series(
                pd.SparseArray(column_values, fill_value=0),
                index=df.index)

        return pd.DataFrame(transformed_columns)

    def get_untransformed_columns(self, df: pd.DataFrame):
        untransformed_columns = df.apply(
            lambda row: next(
                category
                for column_name, category
                in self.onehot_column_name_to_value.items()
                if row[column_name] == 1
            )
        )
        return pd.DataFrame({self.name: untransformed_columns})


class QuantitativeColumn(Generic[DataT], Column[DataT]):
    blank: int
    normalization_divisor: Optional[float]
    normalization_constant: Optional[float]

    def __init__(self, name, blank=0):
        Column.__init__(self, name)
        self.blank = blank
        self.normalization_divisor = None
        self.normalization_constant = None

    def calibrate(self, df: pd.DataFrame) -> None:
        self.normalization_constant = df[self.name].min()
        self.normalization_divisor = (
            df[self.name].max() - self.normalization_constant)
        if self.normalization_divisor == 0.:
            self.normalization_divisor = 1.

    def transform_value(self, value: DataT) -> DataT:
        if value is None:
            value = self.blank
        return (
            (value - self.normalization_constant) / self.normalization_divisor
        )

    def untransform_value(self, transformed_value: Optional[DataT]) -> DataT:
        return (
            transformed_value * self.normalization_divisor +
            self.normalization_constant)

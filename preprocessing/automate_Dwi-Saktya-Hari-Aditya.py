import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari filepath yang diberikan."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus kolom yang tidak relevan untuk model."""
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=cols_to_drop)
    print(f"[INFO] Kolom dihapus: {cols_to_drop}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi missing values pada kolom Age dan Embarked."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    print(f"[INFO] Missing values setelah handling: {df.isnull().sum().sum()}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan label encoding pada kolom kategorikal."""
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    print("[INFO] Encoding selesai: Sex, Embarked")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan standard scaling pada fitur numerik."""
    scaler = StandardScaler()
    cols_to_scale = ['Age', 'Fare', 'SibSp', 'Parch']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"[INFO] Scaling selesai pada kolom: {cols_to_scale}")
    return df


def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """Split data train/test dan simpan ke folder output."""
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    df.to_csv(f'{output_dir}/titanic_preprocessed.csv', index=False)

    print(f"[INFO] Data disimpan di '{output_dir}/'")
    print(f"  - train.csv              : {train_df.shape}")
    print(f"  - test.csv               : {test_df.shape}")
    print(f"  - titanic_preprocessed.csv: {df.shape}")


def preprocess(input_path: str, output_dir: str) -> pd.DataFrame:
    """Pipeline preprocessing utama."""
    df = load_data(input_path)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df)
    split_and_save(df, output_dir)
    print("[INFO] Preprocessing selesai!")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automate Preprocessing Titanic Dataset')
    parser.add_argument('--input', type=str, default='../Titanic-Dataset.csv',
                        help='Path ke file CSV raw dataset')
    parser.add_argument('--output', type=str, default='titanic_preprocessing',
                        help='Folder output hasil preprocessing')
    args = parser.parse_args()

    preprocess(args.input, args.output)

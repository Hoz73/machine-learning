import os

from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    housing: DataFrame = read_csv('housing.csv')

    print(housing.head(8))
    print('*******************************')
    print(housing.info)
    print('*******************************')
    print(housing.value_counts())
    print('*******************************')
    print(housing.describe)
    print('*******************************')
    print(housing.count())
    print('*******************************')
    print(housing.min)
    print('*******************************')
    print(housing.max)

def splitDF(df: DataFrame):
    # Jeu de test/validation
    validation, test = train_test_split(df, test_size=10)
    return validation, test

def draw_graph(df: DataFrame):
    df.plot(kind='bar', x='median_income', y='median_house_value', color='red')
    plt.show()

if __name__ == '__main__':
    df: DataFrame = read_csv('housing.csv')
    print(df.info(verbose=True))
    validation, test = splitDF(df)
    draw_graph(test)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:14:01 2019

@author: dgelleru
"""

import time

from laminar import laminar
from laminar import laminar_examples as le
import pandas as pd


# Module Usage

# iter_flow: one iterable to break up; broken up into n chunks, where n = cores

# le.single_total reuturns sum, but can take odd=True to only sum odd numbers.
result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'])

result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'], True)
result = laminar.iter_flow(le.single_total, le.laminar_df['Col1'], odd=True)

# le.multi_tally returns count of rows that sum to greater than 25
result = laminar.iter_flow(le.multi_tally, le.laminar_df)

# list_flow: list of iterables; can contain more iterables than cores; processes
#            will pull from the queue until it is empty.

# create a list of pd.Series corresponding to laminar_df columns
series_list = [le.laminar_df[col] for col in le.laminar_df.columns]
result = laminar.list_flow(le.single_total, series_list, odd=True)

# Make a list of three copies of le.laminar_df
data_frames_list = [le.laminar_df]*3
result = laminar.list_flow(le.multi_tally, data_frames_list)


# Class Usage

my_lam = laminar.Laminar()

my_lam.show_processes()

my_lam.add_process('avg_s_length', lambda x: x.mean(), le.iris['sepal_length'])

my_lam.add_process('avg_s_width', lambda x: x.mean(), le.iris['petal_width'])

my_lam.add_process('avg_p_length', lambda x: x.mean(), le.iris['petal_length'])

my_lam.add_process('avg_p_width', lambda x: x.mean(), le.iris['petal_width'])

my_lam.drop_process('avg_s_width')

my_lam.add_process('avg_s_width', lambda x: x.mean(), le.iris['sepal_width'])

my_lam.show_processes()

my_lam.launch_processes()

my_lam.get_results()

results = my_lam.get_results()

my_lam.show_processes()



# Another example from kaggle.com/shuyangli94/food-com-recipes-and-user-interactions
def num_recipes_ingredient(iterable, string: str) -> int:
    count = 0
    for ingredients_list in iterable:
        ingredient_words = [x for x in ingredients_list.split(" ")]
        if string in ingredient_words:
            count+=1
    return count


recipes = pd.read_csv('RAW_recipes.csv')

# Without laminar
start = time.time()
number = num_recipes_ingredient(recipes['ingredients'], 'beef')
end = time.time() - start
print(f"Without laminar: {number}, took {end} seconds.") # 1388, 0.37354 seconds

# With laminar
start = time.time()
number = sum(laminar.iter_flow(num_recipes_ingredient, recipes['ingredients'], 'beef').values())
end = time.time() - start
print(f"With laminar: {number}, took {end} seconds.") # 1388, 0.1679 seconds


# Search for multiple ingredients
start = time.time()
num_chicken = num_recipes_ingredient(recipes['ingredients'], 'chicken')
num_beef = num_recipes_ingredient(recipes['ingredients'], 'beef')
num_tomatoes = num_recipes_ingredient(recipes['ingredients'], 'tomatoes')
num_water = num_recipes_ingredient(recipes['ingredients'], 'water')
num_garlic = num_recipes_ingredient(recipes['ingredients'], 'garlic')
num_wine = num_recipes_ingredient(recipes['ingredients'], 'wine')
end = time.time() - start
print(f"""num_chicken: {num_chicken}, num_beef: {num_beef}, num_tomatoes: {num_tomatoes}, 
num_water: {num_water}, num_garlic: {num_garlic}, num_wine: {num_wine}
Seconds: {end}""") # 2.34 seconds


start = time.time()
my_lam = laminar.Laminar()
my_lam.add_process('num_chicken', num_recipes_ingredient, recipes['ingredients'], 'chicken')
my_lam.add_process('num_beef', num_recipes_ingredient, recipes['ingredients'], 'beef')
my_lam.add_process('num_tomatoes', num_recipes_ingredient, recipes['ingredients'], 'tomatoes')
my_lam.add_process('num_water', num_recipes_ingredient, recipes['ingredients'], 'water')
my_lam.add_process('num_garlic', num_recipes_ingredient, recipes['ingredients'], 'garlic')
my_lam.add_process('num_wine', num_recipes_ingredient, recipes['ingredients'], 'wine')

my_lam.launch_processes()
end = time.time() - start
print(f"{my_lam.get_results()}, Seconds: {end}") # 0.67 seconds











#https://datenworks.com/fuzzy-search-buscando-texto-por-aproximacao/

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

a = fuzz.ratio("This is my first sentence","This is my first sentence.")
print(f'a = {a}')

b = fuzz.partial_ratio("This is my first sentence","This is my first sentence.")
print(f'b = {b}')

c = fuzz.partial_ratio("This is my first sentence","This is my first sentence.")
print(f'c = {c}')

d = fuzz.partial_ratio("This is my first sentence","This is my first sentence.")
print(f'd = {d}')

e = fuzz.ratio("São Clemente ganhou o Carnaval", "São Clemente o Carnaval ganhou")
print(f'e = {e}')

f = fuzz.token_sort_ratio("São Clemente ganhou o Carnaval", "São Clemente o Carnaval ganhou")
print (f'f = {f}')

g = fuzz.ratio("São Clemente ganhou o Carnaval", "São Clemente ganhou ganhou o Carnaval")
print(f'g = {g}')

h = fuzz.token_set_ratio("São Clemente ganhou o Carnaval", "São Clemente ganhou ganhou o Carnaval")
print(f'h = {h}')

options = ["Futbol Club Barcelona", "Real Madrid Club de Fútbol", "Valencia Club de Fútbol", "Real Sociedad de Fútbol"]
i = process.extract("real futbol", options, limit=2)
print(f'i = {i}')

j = process.extractOne("real", options)
print(f'j = {j}')
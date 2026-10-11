
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Bread and butter shopping transactions
transactions = [
    ['Bread', 'Butter', 'Milk'],
    ['Bread', 'Butter'],
    ['Bread', 'Milk'],
    ['Butter', 'Milk'],
    ['Bread', 'Butter', 'Jam'],
    ['Bread', 'Butter', 'Milk'],
    ['Bread', 'Jam'],
    ['Bread', 'Butter'],
    ['Bread', 'Milk', 'Jam'],
        ['Bread', 'Milk'],
        ['Bread', 'Bread'],
       
]

# Convert transactions into a binary dataset
te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded, columns=te.columns_)

print("TRANSACTION DATASET")
print(df.astype(int))


frequent = apriori(df, min_support=0.25, use_colnames=True)
print("\nFREQUENT ITEMSETS")
print(frequent)

#  association rules
rules = association_rules(
    frequent, metric="confidence", min_threshold=0.6
)
print("\nASSOCIATION RULES")
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']])

# Identify outliers using the IQR method
data = pd.DataFrame({
    'Daily_Sales': [100, 110, 105, 115, 120, 108, 112, 500,800,200,12,39,82,92,0,98]
})

Q1 = data['Daily_Sales'].quantile(0.25)
Q3 = data['Daily_Sales'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[
    (data['Daily_Sales'] < lower) |
    (data['Daily_Sales'] > upper)
]

print("\nOUTLIERS")
print(outliers)

print("\nOUTLIERS in data set ")



df = pd.read_csv("Salary_Data.csv")
data = df["Salary"]

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outliers = df[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
print("Outliers are:")
print(outliers)


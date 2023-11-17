import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re
from tqdm import tqdm

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument("--min_queries", default=1000, type=int, help="The minimum number of queries per category label (default is 1000)")
parser.add_argument("--output", default=output_file_name, help="the file to output to")
args = parser.parse_args()

min_queries = args.min_queries
output_file_name = args.output

root_category_id = 'cat00000'

# Function to normalize queries
def normalize_query(query):
    query = query.lower()
    query = re.sub(r'[^a-z0-9]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return ' '.join([stemmer.stem(word) for word in query.split()])

# Parse categories XML and prepare DataFrame
tree = ET.parse(categories_file_name)
root = tree.getroot()
categories, parents = [], []
for child in root:
    cat_id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])

# Create DataFrame for categories and their parents
parents_df = pd.DataFrame({'category': categories, 'parent': parents})

# Read and process queries
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['query'] = queries_df['query'].apply(normalize_query)

# Initialize category_counts with zero for all categories and update with actual counts
category_counts = pd.Series(0, index=pd.Index(categories))
actual_counts = queries_df['category'].value_counts()
category_counts.update(actual_counts)

# Initialize parent lookup dictionary
parent_map = parents_df.set_index('category')['parent'].to_dict()

# Roll up categories
continue_rolling_up = True
while continue_rolling_up:
    continue_rolling_up = False
    category_counts = queries_df.groupby('category').size().reset_index(name='query_count')
    category_counts = category_counts.sort_values(by='query_count')

    for _, category_row in tqdm(category_counts.iterrows(), desc="Rolling up categories"):
        if category_row['query_count'] < args.min_queries:
            current_category = category_row['category']
            parent_category = parent_map.get(current_category, root_category_id)
            if parent_category != root_category_id:
                queries_df['category'] = queries_df['category'].replace(current_category, parent_category)
                continue_rolling_up = True


# Recalculate counts after updates
queries_df['count'] = queries_df.groupby('category')['query'].transform('count')

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

# Print the count of unique categories after roll-up
unique_categories_count = queries_df['category'].nunique()
print(f"Number of unique categories after rolling up: {unique_categories_count}")
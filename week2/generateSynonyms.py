import fasttext

# Load the model
model_path = '/workspace/datasets/fasttext/title_model_100.bin'
model = fasttext.load_model(model_path)

# Set the threshold for similarity
threshold = 0.8  

# Process top words and store results
output_file_path = '/workspace/datasets/fasttext/synonyms.csv'
top_words_path = '/workspace/datasets/fasttext/top_words.txt'

with open(top_words_path, 'r') as top_words_file, open(output_file_path, 'w') as output_file:
    for word in top_words_file:
        word = word.strip()
        neighbors = [neighbor[1] for neighbor in model.get_nearest_neighbors(word) if neighbor[0] >= threshold]
        
        # Write to output if there are any neighbors above the threshold
        if neighbors:
            output_file.write(f"{word},{','.join(neighbors)}\n")
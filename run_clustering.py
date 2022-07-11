"""
example run:
python run_clustering.py --min_count 100 --documents_path X_val.csv --output_path ./results
"""

from top2vec import Top2Vec
import pandas as pd
import pickle as pkl
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
from umap import UMAP
import pylab
from matplotlib import pyplot as plt


if __name__ == "__main__":


	parser = argparse.ArgumentParser(description='Clustering with Top2Vec')

	parser.add_argument('--min_count', type=int, help='Min number of terms accepted')
	parser.add_argument('--documents_path', type=str, help='Input documents path')
	parser.add_argument('--output_path', type=str, help='Input documents path')

	args = parser.parse_args()



	documents = pd.read_csv(args.documents_path, index_col='id')['content'].values.tolist()
	min_count = args.min_count
	# instantiate and train the model 
	model = Top2Vec(documents=documents, speed="fast-learn", min_count=min_count)

	# save trained model 
	print(f'The model found {model.get_num_topics()} topics')
	with open(f'{args.output_path}/model.pkl', 'wb') as f:
	    pkl.dump(model, f)

	topic_sizes, topic_nums = model.get_topic_sizes()
	print(f'On averege, each topic includes {round(np.mean(topic_sizes))} +/- {round(np.std(topic_sizes), 2)}')

	# topic redeuction according to the number of topic
	num_topics_list = [5, 10, 20, 30, 50, 100]
	
	hierarchy_lists = []
	for num_topics in tqdm(num_topics_list):

	    hierarchy = model.hierarchical_topic_reduction(num_topics = num_topics)
	    hierarchy_lists.append(hierarchy)

	with open(f'{args.output_path}/hierarchy_list.pkl', 'wb') as f:
		pkl.dump(hierarchy_lists, f)


	# some viz
	topics = ['lavoro', 'scuola', 'musica', 'cinema', 'legge']

	words_embeddings = []
	words_list = []
	topics_list = []

	# words vectors saved in model.word_vectors according to model.word_indexes data
	# topic vectors in model.topic_vectors


	for topic in topics:
		# search topics most similar to ['lavoro', 'scuola', 'musica', 'cinema', 'legge']
	    topics_words, words_scores, topic_scores, topic_nums =  model.search_topics([topic], num_topics = 2)
	    
	    for word in set(np.concatenate(topics_words)):
	        i = model.word_indexes[word]
	        embedding = model.word_vectors[i]
	        words_embeddings.append(embedding)
	        words_list.append(word)
	        topics_list.append(topic)

	n_words = len(words_embeddings)

	# lower dimensional space
	z = TSNE(n_components=2).fit_transform(words_embeddings)
	fig, ax= plt.subplots(figsize=(20, 10))

	ax.grid()
	# to manage colours
	conversion = dict(zip(set(topics_list), range(len(set(topics_list)))))
	ax.scatter(z[:n_words, 0], z[:n_words, 1], c=[conversion[t] for t in topics_list], cmap=pylab.cm.cool)
	for i, word in enumerate(words_list):
	    ax.annotate(word, (z[i, 0], z[i, 1]), )
	    
	ax.set_title(f"TSNE {', '.join(topics)}")

	plt.savefig(f'{args.output_path}/tsne.svg', bbox_inches='tight')



        
        


    
    




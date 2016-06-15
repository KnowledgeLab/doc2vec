import os
import csv
import random
import nltk
from gensim.models.doc2vec import LabeledSentence
import re
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import copy
extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e', 'e.g', 'ph.d', 'eq', 'eqs', 'fig']
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)


def read_cnf(inputs):
    dict_ = {}
    read_info = False
    try:
        with open(inputs[0]['sql'] ,'r') as fp:
            for line in fp:
                if '[client]' in line and not read_info:
                    read_info = True
                    continue
                if read_info:
                    split_line = line.split('=')
                    dict_[split_line[0]] = split_line[1].strip()
                if not line.strip():
                    return dict_
    except:
        return dict_
    return dict_ 

def build_doc_matrix(docs, model, dims, norm=False):
	res = np.zeros((len(docs), dims))
	ix_ = 0
	index_ = []
	for doc in docs:
		try:
			vec = model.docvecs[doc.tags[0]]
			index_.append(int(doc.tags[0]))
		except:
			raise Exception("Couldn't find a document!")
			#print 'in here'
			#vec = model.docvecs[ix_]
			#index_.append(ix_)
		
		if norm:
			vec /= np.linalg.norm(vec, 1)

		res[ix_,:] = vec
		ix_ += 1
	return res, index_


def closest_sents(model, sentence, uid=None):
	internal_vec = False
	if uid:
		vec = model.docvecs[str(uid)]
		internal_vec = True
	else:
		#print sentence
		vec = model.infer_vector(sentence) 
		#print vec

	#print internal_vec
	if uid:
		res = cdist(np.reshape(vec, (1,vec.size)), model.docvecs[[i for i in xrange(len(model.docvecs)) if i != uid]], 'cosine')	
	else:
		res = cdist(np.reshape(vec, (1,vec.size)), model.docvecs, 'cosine')
	ix = np.argsort(res[0])

	return ix, res[0,ix]

def find_section_worker(section, sec, uids):
	res = []
	#print "in worker"
	for type_ in section:
		#print "in type"
		if any(uid in section[type_] for uid in uids):
			#print "in any"
			for ix in range(len(uids)):
				#print "in ix"
				if uids[ix] in section[type_]:
					#print "in uids"
					res.append((ix, sec, type_))
	return res

def find_section(articles, uids, prev_sec=None, prev_type=None):
	res = []
	pool = Pool(processes=4)
	results = []
	if prev_sec and prev_type:
		if any(uid in articles[prev_sec][prev_type] for uid in uids):
			for ix in range(len(uids)):
				if uids[ix] in articles[prev_sec][prev_type]:
					res.append((ix, prev_sec, prev_type))
		if res:
			return res	
	for section in articles:

		try:
			#print find_section_worker(articles[section], section, uids)
			#return
			result = pool.apply_async(find_section_worker, args=(articles[section], section, uids))
		#for type_ in articles[section]:
		except:
			print "error in the worker"
		results.append(result)
			#if any(uid in articles[section][type_] for uid in uids):
			#	for ix in range(len(uids)):
			#		if uids[ix] in articles[section][type_]:
			#			res.append((ix, section, type_))
	for result in results:
		for x in result.get():
			res.append(x)
	pool.close()
	pool.join()
	return res


def get_journal_concat(journal):
    current_ngrams = []
    for doc in journal:
        uid = doc[0]
        for ngram in doc[1]:
            current_ngrams.append(ngram[0].lower())
    unique_ngrams = " ".join(list(set(current_ngrams)))
    return LabeledSentence(words = unique_ngrams.split(), tags=['%s' %(uid)])


def section_doc(doc, granularity='sent', uid=0, patt=r'[,\.-_]+$', min_len=2):
      
        res = None
        if isinstance(doc, tuple):
            uid = doc[0]
            doc = doc[1]
	# Sentence tokenize the input document according to the NLTK tokenizer above
	if granularity== 'sent':
            try:
		sents = nltk.sent_tokenize(doc[1])
            except UnicodeDecodeError:
                sents = nltk.sent_tokenize(doc[1].decode('utf-8', 'ignore'))
            for sent in sents:
                    tokens = sent.lower().split()
                    filtered_tokens = [re.sub('\b[0-9]+[\.,]?[0-9]+\b', '##', x) for x in tokens]
                    res = LabeledSentence(words = [re.sub(patt, '', x) if len(re.sub(patt, '', x)) >= min_len else '' for x in filtered_tokens], tags=['%s' %(uid)])
                    #uid += 1
                    yield res	

	elif granularity=='ngrams':
                for x in doc:
                    try:
	                res = LabeledSentence(words = x[0].lower().split(), tags=['%s' %(uid)])
                    except UnicodeDecodeError:
                        res = LabeledSentence(words = x[0].decode('utf-8', 'ignore').lower().split(), tags=['%s' %(uid)])
                    #uid += 1
                    yield res
	# Just embed the documents directly
	else:
            try:
                tokens = doc.lower().split()
                filtered_tokens = [re.sub('\b[0-9]+[\.,]?[0-9]+\b', '##', x) for x in tokens]
		res = LabeledSentence(words = [re.sub(patt, '', x) if len(re.sub(patt, '', x)) >= min_len else '' for x in filtered_tokens], tags=['%s' %(uid)])
            except UnicodeDecodeError:
                tokens = doc.decode('utf-8', 'ignore').lower().split()
                filtered_tokens = [re.sub('\b[0-9]+[\.,]?[0-9]+\b', '##', x) for x in tokens]
                res = LabeledSentence(words = [re.sub(patt, '', x) if len(re.sub(patt, '', x)) >= min_len else '' for x in filtered_tokens], tags = ['%s' %(uid)])
            #uid += 1
            yield res
		

def gen_vocab(model, docs, win, seed=0, min_count=1):
    mdl = model(docs, window=win, seed=seed, min_count=min_count)
    #mdl.build_vocab(docs)
    #for epoch in xrange(4):
    #    mdl.train(docs)
    mdl.finalize_vocab()
    return mdl

def train_mdl(model, docs, workers=4, pretrained_vocab=None, **params):
        epochs = int(params['epochs'])
        params.pop("epochs", None)

        if pretrained_vocab != None:
            #mdl = model(docs, size, win, min_count, workers, seed=seed, dm=dm, dbow_words=dbow_words )
            mdl = model(docs, workers=workers, **params)
            mdl.reset_from(pretrained_vocab)
        else:
            #mdl = model(docs, size, win, min_count, workers, seed=seed, dm=dm, dbow_words=dbow_words )
            mdl = model(docs, workers=workers, **params)
            #mdl.build_vocab(docs)
	len_docs = len(docs)
        shuffled_docs = copy.deepcopy(docs)
        for epoch in xrange(epochs):
            random.shuffle(shuffled_docs)
	    mdl.train(shuffled_docs, total_examples=len_docs)
	return mdl.docvecs[[doc.tags[0] for doc in docs]], mdl.syn0, mdl

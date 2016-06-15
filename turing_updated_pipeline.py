#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
import random
import numpy as np
import re
import gensim
from gensim.models.doc2vec import LabeledSentence, FAST_VERSION, Doc2Vec
from utils import section_doc, train_mdl, gen_vocab
import pickle
import json
import os
import utils
import csv
import multiprocessing
import argparse

assert gensim.models.doc2vec.FAST_VERSION >-1, 'this will be painfully slow otherwise'
csv.field_size_limit(sys.maxsize)

'''
    Input:
        data_file: path
            path to line-delineated document file
            ideally has .data extension
        mdl_file: path
            path to existing *_mdl.pkl file to pull the vocab from

    Output:
        *_word_mat.pkl
            Pickle object of the actual word numpy matrix
        *_doc_mat.pkl
            Pickle object of the gensim docvecs structure
        *_mdl.pkl
            Pickle object of the gensim Doc2Vec structure

'''

default_params = {'size':100,
                 'window':5,
                 'min_count':1,
                 'epochs':15,
                 'seed': 0}

def write_sample_params(outFileName):
    sample_params = {'size':100,
                     'window':5,
                     'min_count':1,
                     'epochs':15,
                     'seed': 0}
    with open(outFileName, 'w') as fp:
        json.dump(sample_params, fp)
    

def write_sample_data(outFileName):
        # At this point input_docs is a list of strings
        foo = [('1','Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse non ultricies turpis. Donec ut ligula convallis, bibendum eros quis, hendrerit tellus. Sed rutrum nisl faucibus elit pellentesque, eget hendrerit turpis elementum. Nam semper augue non dignissim porta. Nulla rutrum quis nulla in tincidunt. Quisque malesuada, quam et luctus mollis, turpis lectus elementum dolor, sed maximus lorem diam eget magna. Fusce sit amet nulla id ex fringilla interdum at vitae elit. Sed ut posuere eros. Duis a ullamcorper nisl. Vivamus id elit fringilla, dictum est at, dictum elit. In hac habitasse platea dictumst. Phasellus risus arcu, tincidunt at egestas eu, viverra porttitor massa. Sed lacinia mi quam, eu finibus elit consequat quis. Donec ut imperdiet mauris. Quisque a lacus vel diam tempor vestibulum.')\
               , ('2','Duis tincidunt euismod ipsum ut pharetra. Sed hendrerit porta suscipit. Etiam id interdum elit. Etiam euismod dolor quis eros tempor, quis euismod erat elementum. Nulla efficitur vel ex eget facilisis. Suspendisse quis enim et purus viverra malesuada. Cras sed ante id augue lobortis tristique. Sed a ligula eu leo ornare sollicitudin a id turpis. Phasellus ut suscipit nibh, eget congue tortor. Nulla bibendum, arcu in auctor aliquam, quam nisl luctus orci, eget pellentesque quam orci at leo. Vestibulum ut dictum urna.'), \
               ('3', 'Fusce vestibulum felis in sapien fermentum blandit. Maecenas non sapien pharetra, feugiat mi nec, bibendum arcu. Curabitur tincidunt quam in ornare tincidunt. Integer commodo mauris eu semper facilisis. Morbi venenatis massa mauris, sit amet mollis leo tempor at. Vivamus sodales nunc et sem accumsan vestibulum. Nulla ultricies sapien in molestie sollicitudin. Nulla eu varius sem. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Quisque vel nibh ullamcorper, rutrum nunc nec, dignissim quam. Sed orci nisi, eleifend eget velit quis, gravida finibus sem. Nulla faucibus rhoncus pretium. Duis quis tortor nisl. Integer vitae dolor commodo ante imperdiet placerat. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Proin convallis elementum efficitur.'), \
               ('4', ('Donec convallis lobortis feugiat. Vivamus id enim dictum ipsum laoreet interdum at eget justo. Pellentesque eleifend leo nibh, eu commodo eros egestas id. Nulla luctus eu nibh vitae convallis. Curabitur in urna non mauris imperdiet lobortis nec sagittis elit. Pellentesque ornare turpis at commodo aliquet. Proin quis auctor est, non dapibus nulla. Praesent sed lacinia neque. Phasellus egestas elit vel massa posuere, interdum ullamcorper odio vehicula. Sed nec mi quis metus bibendum scelerisque at euismod metus. Phasellus a risus consectetur, suscipit massa ut, pharetra quam.'+unichr(137)).encode('utf-8'))]
        zipped = foo

        with open(outFileName, "w") as f:
            csvWriter = csv.writer(f)
            for tup in zipped:
                csvWriter.writerow([tup[0], tup[1]])


def pickle_to_file(fname, obj):
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle)

def pickle_load_from_file(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

def pipeline(inputs, outputs):
    input_docs = []
    ids = []
    logging.info("Begin reading input data file.")
    with open(inputs[0]["data"], ) as f :
        csvReader = csv.reader(f, delimiter=',')
        for row in csvReader:
            input_docs.append(row)
            #ids.append(row[0])
        #input_docs = f.read().splitlines()
    logging.info("Finished reading input data file.")
    mdl_docs = []
    uid =0
    logging.info("Begin parsing parameters.")
    if inputs[0]['params']:
        with open(inputs[0]['params'], 'r') as f:
            mdl_params = json.load(f)
        logging.info("Loaded parameter file.")
    else:
        mdl_params = default_params
        logging.info("Using default parameters.")
    logging.info("Finished parsing parameters.")
    logging.info("Begin decoding and processing data.")
    for doc in input_docs: 
        for sec in section_doc(tuple(doc), 'abs', uid): #iterate through each sentence in doc
            mdl_docs.append(sec) #Add the encapsulated 'labeledSentence' sentence obj 
            uid = sec.tags[0] #Update the unique identifier for the sentence
    logging.info("Finished decoding and processing data.")

    logging.info("Checking for and unpickling possible model object.")
    if inputs[0]['vocab_file']:
        vocab = pickle_load_from_file(inputs[0]['vocab_file'])
        logging.info("Unpickled model object.")
    else:
        vocab = None
        logging.info("No object, moving on.")

    logging.info("Begin training model.")
    logging.debug("number of documents {0}\t number of cpus {1}.".format(len(mdl_docs), multiprocessing.cpu_count()))
    doc_mat, word_mat, mdl = train_mdl(Doc2Vec, mdl_docs, workers= min(len(mdl_docs), multiprocessing.cpu_count()), pretrained_vocab=vocab, **mdl_params)
    logging.info("Finished training model.")
    output_prefix=outputs[0]['prefix']

    logging.info("Begin serializing objects.")
    pickle_to_file(output_prefix + "doc_mat.pkl",  doc_mat)
    pickle_to_file(output_prefix + "word_mat.pkl", word_mat)
    pickle_to_file(output_prefix + "mdl.pkl",      mdl)
    logging.info("Finished serializing objects.")

#dataset, vocab_matrix_file

def test_no_vocab():
    write_sample_data('test.data')
    data_file = 'test.data'
    mdl_file = None
    #write_sample_params('test.json')
    params='test.json'
    #params=None
    prefix = 'test'
    inputs = [{'data': data_file, 'vocab_file': mdl_file, 'params': params}]
    outputs = [{'prefix':prefix}]
    pipeline(inputs, outputs)
    print "done with test with no vocab matrix"

def test_with_vocab():
    write_sample_data('test.data')
    data_file = 'test.data'
    #write_sample_params('test.json')
    params='test.json'
    #params=None

    mdl_file = 'test_mdl.pkl'
    prefix = 'test'
    inputs = [{'data': data_file, 'vocab_file': mdl_file, 'params': params}]
    outputs = [{'prefix':prefix}]
    pipeline(inputs, outputs)
    print "done with test with vocab matrix"

def main (documentfile, modelfile):
    prefix = ''
    params = None
    inputs = [{'data': documentfile, 'vocab_file': modelfile, 'params':params}]
    outputs = [{'prefix':prefix}]
    
    logging.basicConfig(filename='pipeline.log', level=logging.DEBUG)

    logging.info("Started pipeline.")
    pipeline(inputs, outputs)
    logging.info("Finished pipeline.")

# This code block allows this code to be executed from the commandline
# Do not change this interface.
if __name__ == "__main__" :

    parser   = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", default="pipeline.log", help="Logfile path. Defaults to ./pipeline.log")
    parser.add_argument("-d", "--document", help="File path to the csv document file", required=True)
    parser.add_argument("-m", "--model",    help="File path to a pickled model file", default=None)
    parser.add_argument("-p", "--params",   help="File path to a json params file", default=None)
    args   = parser.parse_args()
    
    logging.basicConfig(filename=args.logfile, level=logging.DEBUG)
    logging.info("Running with args : {0} ".format(sys.argv))

    prefix  = ''
    inputs  = [{'data': args.document, 'vocab_file': args.model, 'params':args.params }]
    outputs = [{'prefix':prefix}]    

    logging.info("Started pipeline.")
    pipeline(inputs, outputs)
    logging.info("Finished pipeline.")



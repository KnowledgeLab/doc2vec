# coding: utf-8
import tsne
import cPickle as pkl
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Document matrix pickle object
with open(sys.argv[1], 'r') as doc_mat_pkl:
    vectors = pkl.load(doc_mat_pkl)

Y = tsne.tsne(vectors.astype(np.float64), no_dims=2, perplexity=5)

# Output name for the pdf, sans the filetype
pp = PdfPages(sys.argv[2] + '.pdf')
fig = plt.figure(figsize=(16,12))
plt.scatter(Y[:,0], Y[:,1], c='k')
pp.savefig()
pp.close()
plt.close()


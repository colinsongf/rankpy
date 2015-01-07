RankPy: Learning to Rank with Python
====================================

Currently, the RankPy project is under intensive development. The goal is to provide reliable, efficient, and stable implementations of the state of the art
learning to rank algorithms. Why not start with one of the most famous algorithms, which is available right now: LambdaMART.


Prerequisities
--------------
- Python (2.7)
- Cython (0.21.1)
- NumPy  (1.8.1)
- SciPy  (0.14.0)
- Scikit-learn (0.14.1)

Installation
------------
Install the prerequisites and RankPY as follows::

    $ git clone https://bitbucket.org/tunystom/rankpy.git
    $ cd rankpy
    $ pip install -r requirements.txt
    $ python setup.py install


Using RankPy: Simple Example
----------------------------
1) Prepare data in svmlight format, e.g., download the *MQ2007* (see next section on `Data`_) ::

        $ mkdir data
        $ wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar -O data/MQ2007.rar
        $ unrar x data/MQ2007.rar data/
        
2) Create a python script file with the following content::

        $ cat << EOF > run_lambdamart.py
        # -*- coding: utf-8 -*-

        import numpy as np

        import logging

        from rankpy.queries import Queries
        from rankpy.models import LambdaMART
        from rankpy.metrics import NormalizedDiscountedCumulativeGain

        # Turn on logging.
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

        # Load the query datasets.
        train_queries = Queries.load_from_text('data/MQ2007/Fold1/train.txt')
        valid_queries = Queries.load_from_text('data/MQ2007/Fold1/vali.txt')
        test_queries = Queries.load_from_text('data/MQ2007/Fold1/test.txt')

        logging.info('================================================================================')

        # Save them to binary format ...
        train_queries.save('data/fold1_train')
        valid_queries.save('data/fold1_vali')
        test_queries.save('data/fold1_test')

        # ... because loading them will be then faster.
        train_queries = Queries.load('data/fold1_train')
        valid_queries = Queries.load('data/fold1_vali')
        test_queries = Queries.load('data/fold1_test')

        logging.info('================================================================================')

        # Print basic info about query datasets.
        logging.info('Train queries: %s' % train_queries)
        logging.info('Valid queries: %s' % valid_queries)
        logging.info('Test queries: %s' %test_queries)

        logging.info('================================================================================')

        # Prepare metric for this set of queries.
        metric = NormalizedDiscountedCumulativeGain(10, queries=[train_queries, valid_queries, test_queries])

        # Initialize LambdaMART model and train it.
        model = LambdaMART(n_estimators=10000, max_depth=4, shrinkage=0.08, estopping=100, n_jobs=-1)
        model.fit(metric, train_queries, validation=valid_queries)

        logging.info('================================================================================')

        # Print out the performance on the test set.
        logging.info('%s on the test queries: %.8f' % (metric, metric.evaluate_queries(test_queries, model.predict(test_queries, n_jobs=-1))))
        EOF

3) Run the script::
        
        $ python run_lambdamart.py

4) Enjoy the results::
   
	2015-01-07 21:10:52,403 : Reading queries from data/MQ2007/Fold1/train.txt.
	2015-01-07 21:10:53,670 : Read 244 queries and 10000 documents so far.
	2015-01-07 21:10:54,937 : Read 479 queries and 20000 documents so far.
	2015-01-07 21:10:56,194 : Read 720 queries and 30000 documents so far.
	2015-01-07 21:10:57,463 : Read 963 queries and 40000 documents so far.
	2015-01-07 21:10:57,734 : Read 1017 queries and 42158 documents out of which 148 queries and 6340 documents were discarded.
	2015-01-07 21:10:58,215 : Reading queries from data/MQ2007/Fold1/vali.txt.
	2015-01-07 21:10:59,481 : Read 246 queries and 10000 documents so far.
	2015-01-07 21:10:59,958 : Read 339 queries and 13813 documents out of which 44 queries and 1843 documents were discarded.
	2015-01-07 21:11:00,113 : Reading queries from data/MQ2007/Fold1/test.txt.
	2015-01-07 21:11:01,373 : Read 245 queries and 10000 documents so far.
	2015-01-07 21:11:01,829 : Read 336 queries and 13652 documents out of which 46 queries and 1910 documents were discarded.
	2015-01-07 21:11:01,981 : ================================================================================
	2015-01-07 21:11:01,990 : Loading queries from data/fold1_train.
	2015-01-07 21:11:02,145 : Loaded 869 queries with 35818 documents in total.
	2015-01-07 21:11:02,146 : Loading queries from data/fold1_vali.
	2015-01-07 21:11:02,197 : Loaded 295 queries with 11970 documents in total.
	2015-01-07 21:11:02,197 : Loading queries from data/fold1_test.
	2015-01-07 21:11:02,247 : Loaded 290 queries with 11742 documents in total.
	2015-01-07 21:11:02,247 : ================================================================================
	2015-01-07 21:11:02,247 : Train queries: Queries (869 queries, 35818 documents, 2 max. relevance)
	2015-01-07 21:11:02,247 : Valid queries: Queries (295 queries, 11970 documents, 2 max. relevance)
	2015-01-07 21:11:02,248 : Test queries: Queries (290 queries, 11742 documents, 2 max. relevance)
	2015-01-07 21:11:02,248 : ================================================================================
	2015-01-07 21:11:02,249 : Training of LambdaMART model has started.
	2015-01-07 21:11:02,807 : #00000001: NDCG@10 (training):    0.49414354  |  (validation):    0.49067198
	2015-01-07 21:11:03,369 : #00000002: NDCG@10 (training):    0.50579211  |  (validation):    0.50022160
	2015-01-07 21:11:03,925 : #00000003: NDCG@10 (training):    0.50744720  |  (validation):    0.50516971
	2015-01-07 21:11:04,483 : #00000004: NDCG@10 (training):    0.51228340  |  (validation):    0.50226151
	2015-01-07 21:11:05,040 : #00000005: NDCG@10 (training):    0.51123786  |  (validation):    0.50246569
	2015-01-07 21:11:05,600 : #00000006: NDCG@10 (training):    0.51223413  |  (validation):    0.50483495
	2015-01-07 21:11:06,157 : #00000007: NDCG@10 (training):    0.51433246  |  (validation):    0.50621523
	2015-01-07 21:11:06,719 : #00000008: NDCG@10 (training):    0.51594499  |  (validation):    0.51007842
	2015-01-07 21:11:07,283 : #00000009: NDCG@10 (training):    0.51770122  |  (validation):    0.51255115
	2015-01-07 21:11:07,845 : #00000010: NDCG@10 (training):    0.52076834  |  (validation):    0.51319742
	(... 238 more lines ...)
	2015-01-07 21:13:24,051 : #00000249: NDCG@10 (training):    0.67300587  |  (validation):    0.54154038
	(... 99 more lines ...)
	2015-01-07 21:14:15,270 : #00000339: NDCG@10 (training):    0.71662568  |  (validation):    0.52989547
	2015-01-07 21:14:15,836 : #00000340: NDCG@10 (training):    0.71777584  |  (validation):    0.52989301
	2015-01-07 21:14:16,414 : #00000341: NDCG@10 (training):    0.71790088  |  (validation):    0.52997733
	2015-01-07 21:14:16,983 : #00000342: NDCG@10 (training):    0.71799624  |  (validation):    0.53022398
	2015-01-07 21:14:17,559 : #00000343: NDCG@10 (training):    0.71935630  |  (validation):    0.52931864
	2015-01-07 21:14:18,143 : #00000344: NDCG@10 (training):    0.71925165  |  (validation):    0.52934598
	2015-01-07 21:14:18,714 : #00000345: NDCG@10 (training):    0.71957871  |  (validation):    0.52985654
	2015-01-07 21:14:19,279 : #00000346: NDCG@10 (training):    0.71984486  |  (validation):    0.53004778
	2015-01-07 21:14:19,841 : #00000347: NDCG@10 (training):    0.72075425  |  (validation):    0.53146773
	2015-01-07 21:14:20,407 : #00000348: NDCG@10 (training):    0.72128240  |  (validation):    0.53341107
	2015-01-07 21:14:20,987 : #00000349: NDCG@10 (training):    0.72149168  |  (validation):    0.53249132
	2015-01-07 21:14:20,987 : Stopping early since no improvement on validation queries has been observed for 100 iterations (since iteration 249)
	2015-01-07 21:14:20,987 : Final model performance (NDCG@10) on validation queries:  0.54154038
	2015-01-07 21:14:20,987 : Training of LambdaMART model has finished.
	2015-01-07 21:14:20,987 : ================================================================================
	2015-01-07 21:14:21,241 : NDCG@10 on the test queries: 0.54800457

Data
----
RankPy acceptes data formatted in the SVMlight (see http://svmlight.joachims.org/) format.
You can download learning to rank data sets here:

- **GOV**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR3.0/Gov.rar (you'll need files in QueryLevelNorm)
- **OHSUMED**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR3.0/OHSUMED.zip
- **MQ2007**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar (files for supervised learning)
- **MQ2008**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2008.rar (files for supervised learning)
- **Yahoo!**: http://webscope.sandbox.yahoo.com/catalog.php?datatype=c
- **MSLR-WEB10K**: http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB10K.zip
- **MSLR-WEB30K**: http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB30K.zip
- **Yandex Internet Mathematics 2009**: http://imat2009.yandex.ru/academic/mathematic/2009/en/datasets (query identifier need to be parsed out of comment into qid feature)


License
-------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
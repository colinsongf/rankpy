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
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
   
        2014-11-27 18:13:59,627 : INFO : Reading queries from data/MQ2007/Fold1/train.txt.
        2014-11-27 18:14:01,122 : INFO : Read 244 queries and 10000 documents so far.
        2014-11-27 18:14:02,548 : INFO : Read 479 queries and 20000 documents so far.
        2014-11-27 18:14:03,996 : INFO : Read 720 queries and 30000 documents so far.
        2014-11-27 18:14:05,455 : INFO : Read 963 queries and 40000 documents so far.
        2014-11-27 18:14:05,796 : INFO : Read 1017 queries and 42158 documents in total.
        2014-11-27 18:14:06,430 : INFO : Reading queries from data/MQ2007/Fold1/vali.txt.
        2014-11-27 18:14:07,872 : INFO : Read 246 queries and 10000 documents so far.
        2014-11-27 18:14:08,418 : INFO : Read 339 queries and 13813 documents in total.
        2014-11-27 18:14:08,602 : INFO : Reading queries from data/MQ2007/Fold1/test.txt.
        2014-11-27 18:14:10,026 : INFO : Read 245 queries and 10000 documents so far.
        2014-11-27 18:14:10,548 : INFO : Read 336 queries and 13652 documents in total.
        2014-11-27 18:14:10,730 : INFO : ================================================================================
        2014-11-27 18:14:10,758 : INFO : Loading queries from data/fold1_train.
        2014-11-27 18:14:10,938 : INFO : Loaded 1017 queries with 42158 documents in total.
        2014-11-27 18:14:10,938 : INFO : Loading queries from data/fold1_vali.
        2014-11-27 18:14:10,996 : INFO : Loaded 339 queries with 13813 documents in total.
        2014-11-27 18:14:10,996 : INFO : Loading queries from data/fold1_test.
        2014-11-27 18:14:11,054 : INFO : Loaded 336 queries with 13652 documents in total.
        2014-11-27 18:14:11,054 : INFO : ================================================================================
        2014-11-27 18:14:11,054 : INFO : Train queries: Queries (1017 queries, 42158 documents, 2 max. relevance)
        2014-11-27 18:14:11,054 : INFO : Valid queries: Queries (339 queries, 13813 documents, 2 max. relevance)
        2014-11-27 18:14:11,054 : INFO : Test queries: Queries (336 queries, 13652 documents, 2 max. relevance)
        2014-11-27 18:14:11,054 : INFO : ================================================================================
        2014-11-27 18:14:11,280 : INFO : Training of LambdaMART model has started.
        2014-11-27 18:14:11,565 : INFO : #00000001: NDCG@10 (training):    0.35528780  |  (validation):    0.35419232
        2014-11-27 18:14:11,826 : INFO : #00000002: NDCG@10 (training):    0.37174118  |  (validation):    0.39656342
        2014-11-27 18:14:12,094 : INFO : #00000003: NDCG@10 (training):    0.37436836  |  (validation):    0.38153144
        2014-11-27 18:14:12,368 : INFO : #00000004: NDCG@10 (training):    0.38177347  |  (validation):    0.40781668
        2014-11-27 18:14:12,626 : INFO : #00000005: NDCG@10 (training):    0.38203238  |  (validation):    0.39377857
        2014-11-27 18:14:12,892 : INFO : #00000006: NDCG@10 (training):    0.38203238  |  (validation):    0.39968925
        2014-11-27 18:14:13,165 : INFO : #00000007: NDCG@10 (training):    0.38203238  |  (validation):    0.39852239
        2014-11-27 18:14:13,460 : INFO : #00000008: NDCG@10 (training):    0.40261836  |  (validation):    0.40465719
        2014-11-27 18:14:13,741 : INFO : #00000009: NDCG@10 (training):    0.40456680  |  (validation):    0.40871477
        2014-11-27 18:14:14,010 : INFO : #00000010: NDCG@10 (training):    0.40457387  |  (validation):    0.40656250
        (... 59 more lines ...)
        2014-11-27 18:14:27,148 : INFO : #00000060: NDCG@10 (training):    0.43396689  |  (validation):    0.45404817
        (... 89 more lines ...)
        2014-11-27 18:14:50,433 : INFO : #00000150: NDCG@10 (training):    0.44450545  |  (validation):    0.45111756
        2014-11-27 18:14:50,685 : INFO : #00000151: NDCG@10 (training):    0.44491334  |  (validation):    0.45155821
        2014-11-27 18:14:50,939 : INFO : #00000152: NDCG@10 (training):    0.44493870  |  (validation):    0.45152778
        2014-11-27 18:14:51,200 : INFO : #00000153: NDCG@10 (training):    0.44487566  |  (validation):    0.45272614
        2014-11-27 18:14:51,460 : INFO : #00000154: NDCG@10 (training):    0.44470577  |  (validation):    0.45253221
        2014-11-27 18:14:51,710 : INFO : #00000155: NDCG@10 (training):    0.44495404  |  (validation):    0.45273629
        2014-11-27 18:14:51,969 : INFO : #00000156: NDCG@10 (training):    0.44470760  |  (validation):    0.45184079
        2014-11-27 18:14:52,216 : INFO : #00000157: NDCG@10 (training):    0.44459801  |  (validation):    0.45160294
        2014-11-27 18:14:52,472 : INFO : #00000158: NDCG@10 (training):    0.44553192  |  (validation):    0.45074300
        2014-11-27 18:14:52,722 : INFO : #00000159: NDCG@10 (training):    0.44564437  |  (validation):    0.45136897
        2014-11-27 18:14:52,977 : INFO : #00000160: NDCG@10 (training):    0.44530525  |  (validation):    0.45052900
        2014-11-27 18:14:52,977 : INFO : Stopping early since no improvement on validation queries has been observed for 100 iterations (since iteration 60)
        2014-11-27 18:14:52,977 : INFO : Final model performance (NDCG@10) on validation queries:  0.45404817
        2014-11-27 18:14:52,977 : INFO : Training of LambdaMART model has finished.
        2014-11-27 18:14:52,977 : INFO : ================================================================================
        2014-11-27 18:14:53,086 : INFO : NDCG@10 on the test queries: 0.46956525

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
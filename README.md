# SphereRE: Distinguishing Lexical Relations with Hyperspherical Relation Embeddings

### By Chengyu Wang (https://chywang.github.io)

**Introduction:** This software does supevised lexical relation classification based on word embeddings. The model learns representations of lexical relation triples by mapping them to the hyperspherical embedding space, where relation triples of different lexical relations are well separated. The types of lexical relations are classified based on hyperspherical relation embeddings and classifical word embeddings.

**Paper** 
1. Wang et al. SphereRE: Distinguishing Lexical Relations with Hyperspherical Relation Embeddings. ACL 2019


**APIs**

1. GenerateLabelMap.py: Establish the mappings between lexical relation types and class indices.

2. TrainProj.py: Train the linear projection models for each lexical relation type.

3. TrainInitialLR.py: Train the logisitic regression classifier for lexical relation classification in the intial stage.

4. PredictInitLabelProb.py: Predict label probabilities for the testing data.

5. TransductiveSample.py: Sample data sequences for SphereRE learning.

6. RelationEmbedLearn.py: Learn SphereRE vectors.

7. TrainFinalNN.py: Train the final neural network for lexical relation classification and make predictions over the testing set.

**Notes**

1. Due to the large size of neural language models, we do not provide the fastText model in this project. Please use your own neural language model instead, if you would like to try the algorithm over your datasets.

2. The example dataset is BLESS. Other datasets mentioned in the paper are also publicly available. 


**Dependencies**

The main Python packages that we use in this project include but are not limited to:

1. gensim: 2.0.0
2. numpy: 1.15.4
3. scikit-learn: 0.18.1

The codes can run properly under the packages of other versions as well.


**More Notes on the Algorithm** 

This is a slightly updated version of the algorithm of our ACL 2019 paper. The produced results can be a little bit higher than the results reported in the original paper.

**Citations**

If you find this software useful for your research, please cite the following paper.

> @inproceedings{acl2019,<br/>
&emsp;&emsp; author    = {Chengyu Wang and Xiaofeng He and Aoying Zhou},<br/>
&emsp;&emsp; title     = {SphereRE: Distinguishing Lexical Relations with Hyperspherical Relation Embeddings},<br/>
&emsp;&emsp; booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},<br/>
&emsp;&emsp; pages     = {1727--1737},<br/>
&emsp;&emsp; year      = {2019}<br/>
}


More research works can be found here: https://chywang.github.io.






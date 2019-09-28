## BM25 search system

### How to run:
In project directory:
0. **make sure that dataset csv-file pulled correctly!** It is rather huge therefore there
is a possibility that something could happen during pushing. If you've got `ReadingDataError`,
then, please, get a dataset csv by yourself 
[here](https://www.kaggle.com/loopdigga/quora-question-pairs-russian) and 
put it in a project directory.
1. run `pip3 install -r requirement.txt`
2. run `python3 bm25_data.py`

### Conclusions

1. Matrix search runs much faster than iterative one more that in ~10 times.
2. Metrics in top-5 version depend on size of dataset an values are not high really, 
~50% accuracy.
3. While changing BM* version, metrics do not change dramatically.
##SEMANTIC SEARCH

Before using the program, please:
1. Download `simple_elmo.zip` from course chat and put files in `elmo` directory
in project root.
2. Download Word2Vec from [here](https://rusvectores.org/en/models/) and put it
into `w2v` directory in project root.
3. run `pip3 install -r requirements.txt`

**Please, make sure your tensorflow version is < 2.0, otherwise `Elmo`
will not run!**

Finally, you can run `python3 main_search.py`

### Conclusions

Word2Vec search is the fastest and shows the highest quality on metrics.
It takes in ~20 times less time for indexing than Elmo and in ~5 times less than
BM25. The quality is in ~2 times higher than Elmo and ~0.15 higher than BM25.
I have no idea why Elmo works so bad.
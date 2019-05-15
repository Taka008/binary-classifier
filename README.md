## binary-classifier
### Binary classifier by Machine learning method
- This can be used for binary classification such as True/False and Positive/Negative.
- As classifier, support vector machine and random forest are used.
### Input Format
- It is assumed that an input file is like the exmaple below.

```
1    I like this movie.
-1     He hates vegetables.
...
```
### How to Run
- It is optional whether you specify the PARALLEL_NUMBER (default is 1).
```
$ python svm.py --data-dir DATA_DIR
$ python random_forest.py --data-dir DATA_DIR --n-jobs PARALLEL_NUMBER 
```

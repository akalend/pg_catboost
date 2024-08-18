# pg_catboost
*in development*

the Machine Learning module based on the [CatBoost] (https://catboost.io) 


```sql
CREATE  [ CLASSIFICATION | REGRESSION | RANKING ]  MODEL model_name
  (  
     [TARGET fieldName ]
     [ITERATIONS itarations],
     [DEPTH   dept],
     [L2 REGULARIZATION l2_leaf_reg],
     [REGULARIZATION  model_size_re],
     [RMSN rmsn],
     [LOSS FUNCTION  [LOGLOSS | CROSSEENTROPY]],
     [TREE COUNT tree_count],
     [USE BEST MODEL],
     [CATEGORIAL FEATUTES 'fieldName_1, fieldName_2, ...'],
     [RANDOM SEED random_seed],
     [LEARNING RATE learning_rate]

  ) AS query;

```

## Parameters

*model_name* - the name of model, use for internal


raining and applying models for the classification problems. Provides compatibility with the scikit-learn tools.

The default optimized objective depends on various conditions:

    Logloss — The target has only two different values or the target_border parameter is not None.
    MultiClass — The target has more than two different values and the border_count parameter is None.



type of model
#### CLASSIFICATION 
short description of model classification

#### REGRESSION - 
short description 

#### RANKING - 
short description 


### TARGET fieldName
the fieldname of target column data,
If the target variable is not specified, the first field is used

### ITERATIONS [iterations](https://catboost.ai/en/docs/references/training-parameters/common#iterations)
The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.

*Type* int
*Default value* 1000

### DEPTH [depth](https://catboost.ai/en/docs/references/training-parameters/common#depth)

Depth of the trees.

The range of supported values depends on the processing unit type and the type of the selected loss function:

    CPU — Any integer up to  16.

    GPU — Any integer up to 8 for pairwise modes (YetiRank, PairLogitPairwise, and QueryCrossEntropy), and up to 16 for all other loss functions.

*Type* int

*Default value* 6 






## EXAMPLE:
```sql
 CREATE CLASSIFICATION MODEL adult (
 	LOSS FUNCTION Logloss,
 	ITERATIONS 100,
 	RANDOM SEED 42,
    learning rate 0.4234185321620083,
    DEPTH 5,
    L2 REGULARIZATION 9.464266235679002,
    TARGET income,
    CATEGORIAL FEATUTES 'workclass,education,marital_status', 'occupation,relationship,race,sex,native_country'
 	)
    AS SELECT * FROM adult;
```


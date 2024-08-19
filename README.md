# pg_catboost
*in development*

the Machine Learning module based on the [CatBoost](https://catboost.io) 



## CREATE MODEL


```sql
CREATE  [ CLASSIFICATION | REGRESSION | RANKING ]  MODEL model_name
  (  
     [TARGET fieldName ]
     [SPLIT split]
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
     [LEARNING RATE learning_rate],
     [EVAL METRIC metrics ]]

  ) AS query;

```

## Parameters

*model_name* - the name of model, use for internal


raining and applying models for the classification problems. Provides compatibility with the scikit-learn tools.

The default optimized objective depends on various conditions:

- Logloss — The target has only two different values or the target_border parameter is not None.
- MultiClass — The target has more than two different values and the border_count parameter is None.



type of model
#### CLASSIFICATION 
short description of model classification

#### REGRESSION - 
short description 

#### RANKING - 
short description 


#### TARGET fieldName
The fieldname of target column data,

If the target variable is not specified, the first field is used

#### SPLIT split
Part of the dataset for test analysis. The max value of split is 0.5

*Type* float

*Default value* 0.2


#### ITERATIONS [iterations](https://catboost.ai/en/docs/references/training-parameters/common#iterations)
The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.

*Type* int

*Default value* 1000

#### DEPTH [depth](https://catboost.ai/en/docs/references/training-parameters/common#depth)
Depth of the trees.

The range of supported values depends on the processing unit type and the type of the selected loss function:

- CPU — Any integer up to  16.

- GPU — Any integer up to 8 for pairwise modes (YetiRank, PairLogitPairwise, and QueryCrossEntropy), and up to 16 for all other loss functions.

*Type* int

*Default value* 6 


#### L2 REGULARIZATION [l2_leaf_reg](https://catboost.ai/en/docs/references/training-parameters/common#l2_leaf_reg)
Coefficient at the L2 regularization term of the cost function.

Any positive value is allowed.

*Type* float

*Default value* 3.0

#### REGULARIZATION  [model_size_re](https://catboost.ai/en/docs/references/model-size-reg)
This parameter influences the model size if training data has categorical features.

The information regarding categorical features makes a great contribution to the final size of the model. The mapping from the categorical feature value hash to some statistic values is stored for each categorical feature that is used in the model. The size of this mapping for a particular feature depends on the number of unique values that this feature takes.

#### RMSN 
???

#### LOSS FUNCTION  
The metric to use in training. The specified value also determines the machine learning problem to solve. Some metrics support optional parameters.

*Default* 
- *LogLoss* for binary
- *CrossEntropy* for multi classification

### TREE COUNT tree_count
Max number of trees in the model.

### USE BEST MODEL
Choose best model from all iterations.

*Default value* False


### CATEGORIAL FEATUTES 'fieldName_1, fieldName_2, ...'
The list of field name of categorical feature. Default: all TEXT,  CHAR or VARCHAR fields



### RANDOM SEED random_seed
The random seed used for training.

*Type* int

### LEARNING RATE learning_rate]
The learning rate used for training.

*Type* float

### EVAL METRIC metrics
the list of metrics. [Support metrics](https://catboost.ai/en/docs/references/custom-metric__supported-metrics)




## EXAMPLE:
```sql
\d adult
                        Table "public.adult"
     Column     |       Type       | Collation | Nullable | Default 
----------------+------------------+-----------+----------+---------
 age            | double precision |           |          | 
 workclass      | text             |           |          | 
 fnlwgt         | double precision |           |          | 
 education      | text             |           |          | 
 education_num  | double precision |           |          | 
 marital_status | text             |           |          | 
 occupation     | text             |           |          | 
 relationship   | text             |           |          | 
 race           | text             |           |          | 
 sex            | text             |           |          | 
 capital_gain   | double precision |           |          | 
 capital_loss   | double precision |           |          | 
 hours_per_week | double precision |           |          | 
 native_country | text             |           |          | 
 income         | text             |           |          | 



 CREATE CLASSIFICATION MODEL adult
 (
	LOSS FUNCTION Logloss,
	ITERATIONS 100,
	RANDOM SEED 42,
	LEARNING RATE 0.4234185321620083,
	DEPTH 5,
	L2 REGULARIZATION 9.464266235679002,
	TARGET income,
	CATEGORIAL FEATUTES 'workclass,education,marital_status', 'occupation,relationship,race,sex,native_country'
	)
	AS
	SELECT * FROM adult;
```




## SHOW MODEL


```sql

SHOW MODEL model_name;

```

Show model information;


## PRODICT MODEL


```sql

PREDICT model_name AS query

```



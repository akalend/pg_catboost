CREATE TABLE IF NOT EXISTS ml_model (
    name        Name PRIMARY KEY,
    model_file  text,
    model_type  char(1),
    acc         real,
    info        text,
    args        text,
    data        bytea
    );


CREATE OR REPLACE FUNCTION ml_predict_internal(
    model text,
    tablename text,
    join_field text DEFAULT 'row',
    isQuery bool  DEFAULT FALSE,
    OUT index text,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'MODULE_PATHNAME','ml_predict_dataset_inner'
LANGUAGE  C STRICT PARALLEL RESTRICTED;


CREATE OR REPLACE FUNCTION ml_test(name Name)
RETURNS text
AS 'catboost','ml_test'
LANGUAGE  C STRICT;


CREATE OR REPLACE FUNCTION ml_predict(
    model text,
    tablename text,
    join_field text DEFAULT 'row',
    OUT index text,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'SELECT ml_predict_internal($1,$2,$3, FALSE);'
LANGUAGE SQL VOLATILE STRICT;


CREATE OR REPLACE FUNCTION ml_predict_query(
    model text,
    query text,
    join_field text DEFAULT 'row',
    OUT index text,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'SELECT ml_predict_internal($1,$2,$3, TRUE);'
LANGUAGE SQL VOLATILE STRICT;



CREATE OR REPLACE FUNCTION ml_learn(
                                name text,          -- name of model
                                model_type int,     -- type of model
                                options text,       -- options
                                table_name text,    -- table for dataset
                                filename text       -- name of file for save model    
                                )
    RETURNS float AS 
$$ 
    import json
    from  catboost import CatBoostClassifier
    from  catboost import CatBoostRegressor
    from  catboost import CatBoostRanker
    from catboost import Pool
    import os.path

    class_names = []
    opt_dict = json.loads(options) 
    query = "SELECT * FROM " + table_name
    rows = plpy.execute(query)  # line 17

    columns = rows.colnames()
    nrows = rows.nrows()

    target_name = ''
    if ('target' in opt_dict):
        target_name = opt_dict['target']
    else:
        target_name = next(iter(columns), None)


    cat_features_idx = []
    i = 0
    bool_types = []
    for col_type in rows.coltypes():
        if col_type in (19,25,1042,1043):
            cat_features_idx.append(i)
        if col_type == 16:
            bool_types.append(i)
        i += 1

    is_not_ignore = False
    ignored_names = None
    if ('ignored' in opt_dict):
        ignored_names = list(opt_dict['ignored'])
        is_not_ignore = True

    ignored_idx = []
    if is_not_ignore:
        for i in range(0,len(columns)):
            if columns[i] in ignored_names:
                ignored_idx.append(i)


    cat_features = []
    for idx in cat_features_idx:
        if idx not in ignored_idx:
            cat_features.append(columns[idx])

    if target_name in cat_features:
        cat_features.remove(target_name)

    if 'split' in opt_dict:
        split = opt_dict['split']
        del(opt_dict['split'])
        split = split / 100
    else:
        split = 0.20

    if (split > 0.5 or split == 0):
        plpy.error("split value must be in 1-50 interval")

    is_class_name = False
    if ('classes' in opt_dict):
        class_names = opt_dict['classes']
    else:
        is_class_name = True


    use_columns = []
    for it in columns:
        if is_not_ignore and it in ignored_names:
            continue
        use_columns.append(it)

    drop_clolumn_num = columns.index(target_name)
    columns.pop(drop_clolumn_num)

    X = []
    y = []

    X_test = []
    y_test = []
    
    train_rows = nrows * split
    counter = 0
    is_test = False
    for row in rows:
        if counter > train_rows:
            is_test = True
        counter += 1
        if row[target_name] is None:
            continue

        ###### get classes ######
        if is_class_name:
            if type(row[target_name] ) == type(True):
                class_names = [0,1]
                is_class_name = False
            elif not row[target_name] in class_names:
                    class_names.append(row[target_name])


        append_values = []
        for col in row:
            if col not in use_columns:
                    continue
            if row[col] is None:
                row[col] = 'Nan'
            val = row[col]
            if type(row[col] ) == type(True):
                row[col] = int(row[col])
            if col == target_name:
                continue
            append_values.append(row[col])

        
        row_values = row.values()
        target = row[target_name]
 
        if is_test:
            y_test.append(target)
            X_test.append(append_values)
        else:        
            y.append(target)
            X.append(append_values)
    
    ###### options ######


    drop_clolumn_num = use_columns.index(target_name)
    use_columns.pop(drop_clolumn_num)

    depth = opt_dict.get('rept')
    loss_function = opt_dict.get('loss_function')
    eval_metric = opt_dict.get('eval_metric')
    iterations = opt_dict.get('iterations')
    random_seed = opt_dict.get('random_seed')
    learning_rate = opt_dict.get('lerning')
    l2_leaf_reg = opt_dict.get('l2')

    ###### model ######

    pool = Pool(X, label=y, feature_names=use_columns, cat_features=cat_features)
    

    if model_type == 1:             # classification
        if loss_function is None:
            loss_function = 'Logloss'
        model = CatBoostClassifier(
            loss_function=loss_function,
            eval_metric=eval_metric,
            iterations=iterations,
            random_seed=random_seed,
            learning_rate=learning_rate,
            class_names=class_names,
            # ignored_features=ignored_names,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg )
    elif model_type == 2:             # regresssion
        if loss_function is None:
            loss_function = 'RMSE'
        model = CatBoostRegressor(
            loss_function=loss_function,
            eval_metric=eval_metric,
            iterations=iterations,
            random_seed=random_seed,
            learning_rate=learning_rate,
            # ignored_features=ignored_names,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg )

    elif model_type == 3:             # ranking
        model = CatBoostRanker(
            loss_function=loss_function,
            eval_metric=eval_metric,
            iterations=iterations,
            random_seed=random_seed,
            learning_rate=learning_rate,
            class_names=class_names,
            # ignored_features=ignored_names,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg )
    else:
        plpy.error("undefined model type", model_type)    

    model.fit(pool)
    score = model.score(X_test,y_test)

    model.save_model(filename)

    return score;
$$ LANGUAGE plpython3u;

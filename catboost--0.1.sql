CREATE TABLE IF NOT EXISTS ml_model (
    name        Name PRIMARY KEY,
    model_file  text,
    model_type  char(1),
    acc         real,
    info        text,
    args        text
    );


CREATE OR REPLACE FUNCTION ml_learn_classifier(
                                name text,          -- name of model
                                options jsonb,      -- options
                                table_name text          -- table for dataset 
                                )     
    RETURNS float AS 
$$ 
    import json
    from  catboost import CatBoostClassifier
    from catboost import Pool
    import os.path
    # from sklearn import metrics

    opt_dict = json.loads(options) 

    query = "SELECT * FROM " + table_name
    rows = plpy.execute(query)

    columns = rows.colnames()
    nrows = rows.nrows()
    # plpy.warning("rows is ", nrows)

    target_name = ''
    if ('target' in opt_dict):
        target_name = opt_dict['target']
    else:
        target_name = next(iter(columns), None)

    # plpy.warning( 'target:', target_name)

  
    cat_features_idx = []
    i = 0
    bool_types = []
    for col_type in rows.coltypes():
        if col_type in (19,25,1042,1043):
            cat_features_idx.append(i)
        if col_type == 16:
            bool_types.append(i)
        i += 1

    cat_features = []
    for idx in cat_features_idx:
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
        plpy.error("split value must be in 0-0.5 interval")

    is_class_name = False
    if ('classes' in opt_dict):
        class_names = opt_dict['classes']
    else:
        is_class_name = True

    drop_clolumn_num = columns.index(target_name)
    columns.pop(drop_clolumn_num)

    X = []
    y = []

    X_test = []
    y_test = []
    
    class_names = []
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

        for col in row:
            if row[col] is None:
                row[col] = 'Nan'
            val = row[col]
            if type(row[col] ) == type(True):
                row[col] = int(row[col])

        append_values = row.values()
        # target = row[target_name]
        
        
        append_values = list(append_values)
        target = append_values.pop(drop_clolumn_num)
        if is_test:
            y_test.append(target)
            X_test.append(append_values)
        else:        
            y.append(target)
            X.append(append_values)

    # plpy.notice('class_names', class_names)
    # # plpy.warning('class_names',class_names);
    # plpy.warning('options', opt_dict);
    
    ###### options ######

    depth = opt_dict.get('rept')
    loss_function = opt_dict.get('loss_function')
    eval_metric = opt_dict.get('eval_metric')
    iterations = opt_dict.get('iterations')
    random_seed = opt_dict.get('random_seed')
    learning_rate = opt_dict.get('lerning')
    l2_leaf_reg = opt_dict.get('l2')

    ###### model ######

    pool = Pool(X, label=y, feature_names=columns, cat_features=cat_features)
    model = CatBoostClassifier(
        loss_function=loss_function,
        eval_metric=eval_metric,
        iterations=iterations,
        random_seed=random_seed,
        learning_rate=learning_rate,
        class_names=class_names,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg )

    model.fit(pool)
    score = model.score(X_test,y_test)

    ## get model path
    rows = plpy.execute("SHOW ml.model_path")
    val = rows[0].values()
    nrows = rows.nrows()
    vals = list(val)
    path = vals[0]

    modelFile = name + '.sql.cbm'
    if len(path) > 0:
        modelFile = os.path.join(path,modelFile)
    
    model.save_model(modelFile)
    ## plpy.warning("save to", modelFile)
    return score;

$$ LANGUAGE plpython3u;

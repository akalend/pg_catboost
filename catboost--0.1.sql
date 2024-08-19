CREATE OR REPLACE FUNCTION ml_learn_classifier(
                                name text,          -- name of model
                                options jsonb,      -- options
                                query text          -- query for dataset 
                                )     
    RETURNS float AS 
$$ 
    import json
    from  catboost import CatBoostClassifier
    from catboost import Pool
    # from sklearn import metrics

    plpy.warning('name',name);
    opt_dict = json.loads(options) 


    ## query = query + " ORDER BY random()"

    plpy.warning(query);
    rows = plpy.execute(query)

    columns = rows.colnames()
    nrows = rows.nrows()
    plpy.warning("rows is ", nrows)

    target_name = ''
    if ('TARGET' in opt_dict):
        target_name = opt_dict['TARGET']
    else:
        target_name = next(iter(columns), None)

    plpy.warning( 'target:', target_name)

  
    cat_features_idx = []
    i = 0
    for type in rows.coltypes():
        if type in (19,25,1042,1043):
            cat_features_idx.append(i)
        i += 1

    cat_features = []
    for idx in cat_features_idx:
        cat_features.append(columns[idx])

    cat_features.remove(target_name)
    ## split
    if 'SPLIT' in opt_dict:
        split = opt_dict['SPLIT']
        del(opt_dict['SPLIT'])
    else:
        split = 0.20

    if (split > 0.5 or split == 0):
        plpy.error("split value must be in 0-0.5 interval")

    plpy.warning("split ", split)

    is_class_name = False
    if ('classes' in opt_dict):
        class_names = opt_dict['classes']
    else:
        is_class_name = True

    drop_clolumn_num = columns.index(target_name)
    plpy.notice( "target_idx:", drop_clolumn_num )
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

        if is_class_name:
            if len(class_names) == 0:
                class_names.append(row[target_name])
            if len(class_names) == 1:
                if class_names[0] != row[target_name]:
                    class_names.append(row[target_name])
                    is_class_name = False

        for col in row:
            if row[col] is None:
                row[col] = 'Nan'

        append_values = row.values()

        # target = row[target_name]
        
        
        append_values = list(append_values)
        target = append_values.pop(drop_clolumn_num)
        if is_test:
            y_test.append(target)
            X_test.append(append_values)
            # plpy.warning('train', append_values);
        else:        
            y.append(target)
            X.append(append_values)
            # plpy.warning('test', append_values);


    # plpy.warning('class_names',class_names);
    plpy.warning('options', opt_dict);
    
    ###### options ######

    depth = opt_dict.get('DEPTH')
    loss_function = opt_dict.get('LOSS_FUNCTION')
    eval_metric = opt_dict.get('EVAL_METRIC')
    iterations = opt_dict.get('ITERATIONS')
    random_seed = opt_dict.get('RANDOM_SEED')
    learning_rate = opt_dict.get('LEARNING_RATE')
    l2_leaf_reg = opt_dict.get('L2_REGULARIZATION')

    ###### model ######

    pool = Pool(X, label=y, feature_names=columns, cat_features=cat_features)
    model = CatBoostClassifier(
        loss_function=loss_function,
        eval_metric=eval_metric,
        iterations=iterations,
        random_seed=random_seed,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg )

    model.fit(pool)
    score = model.score(X_test,y_test)

    modelFile = name + '.sql.cbm'
    model.save_model(modelFile)
    plpy.warning("save to", modelFile)
    return score;

$$ LANGUAGE plpython3u;


CREATE OR REPLACE FUNCTION ml_parse(text)
RETURNS integer
AS 'MODULE_PATHNAME'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
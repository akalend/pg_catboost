CREATE TABLE IF NOT EXISTS ml_model (
    name        Name PRIMARY KEY,
    model_file  text,
    model_type  char(1),
    acc         real,
    info        text,
    args        text
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

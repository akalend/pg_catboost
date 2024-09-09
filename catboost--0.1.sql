CREATE TABLE IF NOT EXISTS ml_model (
    name        Name PRIMARY KEY,
    model_file  text,
    model_type  char(1),
    acc         real,
    info        text,
    args        text
    );


CREATE OR REPLACE FUNCTION ml_predict(
    modelname text,
    tablename text,
    keyfield text
) RETURNS text
AS 'MODULE_PATHNAME','ml_predict'
LANGUAGE C STRICT PARALLEL RESTRICTED;

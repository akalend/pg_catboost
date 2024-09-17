#ifndef ML_CATBOOST_H
#define ML_CATBOOST_H


#include "executor/spi.h"

enum model_type_t {
    MODEL_NONE,
    MODEL_CLASSIFICATION,
    MODEL_REGRESSION,
    MODEL_RANKING,
};


#define ML_TABLE_NAME "ml_model"
#define ML_TABLE_PKEY "ml_model_pkey"

typedef struct FormData_model
{
    NameData name;
    text* file;
    char type;
    float acc;
    text* info;
    text* args;
} FormData_model;

typedef FormData_model* Form_model;

typedef enum Anum_model
{
    Anum_ml_name = 1,
    Anum_ml_model_file,
    Anum_ml_model_type,
    Anum_ml_model_acc,
    Anum_ml_model_info,
    Anum_ml_model_args,
    _Anum_ml_max,
} Anum_model;

#define Natts_model (_Anum_ml_max - 1)

typedef enum Anum_ml_name_idx
{
    Anum_ml_name_idx_name = 1,
    _Anum_ml_name_idx_max,
} Anum_ml_name_idx;

#define Natts_ml_name_idx (_Anum_ml_name_idx_max - 1)

typedef struct ArrayDatum {
    int count;
    Datum *elements;
} ArrayDatum;

typedef struct MLmodelData
{
    ModelCalcerHandle  *modelHandle;
    SPITupleTable      *spi_tuptable;
    TupleDesc           spi_tupdesc;
    char***             modelClasses;
    int8               *iscategory;
    float              *row_fvalues;
    char*              *row_cvalues;
    char*              *row_tvalues;
    char*               cat_value_buffer;
    char*               txt_value_buffer;
    double             *result_pa;
    double             *result_exp;
    char*               keyField;
    char*               modelType;
    int64               current;
    size_t              cat_count;
    size_t              num_count;
    size_t              txt_count;
    size_t              attCount;
    size_t              dimension;
    ArrayDatum          cat_fields;
    ArrayDatum          text_fields;
} MLmodelData;

#define MLmodel MLmodelData*

typedef struct SpiData
{
    SPITupleTable      *tuple_table;
    TupleDesc           tuple_desc;
    int                 current;
} SpiData;

#define SpiInfo SpiData*

#define FEATURES_BUFSIZE    1024
#define FIELDLEN            64
#define TXT_FIELDLEN        1024
#define PAGESIZE            8192
#define QNaN                0x7fffffff
#define MAXDIGIT            12
#define FAIL                -1

#define ModelGetFieldName(i)  model->spi_tupdesc->attrs[i].attname.data


#endif                          /* ML_CATBOOST_H */
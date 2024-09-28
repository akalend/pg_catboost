/*
 * contrib/catboost/catboost.c
 *
 * Alexandre Kalendarev <akalend@mail.ru>
 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>
// #include <stdbool.h>

#include "postgres.h"
#include "fmgr.h"
#include "c.h"
#include "c_api.h"
#include "funcapi.h"
#include "catboost.h"

// #include "access/htup.h"
// #include "access/slru.h"
// #include "access/amapi.h"
// #include "access/heapam.h"
// #include "access/htup_details.h"
// #include "access/table.h"
// #include "access/tableam.h"
// #include "catalog/indexing.h"
// #include "executor/executor.h"
#include "executor/spi.h"
// #include "nodes/parsenodes.h"
// #include "nodes/primnodes.h"
// #include "nodes/pg_list.h"
// #include "parser/analyze.h"
// #include "tcop/utility.h"
#include "utils/builtins.h"
#include "utils/errcodes.h"
#include "utils/json.h"
#include "utils/jsonb.h"
#include "utils/jsonfuncs.h"
#include "utils/guc.h"
// #include "utils/fmgrprotos.h"
// #include "utils/fmgroids.h"
// #include "utils/memutils.h"
// #include "utils/numeric.h"
// #include "utils/rel.h"
// #include "utils/snapmgr.h"
// #include "utils/varlena.h"




/* Function declarations */
static bool check_model_path(char **newval, void **extra, GucSource source);
static bool checkInArray(char* name, char **features, int featureCount);
static const char* getModelParms(ModelCalcerHandle* modelHandle);
static char* getModelType(ModelCalcerHandle* modelHandle, const char* info);
static char** getModelFeatures(ModelCalcerHandle *modelHandle, size_t *featureCount);
static char*** getModelClasses(ModelCalcerHandle* modelHandle, const char* info);
static void LoadModel(text  *filename, ModelCalcerHandle** modelHandle);
static Datum PredictGetDatum(char* id, int64 row_no, float8 predict, char* className,
                TupleDesc tupleDescriptor);
static bool pstrcasecmp(char  *s1, char  *s2);

static double sigmoid(double x);
void _PG_init(void);

static char* model_path = "";

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(ml_predict_dataset_inner);
PG_FUNCTION_INFO_V1(ml_predict_tmp);

static double
sigmoid(double x) {
    return 1. / (1. + exp(-x));
}


/*
* case compare column name and model feature name
* and replace symbol '-' to '_'
* as the postgers can't use symbol '-' in column name
*/
static bool
pstrcasecmp(char  *s1, char  *s2)
{
    char *p1,*p2, pp1, pp2;
    p1=s1;
    p2=s2;

    while (*p1 && *p2)
    {
        if (isalpha(*p1))
            pp1 = tolower(*p1);
        else
            if (*p1 == '-')
                pp1 = '_';
            else
                pp1 = *p1;

        if (isalpha(*p2))
            pp2 = tolower(*p2);
        else
            if (*p2 == '-')
                pp2 = '_';
            else
                pp2 = *p2;

        if (pp1 == '_' && strcasecmp(p1+1,"id") == 0 && strcasecmp(p2,"ID") == 0)
            return true;

        if (pp2 == '_' && strcasecmp(p2+1,"id") == 0 && strcasecmp(p1,"ID") == 0)
            return true;

        if (pp1!= pp2)
        {
            return false;
        }

        p1 ++;
        p2 ++;
    }

    if (*p1 !=*p2)
        return false;

    return true;
}


static bool
checkInArray(char* name, char **features, int featureCount)
{
    int i;
    for(i=0; i < featureCount; i++)
    {
        if (!features[i])
            elog(ERROR, "the feature %d is NULL", i);
        if ( pstrcasecmp(features[i], name) ){
            return true;
        }
    }
    return false;
}


/*
* check filename and load CatBosot model
*/
static void
LoadModel(text  *filename, ModelCalcerHandle** modelHandle)
{
    struct stat buf;
    StringInfoData sbuf;
    char slash[2] = "/\0";

    const char  *filename_str = text_to_cstring(filename);

    initStringInfo(&sbuf);
    if (strstr(filename_str, slash) == NULL)
    {
        int len = strlen( model_path);
        if (model_path[len-1] == '/')
            appendStringInfo(&sbuf, "%s%s", model_path, filename_str);
        else
            appendStringInfo(&sbuf, "%s/%s", model_path, filename_str);

        filename_str = sbuf.data;
    }


    if (stat(filename_str, &buf) == FAIL)
    {
        int         err = errno;
        elog(ERROR, "file %s has error: %d:%s",filename_str, err, strerror(err));
    }

    *modelHandle = ModelCalcerCreate();
    if (!LoadFullModelFromFile(*modelHandle, filename_str))
    {
        elog(ERROR, "LoadFullModelFromFile error message: %s\n", GetErrorString());
    }
    resetStringInfo(&sbuf);
    pfree(sbuf.data);
}


static const char* getModelParms(ModelCalcerHandle* modelHandle)
{
    const char* info = GetModelInfoValue(modelHandle, "params", 6); // strlen("parms")
    if( !info )
    {
        return NULL;
    }
    return info;
}


static char***
getModelClasses(ModelCalcerHandle* modelHandle, const char* info)
{
    char ***res = NULL;
    StringInfoData buf;
    Jsonb* j;
    SPITupleTable *tuptable;
    int ret;
    bool is_null = false;
    Datum classes;
    TupleDesc tupdesc;

    if( !info )
    {
        return NULL;
    }

    initStringInfo(&buf);
    appendStringInfo(&buf,
        "SELECT '%s'::jsonb #> '{data_processing_options,class_names}';",
        info);

    tuptable = SPI_tuptable;
    ret = SPI_exec(buf.data, 0);
    tuptable = SPI_tuptable;
    if (ret < 1 || tuptable == NULL)
    {
        elog(ERROR, "Query errorcode=%d", ret);
    }

    tupdesc = tuptable->tupdesc;

    if (0 == strcmp(SPI_getvalue(tuptable->vals[0], tupdesc, 1), "[]"))
    {
        return NULL;
    }

    classes = SPI_getbinval(tuptable->vals[0], tupdesc, 1,&is_null);

    if (is_null){
        elog(WARNING, "result is NULL");
        return NULL;
    }

    j = DatumGetJsonbP(classes);

    if(JB_ROOT_IS_ARRAY(j))
    {
        JsonbIterator *it;
        JsonbIteratorToken type;
        JsonbValue  jb;
        char*** p;

        res = (char***) palloc( sizeof(char*) * (
                getJsonbLength((const JsonbContainer*) j,0) + 1)
              );
        p = res;
        it = JsonbIteratorInit(&j->root);

        while ((type = JsonbIteratorNext(&it, &jb, false))
               != WJB_DONE)
        {
            if (WJB_ELEM == type){
                if(jb.type == jbvString)
                {
                  *p = (char**)pnstrdup(jb.val.string.val, jb.val.string.len);
                   p++;
                   continue;
                }
                if(jb.type == jbvNumeric)
                {
                    Numeric num;
                    num = jb.val.numeric;

                    *p = (char**)pstrdup(DatumGetCString(
                                            DirectFunctionCall1(numeric_out,
                                              NumericGetDatum(num))));
                    p++;
                   continue;
                }
                elog(ERROR, "undefined jsonb type num=%d",jb.type);
            }
        }
        *p = NULL;

        return res;
    }

    return NULL;
}


static char*
getModelType(ModelCalcerHandle* modelHandle, const char* info)
{
    StringInfoData buf;
    TupleDesc tupdesc;
    SPITupleTable *tuptable;
    int ret;

    initStringInfo(&buf);

    if( !info )
    {
        appendStringInfo(&buf, "NULL");
        return buf.data;
    }

    appendStringInfo(&buf,
        "SELECT '%s'::jsonb #> '{loss_function,type}';", info);

    tuptable = SPI_tuptable;
    ret = SPI_exec(buf.data, 0);
    tuptable = SPI_tuptable;

    if (ret < 1 || tuptable == NULL)
    {
        elog(ERROR, "Query errorcode=%d", ret);
    }

    tupdesc = tuptable->tupdesc;

    resetStringInfo(&buf);
    appendStringInfo(&buf, "%s", SPI_getvalue(tuptable->vals[0], tupdesc, 1));

    return buf.data;
}


/*
*  get array names Model features
*/
static char**
getModelFeatures(ModelCalcerHandle *modelHandle, size_t *featureCount)
{
    char** featureName = palloc(FEATURES_BUFSIZE);

    bool rc = GetModelUsedFeaturesNames(modelHandle, &featureName, featureCount);

    if (!rc)
    {
        elog(ERROR,"get model feature name error: %s", GetErrorString());
    }
    return featureName;
}


static Datum
PredictGetDatum(char* id, int64 row_no, float8 predict, char* className,
                TupleDesc tupleDescriptor)
{
    Datum values[3];
    bool isNulls[3];
    HeapTuple htuple;

    memset(values, 0, sizeof(values));
    memset(isNulls, false, sizeof(isNulls));


    if ( strncmp("row", id, 3) )
    {
        values[0] = CStringGetTextDatum(id);
    }
    else
    {
        char row_str[MAXDIGIT];
        pg_itoa(row_no, row_str);
        values[0] = CStringGetTextDatum(row_str);
    }

    values[1] = Float8GetDatum(predict);

    if (className) {
        values[2] =  CStringGetTextDatum(className);
    }
    else
    {
        isNulls[2] = true;
        values[2] =  CStringGetTextDatum("");
    }

    htuple = heap_form_tuple(tupleDescriptor, values, isNulls);
    return (Datum) HeapTupleGetDatum(htuple);
}


Datum
ml_predict_tmp(PG_FUNCTION_ARGS)
{
    FuncCallContext *functionContext = NULL;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldContext;
        TupleDesc tupleDescriptor;
        ArrayDatum  cat_fields = {0, NULL};
        StringInfoData buf;
        ModelCalcerHandle* modelHandle;
        // SPITupleTable   *spi_tuptable;
        const char*     model_info;
        const int resultColumnCount = 3;
        size_t          model_float_feature_count;
        size_t          model_cat_feature_count ;
        size_t          model_dimension;
        size_t          featureCount = 0;
        char            **features;
        size_t          cat_feature_counter=0;
        size_t          feature_counter=0;
        int             i;
        MLmodel         model;
        text            *filename;
        char            *key_field;
        int             res;
        bool            function_type = PG_GETARG_BOOL(3);
        char            **arr_cat_fields = NULL;

        /* create a function context for cross-call persistence */
        functionContext = SRF_FIRSTCALL_INIT();

        filename = PG_GETARG_TEXT_PP(0);


        LoadModel(filename, &modelHandle);

        /* switch to memory context appropriate for multiple function calls */
        oldContext = MemoryContextSwitchTo(
            functionContext->multi_call_memory_ctx);

        initStringInfo(&buf);

        features = getModelFeatures(modelHandle, &featureCount);

        model_cat_feature_count = (size_t)GetCatFeaturesCount(modelHandle);

        if (model_cat_feature_count)
        {
            size_t cat_feature_count;
            size_t* indices;

            if (!GetCatFeatureIndices(modelHandle, &indices, &cat_feature_count))
            {
                elog(ERROR, "cat feature %s", GetErrorString());
            }

            arr_cat_fields = palloc( sizeof(char*) * cat_feature_count);

            for (i=0; i < cat_feature_count; i++)
            {
                arr_cat_fields[i] = pstrdup(features[indices[i]]);
            }

            free(indices); // allocated by CatBoostModel::GetCatFeatureIndices()
        }

        key_field = text_to_cstring(PG_GETARG_TEXT_PP(2));

        model = (MLmodel) palloc0(sizeof(MLmodelData));

        model->modelHandle = modelHandle;
        model_info = getModelParms(model->modelHandle);
        SPI_connect();
        model->modelType = getModelType(model->modelHandle, model_info);
        model->modelClasses = getModelClasses(model->modelHandle, model_info);

        if (key_field)
            model->keyField = pstrdup(key_field);

        if (function_type) {
            char  *query = text_to_cstring(PG_GETARG_TEXT_PP(1));
            res = SPI_exec(query, 0);
        }
        else
        {
            char  *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));
            appendStringInfo(&buf, "SELECT * FROM %s;", tabname);
            res = SPI_exec(buf.data, 0);
        }
        if (res < 1 || SPI_tuptable == NULL)
        {
            elog(ERROR, "Query %s error", buf.data);
        }

        model->cat_fields = cat_fields;

        model->spi_tuptable = SPI_tuptable;
        model->spi_tupdesc  = SPI_tuptable->tupdesc;
        model->attCount = SPI_tuptable->tupdesc->natts;


        model->iscategory = palloc0( model->attCount * sizeof(int8));
        model_float_feature_count = (size_t)GetFloatFeaturesCount(model->modelHandle);
        model_dimension = (size_t)GetDimensionsCount(model->modelHandle);

        for(i=0; i < model->attCount; i++)
        {
            if(! checkInArray(ModelGetFieldName(i), features, featureCount))
            {
                model->iscategory[i] = -1;  // not in features
                continue;
            }
            if ( checkInArray(ModelGetFieldName(i), arr_cat_fields, model_cat_feature_count))
            {
                cat_feature_counter ++;
                model->iscategory[i] = 1;
            }
            else
            {
                feature_counter ++;
                model->iscategory[i] = 0;
            }
        }

         //  check model features
        if (feature_counter != model_float_feature_count)
        {
            elog(ERROR,
                "count of numeric features is not valid, must be %ld is %ld",
                model_float_feature_count, feature_counter
            );
        }

        if (cat_feature_counter != model_cat_feature_count)
        {
            elog(ERROR,
                "count of categocical features is not valid, must be %ld is %ld",
                model_cat_feature_count, cat_feature_counter
            );
        }

        model->current = 0;
        functionContext->user_fctx = model;
        functionContext->max_calls = SPI_processed;
        model->dimension = model_dimension;
        model->cat_count = cat_feature_counter;
        model->num_count = feature_counter;

        /*
         * This tuple descriptor must match the output parameters declared for
         * the function in pg_proc.
         */
        tupleDescriptor = CreateTemplateTupleDesc(resultColumnCount);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 1, key_field,
                           TEXTOID, -1, 0);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 2, "predict",
                           FLOAT8OID, -1, 0);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 3, "class",
                           TEXTOID, -1, 0);

        functionContext->tuple_desc =  BlessTupleDesc(tupleDescriptor);

        MemoryContextSwitchTo(oldContext);
    }

    functionContext = SRF_PERCALL_SETUP();

    if (  ((MLmodel) functionContext->user_fctx)->current < functionContext->max_calls)
    {
        char*   class;
        int     feature_counter = 0;
        int     cat_feature_counter = 0;
        MLmodel model = (MLmodel)functionContext->user_fctx;
        Datum   recordDatum;
        int j;
        char *p;
        char* yes = "yes";
        char* no = "no";
        HeapTuple   spi_tuple = ((MLmodel)functionContext->user_fctx)->spi_tuptable->vals[((MLmodel)functionContext->user_fctx)->current];
        int memsize = model->cat_count * sizeof(char) * FIELDLEN;
        char* key_field_value = NULL;

        model->row_fvalues = palloc0( model->num_count * sizeof(float));
        p = model->cat_value_buffer  = palloc0(memsize );
        model->row_cvalues = palloc0(model->cat_count * sizeof(char*));

        model->result_pa  = (double*) palloc( sizeof(double) * model->dimension);
        model->result_exp = (double*) palloc( sizeof(double) * model->dimension);

        for(j=0; j < model->attCount; j++)
        {
            char    *value;
            value = SPI_getvalue(spi_tuple, model-> spi_tupdesc, j+1);

            if (strcmp(model->keyField,  ModelGetFieldName(j)) == 0)
            {
                key_field_value = value;
            }

            if( model->iscategory[j] == -1) // not in features
                continue;

            if( model->iscategory[j] == 0)
            {
                if (value == 0 )
                {
                    model->row_fvalues[feature_counter] = QNaN;
                }
                else
                {
                    int res;
                    res  = sscanf(value, "%f", &model->row_fvalues[feature_counter]);
                    if(res < 1)
                    {
                        elog(WARNING,"error input j/cnt=%d/%d %f\n", j,
                            feature_counter, model->row_fvalues[feature_counter]);
                    }
                }

                feature_counter++;
            }

            if( model->iscategory[j] == 1)
            {
                if (!value)
                {
                    model->row_cvalues[cat_feature_counter] = pstrdup("NaN");
                    p += 3;
                }
                else
                {
                    model->row_cvalues[cat_feature_counter] = strcpy(p, value);
                    p += strlen(value);
                }
                cat_feature_counter++;
                *p = '\0';
                p++;
            }
        } // column

        if (!CalcModelPredictionSingle(model->modelHandle,
                    model->row_fvalues, feature_counter,
                    (const char** )model->row_cvalues, cat_feature_counter,
                    model->result_pa, model->dimension)
            )
        {
            StringInfoData str;
            initStringInfo(&str);
            for(j = 0; j < feature_counter; j++)
            {
                appendStringInfo(&str, "%f,", model->row_fvalues[j]);
            }
            elog( ERROR, "CalcModelPrediction error message: %s \nrow num=%ld",
                    GetErrorString(),  model->current );
        }

        if (strncmp("\"MultiClass\"", model->modelType, 12) == 0)
        {
            char  ***p;
            double max = 0., sm = 0.;
            int max_i = -1;
            char* out = model->keyField;

            for( j = 0; j < model->dimension; j ++)
            {
                model->result_exp[j] = exp(model->result_pa[j]);
                sm += model->result_exp[j];
            }
            for( j = 0; j < model->dimension; j ++)
            {
                model->result_exp[j] = model->result_exp[j] / sm;
                if (model->result_exp[j] > max){
                    max = model->result_exp[j];
                    max_i = j;
                }
            }

            p = model->modelClasses + max_i;

            if (key_field_value)
            {
                out = key_field_value;
            }


            recordDatum = PredictGetDatum(out, model->current, max, (char*)*p,
                            functionContext->tuple_desc);

        }
        else if (strcmp(model->modelType, "\"RMSE\"") == 0)
        {
            char* out = "";

            if (key_field_value)
            {
                out = key_field_value;
            }
            recordDatum = PredictGetDatum(out, model->current, model->result_pa[0], NULL,
                            functionContext->tuple_desc);

        }
        else if (strncmp("\"Logloss\"", model->modelType, 9) == 0)
        {
            double probability = sigmoid(model->result_pa[0]);
            char* out = model->keyField;
            int n = 0;
            if (probability > 0.5)
            {
                n = 1;
            }

            if (key_field_value)
            {
                out = key_field_value;
            }
            recordDatum = PredictGetDatum(out, model->current, probability,
                            (char*)*(model->modelClasses + n),
                            functionContext->tuple_desc);
        }
        else
        {
            double probability = sigmoid(model->result_pa[0]);
            char* out = model->keyField;
            if (probability > 0.5)
            {
                class=yes;
            }
            else
            {
                class=no;
            }

            if (key_field_value)
            {
                out = key_field_value;
            }

            recordDatum = PredictGetDatum(out, model->current,
                                          probability, class,
                                          functionContext->tuple_desc);

        }


        ((MLmodel) functionContext->user_fctx)->current++;

        pfree(model->row_fvalues);
        pfree(model->cat_value_buffer);
        pfree(model->row_cvalues);

        pfree(model->result_pa);
        pfree(model->result_exp);


        SRF_RETURN_NEXT(functionContext, recordDatum);
    }
    else
    {
        MLmodel model = (MLmodel)functionContext->user_fctx;
        pfree(model->keyField);

        if (SPI_finish() != SPI_OK_FINISH)
            elog(WARNING, "could not finish SPI");

        SRF_RETURN_DONE(functionContext);
    }
}

Datum
ml_predict_dataset_inner(PG_FUNCTION_ARGS)
{
    FuncCallContext *functionContext = NULL;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldContext;
        TupleDesc tupleDescriptor;
        ArrayDatum  cat_fields = {0, NULL};
        StringInfoData buf;
        ModelCalcerHandle* modelHandle;
        // SPITupleTable   *spi_tuptable;
        const char*     model_info;
        const int resultColumnCount = 3;
        size_t          model_float_feature_count;
        size_t          model_cat_feature_count ;
        size_t          model_dimension;
        size_t          featureCount = 0;
        char            **features;
        size_t          cat_feature_counter=0;
        size_t          feature_counter=0;
        int             i;
        MLmodel         model;
        text            *filename;
        char            *key_field;
        int             res;
        bool            function_type = PG_GETARG_BOOL(3);
        char            **arr_cat_fields = NULL;

        /* create a function context for cross-call persistence */
        functionContext = SRF_FIRSTCALL_INIT();

        filename = PG_GETARG_TEXT_PP(0);


        LoadModel(filename, &modelHandle);

        /* switch to memory context appropriate for multiple function calls */
        oldContext = MemoryContextSwitchTo(
            functionContext->multi_call_memory_ctx);

        initStringInfo(&buf);

        features = getModelFeatures(modelHandle, &featureCount);

        model_cat_feature_count = (size_t)GetCatFeaturesCount(modelHandle);

        if (model_cat_feature_count)
        {
            size_t cat_feature_count;
            size_t* indices;

            if (!GetCatFeatureIndices(modelHandle, &indices, &cat_feature_count))
            {
                elog(ERROR, "cat feature %s", GetErrorString());
            }

            arr_cat_fields = palloc( sizeof(char*) * cat_feature_count);

            for (i=0; i < cat_feature_count; i++)
            {
                arr_cat_fields[i] = pstrdup(features[indices[i]]);
            }

            free(indices); // allocated by CatBoostModel::GetCatFeatureIndices()
        }

        key_field = text_to_cstring(PG_GETARG_TEXT_PP(2));

        model = (MLmodel) palloc0(sizeof(MLmodelData));

        model->modelHandle = modelHandle;
        model_info = getModelParms(model->modelHandle);
        SPI_connect();
        model->modelType = getModelType(model->modelHandle, model_info);
        model->modelClasses = getModelClasses(model->modelHandle, model_info);

        if (key_field)
            model->keyField = pstrdup(key_field);

        if (function_type) {
            char  *query = text_to_cstring(PG_GETARG_TEXT_PP(1));
            res = SPI_exec(query, 0);
        }
        else
        {
            char  *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));
            appendStringInfo(&buf, "SELECT * FROM %s;", tabname);
            res = SPI_exec(buf.data, 0);
        }
        if (res < 1 || SPI_tuptable == NULL)
        {
            elog(ERROR, "Query %s error", buf.data);
        }

        model->cat_fields = cat_fields;

        model->spi_tuptable = SPI_tuptable;
        model->spi_tupdesc  = SPI_tuptable->tupdesc;
        model->attCount = SPI_tuptable->tupdesc->natts;


        model->iscategory = palloc0( model->attCount * sizeof(int8));
        model_float_feature_count = (size_t)GetFloatFeaturesCount(model->modelHandle);
        model_dimension = (size_t)GetDimensionsCount(model->modelHandle);

        for(i=0; i < model->attCount; i++)
        {
            if(! checkInArray(ModelGetFieldName(i), features, featureCount))
            {
                model->iscategory[i] = -1;  // not in features
                continue;
            }
            if ( checkInArray(ModelGetFieldName(i), arr_cat_fields, model_cat_feature_count))
            {
                cat_feature_counter ++;
                model->iscategory[i] = 1;
            }
            else
            {
                feature_counter ++;
                model->iscategory[i] = 0;
            }
        }

         //  check model features
        if (feature_counter != model_float_feature_count)
        {
            elog(ERROR,
                "count of numeric features is not valid, must be %ld is %ld",
                model_float_feature_count, feature_counter
            );
        }

        if (cat_feature_counter != model_cat_feature_count)
        {
            elog(ERROR,
                "count of categocical features is not valid, must be %ld is %ld",
                model_cat_feature_count, cat_feature_counter
            );
        }

        model->current = 0;
        functionContext->user_fctx = model;
        functionContext->max_calls = SPI_processed;
        model->dimension = model_dimension;
        model->cat_count = cat_feature_counter;
        model->num_count = feature_counter;

        /*
         * This tuple descriptor must match the output parameters declared for
         * the function in pg_proc.
         */
        tupleDescriptor = CreateTemplateTupleDesc(resultColumnCount);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 1, key_field,
                           TEXTOID, -1, 0);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 2, "predict",
                           FLOAT8OID, -1, 0);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 3, "class",
                           TEXTOID, -1, 0);

        functionContext->tuple_desc =  BlessTupleDesc(tupleDescriptor);

        MemoryContextSwitchTo(oldContext);
    }

    functionContext = SRF_PERCALL_SETUP();

    if (  ((MLmodel) functionContext->user_fctx)->current < functionContext->max_calls)
    {
        char*   class;
        int     feature_counter = 0;
        int     cat_feature_counter = 0;
        MLmodel model = (MLmodel)functionContext->user_fctx;
        Datum   recordDatum;
        int j;
        char *p;
        char* yes = "yes";
        char* no = "no";
        HeapTuple   spi_tuple = ((MLmodel)functionContext->user_fctx)->spi_tuptable->vals[((MLmodel)functionContext->user_fctx)->current];
        int memsize = model->cat_count * sizeof(char) * FIELDLEN;
        char* key_field_value = NULL;

        model->row_fvalues = palloc0( model->num_count * sizeof(float));
        p = model->cat_value_buffer  = palloc0(memsize );
        model->row_cvalues = palloc0(model->cat_count * sizeof(char*));

        model->result_pa  = (double*) palloc( sizeof(double) * model->dimension);
        model->result_exp = (double*) palloc( sizeof(double) * model->dimension);

        for(j=0; j < model->attCount; j++)
        {
            char    *value;
            value = SPI_getvalue(spi_tuple, model-> spi_tupdesc, j+1);

            if (strcmp(model->keyField,  ModelGetFieldName(j)) == 0)
            {
                key_field_value = value;
            }

            if( model->iscategory[j] == -1) // not in features
                continue;

            if( model->iscategory[j] == 0)
            {
                if (value == 0 )
                {
                    model->row_fvalues[feature_counter] = QNaN;
                }
                else
                {
                    int res;
                    res  = sscanf(value, "%f", &model->row_fvalues[feature_counter]);
                    if(res < 1)
                    {
                        elog(WARNING,"error input j/cnt=%d/%d %f\n", j,
                            feature_counter, model->row_fvalues[feature_counter]);
                    }
                }

                feature_counter++;
            }

            if( model->iscategory[j] == 1)
            {
                if (!value)
                {
                    model->row_cvalues[cat_feature_counter] = pstrdup("NaN");
                    p += 3;
                }
                else
                {
                    model->row_cvalues[cat_feature_counter] = strcpy(p, value);
                    p += strlen(value);
                }
                cat_feature_counter++;
                *p = '\0';
                p++;
            }
        } // column

        if (!CalcModelPredictionSingle(model->modelHandle,
                    model->row_fvalues, feature_counter,
                    (const char** )model->row_cvalues, cat_feature_counter,
                    model->result_pa, model->dimension)
            )
        {
            StringInfoData str;
            initStringInfo(&str);
            for(j = 0; j < feature_counter; j++)
            {
                appendStringInfo(&str, "%f,", model->row_fvalues[j]);
            }
            elog( ERROR, "CalcModelPrediction error message: %s \nrow num=%ld",
                    GetErrorString(),  model->current );
        }

        if (strncmp("\"MultiClass\"", model->modelType, 12) == 0)
        {
            char  ***p;
            double max = 0., sm = 0.;
            int max_i = -1;
            char* out = model->keyField;

            for( j = 0; j < model->dimension; j ++)
            {
                model->result_exp[j] = exp(model->result_pa[j]);
                sm += model->result_exp[j];
            }
            for( j = 0; j < model->dimension; j ++)
            {
                model->result_exp[j] = model->result_exp[j] / sm;
                if (model->result_exp[j] > max){
                    max = model->result_exp[j];
                    max_i = j;
                }
            }

            p = model->modelClasses + max_i;

            if (key_field_value)
            {
                out = key_field_value;
            }


            recordDatum = PredictGetDatum(out, model->current, max, (char*)*p,
                            functionContext->tuple_desc);

        }
        else if (strcmp(model->modelType, "\"RMSE\"") == 0)
        {
            char* out = "";

            if (key_field_value)
            {
                out = key_field_value;
            }
            recordDatum = PredictGetDatum(out, model->current, model->result_pa[0], NULL,
                            functionContext->tuple_desc);

        }
        else if (strncmp("\"Logloss\"", model->modelType, 9) == 0)
        {
            double probability = sigmoid(model->result_pa[0]);
            char* out = model->keyField;
            int n = 0;
            if (probability > 0.5)
            {
                n = 1;
            }

            if (key_field_value)
            {
                out = key_field_value;
            }
            recordDatum = PredictGetDatum(out, model->current, probability,
                            (char*)*(model->modelClasses + n),
                            functionContext->tuple_desc);
        }
        else
        {
            double probability = sigmoid(model->result_pa[0]);
            char* out = model->keyField;
            if (probability > 0.5)
            {
                class=yes;
            }
            else
            {
                class=no;
            }

            if (key_field_value)
            {
                out = key_field_value;
            }

            recordDatum = PredictGetDatum(out, model->current,
                                          probability, class,
                                          functionContext->tuple_desc);

        }


        ((MLmodel) functionContext->user_fctx)->current++;

        pfree(model->row_fvalues);
        pfree(model->cat_value_buffer);
        pfree(model->row_cvalues);

        pfree(model->result_pa);
        pfree(model->result_exp);


        SRF_RETURN_NEXT(functionContext, recordDatum);
    }
    else
    {
        MLmodel model = (MLmodel)functionContext->user_fctx;
        pfree(model->keyField);

        if (SPI_finish() != SPI_OK_FINISH)
            elog(WARNING, "could not finish SPI");

        SRF_RETURN_DONE(functionContext);
    }
}


/*
 * Check existing model folder
 *
 */
static bool
check_model_path(char **newval, void **extra, GucSource source)
{
    struct stat st;

    /*
     * The default value is an empty string, so we have to accept that value.
     * Our check_configured callback also checks for this and prevents
     * archiving from proceeding if it is still empty.
     */
    if (*newval == NULL || *newval[0] == '\0')
        return true;

    /*
     * Make sure the file paths won't be too long.  The docs indicate that the
     * file names to be archived can be up to 64 characters long.
     */
    if (strlen(*newval) + 64 + 2 >= MAXPGPATH)
    {
        GUC_check_errdetail("directory too long.");
        return false;
    }

    /*
     * Do a basic sanity check that the specified archive directory exists. It
     * could be removed at some point in the future, so we still need to be
     * prepared for it not to exist in the actual archiving logic.
     */
    if (stat(*newval, &st) != 0 || !S_ISDIR(st.st_mode))
    {
        GUC_check_errdetail("Specified  directory does not exist.");
        return false;
    }

    return true;
}


void
_PG_init(void)
{
    DefineCustomStringVariable("ml.model_path",
                               "Path to model folder",
                               NULL,
                               &model_path,
                               "",
                               PGC_USERSET,
                               0,
                               check_model_path, NULL, NULL);

    MarkGUCPrefixReserved("ml");
}

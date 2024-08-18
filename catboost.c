/*
 * contrib/debug/debug.c
 */
#include "postgres.h"
#include "fmgr.h"
#include "c.h"

#include "access/slru.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"

#include <stdbool.h>

#define TOKEN_LEN 32

#define PARSE_TOKEN   p2++;    \
    p3 = strip(&p2, TOKEN_LEN);\
    p2 = unstrip(&p3);         \
    *p2 = '\0';


#define CHECK_TOKEN(tok, len) \
if (is_three == false && pg_strncasecmp(key, tok, len) == 0) \
{   *(p3++) = '_';                                           \
    is_three = true;                                         \
    continue;}

enum model_type_t {
    MODEL_NONE,
    MODEL_CLASSIFICATION,
    MODEL_REGRESSION,
    MODEL_RANKING,
};

PG_MODULE_MAGIC;



/* Function declarations */


char* strip(char **buf, int len);
char* unstrip(char **buf);
char* comma(char **buf);
char* parse_name(char** buf, char **out);


PG_FUNCTION_INFO_V1(test_python);
PG_FUNCTION_INFO_V1(test_table);
PG_FUNCTION_INFO_V1(ml_parse);


char*
strip(char **buf, int len)
{
    char* p  = *buf;
    while( *p == ' ' && --len)
        p++;
    return p;
}

char*
unstrip(char **buf)
{
    char* p  = *buf;
    while( *p != ' ')
        p++;
    return p;
}

char*
comma(char **buf)
{
    char* p  = *buf;
    while( *p != ',')
        p++;
    return p;
}

 // "   CREATE MODEL adult (xxx 10, yyy aaa, cut_features 'age, education, occupation', metrics logloss) "
                            // "AS SELECT * FROM adult10; ";
char *
parse_name(char** buf, char **out)
{
    char  *p, *p2, *p3;
    enum model_type_t model_type;
    bool flag;
    char *modelName;
    model_type = MODEL_NONE;

    p = *buf;

    /* CREATE */    
    p3 = strip(&p, TOKEN_LEN);
    p2 = unstrip(&p3);
    *p2 = '\0';  
    
    if (pg_strcasecmp("create", p3) != 0)
    {
        elog(WARNING, "DO NOT PARSE");
        return NULL;
    }

    // next token
    PARSE_TOKEN;

    flag = false;
    if (pg_strcasecmp("model", p3) == 0  )
    {
      flag = true;
      model_type = MODEL_CLASSIFICATION;
    }

    if (pg_strcasecmp("classification", p3) == 0 )
    {
        model_type = MODEL_CLASSIFICATION;
    }

    if (pg_strcasecmp("regression", p3) == 0 )
    {
        model_type = MODEL_REGRESSION;
    }

    if (pg_strcasecmp("ranking", p3) == 0 )
    {
        model_type = MODEL_RANKING;
    }

    if (flag == false && model_type == MODEL_NONE)
    {
        elog(ERROR, "model type undefined");
        return NULL;
    }
    
    elog(WARNING, "PARSE Ok type=%d flag=%d", model_type, flag);

    // check token MODEL
    PARSE_TOKEN;
    if (pg_strcasecmp("model", p3) == 0 )
    {
        PARSE_TOKEN;
    }

    modelName = p3;
    *out = p2 + 1;

    return modelName;
}
 

Datum
ml_parse(PG_FUNCTION_ARGS)
{
    char *modelName, *p2, *p3, *p;
    bool is_begin = true;
    bool is_three = false;
    bool is_apostrophe = false;
    bool key_start = false;
    char *query = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char key[TOKEN_LEN];
    char value[TOKEN_LEN];
    char c;

    StringInfoData buf;
    initStringInfo(&buf);
    appendStringInfoChar(&buf, '{');

    modelName = parse_name(&query, &p2);

    if (modelName == NULL)
    {        
        elog(ERROR, "the name model is absent");
        PG_RETURN_INT64(-1);
    }
    p3 = strip(&p2, TOKEN_LEN);

    if (*p3 != '('){
        elog(ERROR, "parse error: '%s'", p2 );        
        PG_RETURN_INT64(-1);
    }

    p = p3;
    p2 = NULL;
    is_begin = true;

    while( *p++ != ')')
    {
        if (*p == ' ' && !key_start)
            continue;

        if (is_begin)
        {
            p2 = p;
            p3 = key;
            is_begin = false;
            is_three = false;
            key_start = true;
        }

        if (*p == ' ') 
        {

            CHECK_TOKEN("LEARNING", 8);
            CHECK_TOKEN("L2", 2);
            CHECK_TOKEN("LOSS", 4);
            CHECK_TOKEN("TREE", 4);
            CHECK_TOKEN("USE", 3);
            CHECK_TOKEN("RANDOM", 6);

            *p3 = '\0';
            key_start = false;
            p3 = value;
            continue;
        }


        if (*p == ',' && !is_apostrophe)
        {
            is_begin = true;
            *p = '\0';

            appendStringInfo(&buf, "\"%s\":",key);
            
            if (isdigit(value[0]))
                appendStringInfo(&buf, "%s",value);
            else
                appendStringInfo(&buf, "\"%s\"",value);
            appendStringInfoChar(&buf, ',');
        }

        if(*p == '\'')
        {   
            if (is_apostrophe)
            {
                is_apostrophe = false;            
                *p3++ = '}';
                *p3 = '\0';
                continue;
            }
            else 
            {                
                is_apostrophe = true;
                value[0] = '{';
                p3 = value + 1;
                continue;
            }
        }

        c = toupper(*p);
        *p3++ = c;
    }

    *--p3 = '\0';
    appendStringInfo(&buf, "\"%s\":",key);
    if (isdigit(value[0]))
        appendStringInfo(&buf, "%s",value);
    else
        appendStringInfo(&buf, "\"%s\"",value);
    appendStringInfoChar(&buf, '}');


    elog(NOTICE, "%s", buf.data);

    pfree(buf.data); // входные данные для catBoost


    /*  select  query */


    p3 = strip(&p, TOKEN_LEN);
    if (*p3 == 'A'  ||  *p3 == 'a' )
        p3++;
    else
        PG_RETURN_UINT64(0);

    if (*p3 == 'S'  ||  *p3 == 's' )
        p3++;
    else
        PG_RETURN_UINT64(0);


    p = strip(&p3, TOKEN_LEN);
    elog(WARNING, "**** '%s'", p);


    PG_RETURN_UINT64(777);
}

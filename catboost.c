/*
 * contrib/debug/debug.c
 */
#include "postgres.h"
#include "fmgr.h"
#include "c.h"

#include "access/slru.h"
#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/parsenodes.h"
#include "nodes/primnodes.h"
#include "nodes/pg_list.h"
#include "parser/analyze.h"
#include "tcop/utility.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"


#include "optimizer/planner.h"
#include <stdbool.h>

enum model_type_t {
    MODEL_NONE,
    MODEL_CLASSIFICATION,
    MODEL_REGRESSION,
    MODEL_RANKING,
};

PG_MODULE_MAGIC;


static planner_hook_type prev_planner_hook = NULL;

/* Function declarations */

// extern Datum ml_learn_classifier(PG_FUNCTION_ARGS);
PlannedStmt *my_planner(Query *parse, const char *query_string, 
                       int cursorOptions, ParamListInfo boundParams);
void ml_parse(const char *query_string, char **modelName);
void ModelProcessUtility(PlannedStmt *pstmt, const char *queryString,
                        bool readOnlyTree, ProcessUtilityContext context,
                        ParamListInfo params, QueryEnvironment *queryEnv,
                        DestReceiver *dest, QueryCompletion *qc);
void ModelExecutorStart(QueryDesc *queryDesc, int eflags);
void ModelExecutorRun(QueryDesc *queryDesc,
                     ScanDirection direction, uint64 count, bool execute_once);
void ModelExecutorEnd(QueryDesc *queryDesc);
void model_post_parse_analyze(ParseState *pstate,
                             Query *query,
                             JumbleState *jstate);

void _PG_init(void);




PlannedStmt *
my_planner(Query *parse, const char *query_string, int cursorOptions,
                 ParamListInfo boundParams)
{
    // elog(WARNING, "%s", __FUNCTION__);

    return standard_planner(parse, query_string, cursorOptions, boundParams);
}





void
ModelProcessUtility(PlannedStmt *pstmt,
                        const char *queryString,
                        bool readOnlyTree,
                        ProcessUtilityContext context,
                        ParamListInfo params,
                        QueryEnvironment *queryEnv,
                        DestReceiver *dest,
                        QueryCompletion *qc)
{

    Node *parsetree = pstmt->utilityStmt;
    NodeTag nodeTag = nodeTag(parsetree);

    // // elog(WARNING, "ModelProcessUtility");


    if (nodeTag == T_CreateModelStmt) 
    {
        CreateModelStmt *stm = (CreateModelStmt *)parsetree;
        ListCell *lc;
        StringInfoData buf;
        int ret;
        // uint64 proc;

        initStringInfo(&buf);
        appendStringInfo(&buf, "SELECT ml_learn_classifier('%s', '{", stm->modelname);

        // // elog(WARNING, "Model name %s  %s", stm->modelname, stm->modelclass ? "regression" : "classification");
        

        foreach(lc, stm->options)
        {        
            ModelOptElement *opt;
            opt = (ModelOptElement *) lfirst(lc);
            if (opt->value != NULL)
                // // elog(WARNING,"key:%s value:%s\n",opt->key,opt->value );
                appendStringInfo(&buf, "\"%s\":\"%s\"",opt->key,opt->value);
            else
                appendStringInfo(&buf, "\"%s\":%s",opt->key,opt->value);

            appendStringInfo(&buf, ", ");
        }
        
        buf.data[buf.len-2] = '}';
        appendStringInfo(&buf, "', '%s')", stm->tablename);

        SPI_connect();
        ret = SPI_exec(buf.data, 1);
        // proc = SPI_processed;

        // elog(WARNING, "get %ld records ret = %d" , proc, ret);
        if (ret > 0 && SPI_tuptable != NULL)
        {
            TupleDesc tupdesc = SPI_tuptable->tupdesc;
            // elog(WARNING, "attr count=%d", tupdesc->natts);
            SPITupleTable *tuptable = SPI_tuptable;
            HeapTuple tuple = tuptable->vals[0];
            char * res = SPI_getvalue(tuple, tupdesc, 1);
            elog(INFO, "Acc=%s", res);
        }

        SPI_finish();
        pfree(buf.data);


        CommandCounterIncrement();
    }
    else
        standard_ProcessUtility(pstmt, queryString, readOnlyTree,
                                context, params, queryEnv,
                                dest, qc);
}


void
ModelExecutorRun(QueryDesc *queryDesc,
                     ScanDirection direction, uint64 count, bool execute_once)
{
    // elog(WARNING, "%s", __FUNCTION__);
    standard_ExecutorRun(queryDesc, direction, count, execute_once);
}

void
ModelExecutorStart(QueryDesc *queryDesc, int eflags)
{
    // elog(WARNING, "%s", __FUNCTION__);
    standard_ExecutorStart(queryDesc, eflags);
}

void
ModelExecutorEnd(QueryDesc *queryDesc)
{
    // elog(WARNING, "%s", __FUNCTION__);
    standard_ExecutorEnd(queryDesc);
}

void
model_post_parse_analyze(ParseState *pstate,
                        Query *query,
                        JumbleState *jstate)
{
    // elog(WARNING, "%s", __FUNCTION__);    
};

void
_PG_init(void)
{
 
    /*
     * Perform checks before registering any hooks, to avoid erroring out in a
     * partial state.
     *
     * In many cases (e.g. planner and utility hook, to run inside
     * pg_stat_statements et. al.) we have to be loaded before other hooks
     * (thus as the innermost/last running hook) to be able to do our
     * duties. For simplicity insist that all hooks are previously unused.
     */
    if (planner_hook != NULL )
    {
        ereport(ERROR, (errmsg("CatBoost extension has to be loaded first"),
                        errhint("Place this extension at the beginning of "
                                "shared_preload_libraries.")));
    }

    /* intercept planner */
    prev_planner_hook = planner_hook;
    planner_hook = my_planner;
    
    // prev_ProcessUtility = ProcessUtility_hook;
    ProcessUtility_hook = ModelProcessUtility;

    ExecutorStart_hook = ModelExecutorStart;
    ExecutorRun_hook = ModelExecutorRun;
    // ExplainOneQuery_hook = CitusExplainOneQuery;
    // prev_ExecutorEnd = ExecutorEnd_hook;
    ExecutorEnd_hook = ModelExecutorEnd;


    // prev_post_parse_analyze_hook = post_parse_analyze_hook;
    post_parse_analyze_hook = model_post_parse_analyze;
}
/*
 * contrib/debug/debug.c
 */
#include <sys/stat.h>

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
static char* model_path = "";

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
static bool check_model_path(char **newval, void **extra, GucSource source);
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
    ListCell  *lc;
    StringInfoData  buf;
    int  ret;
    char * acc = "" ;
    Node *parsetree = pstmt->utilityStmt;
    CreateModelStmt  *stm = (CreateModelStmt *)parsetree;
    NodeTag nodeTag = nodeTag(parsetree);
    char model_class;
    char *model_full_path;
    int len = 0;
    char *p, *p2, *parms;
    // // elog(WARNING, "ModelProcessUtility");


    if (nodeTag != T_CreateModelStmt) 
    {
        
        standard_ProcessUtility(pstmt, queryString, readOnlyTree,
                                context, params, queryEnv,
                                dest, qc);
    return;
    }
    

    initStringInfo(&buf);
    appendStringInfo(&buf, "SELECT ml_learn_classifier('%s', '{", stm->modelname);

    len = buf.len;    

    foreach(lc, stm->options)
    {        
        ModelOptElement *opt;
        opt = (ModelOptElement *) lfirst(lc);
        if (opt->value != NULL)
            appendStringInfo(&buf, "\"%s\":\"%s\"",opt->key,opt->value);
        else
            appendStringInfo(&buf, "\"%s\":%s",opt->key,opt->value);

        appendStringInfo(&buf, ", ");
    }
    p2 = buf.data + len;
    len = buf.len - len;

    parms = palloc(len);
    p = parms;
    while (len --)
        *p++ = *p2++;
    *--p = '\0';


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
        acc = SPI_getvalue(tuple, tupdesc, 1);
        elog(INFO, "Acc=%s", acc);
    }

    switch (stm->modelclass) {
        case 0: model_class = 'C';
                break;
        case 1 :model_class = 'R';
                break;
        case 2 :model_class = 'N';
                break;
        default: elog(ERROR, "Undefined model name");    
    }

    resetStringInfo(&buf);
    if (model_path[0] == ' ')
        appendStringInfo(&buf, "%s.sql.cbm",stm->modelname);
    else
        appendStringInfo(&buf, "%s/%s.sql.cbm",model_path, stm->modelname);
    model_full_path = pstrdup(buf.data);
    resetStringInfo(&buf);

    appendStringInfo(&buf, "INSERT INTO ml_model(name, acc, model_type, model_file,args) "
                                        "VALUES ('%s', %s, '%c', '%s', '%s') "
                           "ON CONFLICT (name) DO UPDATE SET acc=%s, model_type='%c', model_file='%s', args='%s'", 
                            stm->modelname, acc, model_class, model_full_path, parms,
                            acc, model_class, model_full_path, parms);

    ret = SPI_exec(buf.data, 0);
    if (ret != SPI_OK_INSERT)
    {
        elog(WARNING, "meta data is not inserting");
    }

    SPI_finish();
    pfree(buf.data);
    pfree(parms);
    CommandCounterIncrement();
        
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
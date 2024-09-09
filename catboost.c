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
// #include "executor/spi.h"
// #include "nodes/parsenodes.h"
// #include "nodes/primnodes.h"
// #include "nodes/pg_list.h"
// #include "parser/analyze.h"
// #include "tcop/utility.h"
#include "utils/builtins.h"
// #include "utils/errcodes.h"
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
void _PG_init(void);

static char* model_path = "";



PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(ml_predict);


Datum
ml_predict(PG_FUNCTION_ARGS)
{
    char *str = "***";
    PG_RETURN_TEXT_P(cstring_to_text(str));
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

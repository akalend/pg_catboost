
# sudo  PATH=$PATH:/usr/local/pgsql/bin    make install

MODULE_big = catboost
# MODULES = catboost

OBJS = catboost.o
EXTENSION = catboost
DATA = catboost--0.1.sql

PGFILEDESC = "machine learning module using catboost"

PATH += $(shell pg_config  --bindir)

# PYTHON_LIB = $(shell python3-config  --ldflags)

# PG_LIBDIR = $(shell pg_config  --libdir)


# PG_CPPFLAGS = -I$(libpq_srcdir) -ggdb -I/usr/include/python3.10 $(python3-config --includes )
# PG_LDFLAGS +=   -L$(PG_LIBDIR) $(PYTHON_LIB)  $(python3-config --libs)

# SHLIB_LINK +=  -lpython3.10 -lpq

# SHLIB_LINK_INTERNAL += 

#  $(LIBS) 
#    $(python3-config --includes --ldflags --libs)  -lpython3.10

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/catboost
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif


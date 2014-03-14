


#include "stdio.h"
#include "mex.h"
#include "string.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const 
mxArray*prhs[] ) {

     char arg1[65];
     mxGetString(prhs[0], arg1, sizeof(arg1)-1);

     if (!strcmp(arg1, "MPI_Init")) {

       MPI_Init(0,0);

     }

     else if (!strcmp(arg1, "MPI_Finalize")) {

         MPI_Finalize();

     }
}
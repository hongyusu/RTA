


#define mint mwSize


struct type_element_list
{
   double id;
   double val;
   struct type_element_list * next;
};

void printm(double * M, mint nrow, mint ncol);
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] );
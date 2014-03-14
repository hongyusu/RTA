

#define PICK -100

typedef struct back_v2i
{
    double v;
    int nrow;
    int ncol;
} back_t_v2i;


//void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] );
double * backward_alg_omp(double * P_node, double * T_node, int K, double * E, int nlabel, double * node_degree, int max_node_degree);
int back_compare_structs (const void *a, const void *b);
void back_printm(double * M, int nrow, int ncol);


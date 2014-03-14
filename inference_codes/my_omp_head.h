
#define mint mwSize
#define PICK -100

typedef struct v2i
{
    double v;
    mint nrow;
    mint ncol;
} t_v2i;


//void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] );
double * backward_alg_omp(double * P_node, double * T_node, mint K, double * E, mint nlabel, double * node_degree, mint max_node_degree);

// int compare_structs (const void *a, const void *b);
// 
// 
// void printm(double * M, mint nrow, mint ncol);
// 
    
typedef struct v2is
{
    double v;
    mint *i;
} t_v2is;


struct type_heap_array
{
   double v;
   mint x;
   mint y;
   struct type_heap_array * next;
};

void printm(double * M, mint nrow, mint ncol);
int coo2ind(mwSize x, mwSize, mwSize len);
double * LinearMaxSum(mxArray * M, mint current_node_degree);
int compare_structs (const void *a, const void *b);
int compare_structs_is (const void *a, const void *b);
double * forward_alg_omp(double * gradient, mint K, double * E, mint l, double * node_degree, mint max_node_degree);
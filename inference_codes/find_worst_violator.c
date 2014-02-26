

#include "matrix.h"
#include "mex.h"
#include "find_worst_violator.h"
#include "stdio.h"




/* The gateway function 
 * Input:
 *      Y_kappa
 *      Y_kappa_val
 * Output:
 *      Ymax
 *      YmaxVal
 *      break_flag
 */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    #define IN_Y_kappa          prhs[0]
    #define IN_Y_kappa_val      prhs[1]
    #define OUT_Ymax            plhs[0]
    #define OUT_YmaxVal         plhs[1]
    #define OUT_break_flag      plhs[2]
    
    double * Y_kappa;
    double * Y_kappa_val;
    double * Ymax;
    double YmaxVal;
    double break_flag=0;
    
    mint Y_kappa_nrow;
    mint Y_kappa_ncol;
    mint Y_kappa_val_nrow;
    mint Y_kappa_val_ncol;
    mint nlabel;
    
    /* INPUT VARIABLES */
    Y_kappa = mxGetPr(IN_Y_kappa);
    Y_kappa_nrow = mxGetM(IN_Y_kappa);
    Y_kappa_ncol = mxGetN(IN_Y_kappa);
    Y_kappa_val = mxGetPr(IN_Y_kappa_val);
    Y_kappa_val_nrow = mxGetM(IN_Y_kappa_val);
    Y_kappa_val_ncol = mxGetN(IN_Y_kappa_val);
    nlabel=Y_kappa_ncol/Y_kappa_val_ncol;
    /* OUTPUT VARIABLES */
    OUT_Ymax = mxCreateDoubleMatrix(1, nlabel,mxREAL);
    OUT_YmaxVal = mxCreateDoubleScalar(YmaxVal);
    OUT_break_flag = mxCreateDoubleScalar(break_flag);
    
    double * Y_kappa_ind = (double *) malloc(Y_kappa_val_nrow*Y_kappa_val_ncol);
    /* ASSIGN DECIMAL TO EACH BINARY MULTILABEL */
    for(mint ii=0;ii<Y_kappa_nrow;ii++)
    {
        for(mint jj=0;jj<Y_kappa_val_ncol;jj++)
        {
            /*printf("%d %d %d\n",ii,jj,nlabel);*/
            Y_kappa_ind[ii+jj*Y_kappa_nrow] = 0;
            for(mint kk=0;kk<nlabel;kk++)
            {
                /* printf("%.2f %.2f\n",Y_kappa_ind[ii+jj*Y_kappa_nrow], Y_kappa[ii+(jj*nlabel+kk)*Y_kappa_nrow]); */
                Y_kappa_ind[ii+jj*Y_kappa_nrow] = Y_kappa_ind[ii+jj*Y_kappa_nrow]*2 + ((Y_kappa[ii+(jj*nlabel+kk)*Y_kappa_nrow]+1)/2==1 ? 1:0);
            }
        }
    }
    
    /* printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol); */
    
    /* LOOP THROUGHT KAPPA*/
    type_element_list element_list;
    element_list.score=-1;
    element_list.next_element = NULL;
    type_element_list * my_list;
    type_element_list * cur_pos;
    my_list=NULL;
    //my_list = (type_element_list *) malloc (sizeof(element_list));
    //my_list->score = -1;
    //my_list->next_element = NULL;

    
    for(mint ii=0;ii<Y_kappa_val_ncol;ii++)
    {
        double theta=0;
        /* GET CURRENT LINE THRESHOLD THETA */ 
        for(mint jj=0;jj<Y_kappa_val_nrow;jj++)
        {theta = theta + Y_kappa_val[jj+ii*Y_kappa_val_nrow];}
        /* UPATE SCORE */
        for(mint jj=0;jj<Y_kappa_val_nrow;jj++)
        {
            cur_pos = my_list;
            while(cur_pos!=NULL)
            {
                
            }
            continue;
        }

    }
    
    
    
    // destroy my_list;
    free(Y_kappa_ind);
    
   
}


void printm(double * M, mint nrow, mint ncol)
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
    for(mint i=0; i<nrow; i++)
    {
        for(mint j=0; j<ncol; j++)
        {
            printf("%.4f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
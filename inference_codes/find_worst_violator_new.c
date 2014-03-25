

#include "matrix.h"
#include "mex.h"
#include "find_worst_violator.h"
#include "stdio.h"
#include "time.h"


/* INPUT K*T_SIZE LABELS AND VALUES MATRIX
 * OUTPUT
 *  BEST LABEL Y*
 *  TRUE LABEL Yi
 *  SCORE OF Y* AND Yi
 * margin is defined as the different between score of Y* and Yi
 * output the different as value for Y_*
 *
 *  POSITION OF BEST LABEL
 * 
 *
 *
 *
 */

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    #define IN_Y_kappa          prhs[0] // MATRIX OF MULTILABELS
    #define IN_Y_kappa_val      prhs[1] // MATRIX OF MULTILABEL SCORES
    #define IN_Y                prhs[2] // CORRECT MULTILABEL COULD BE EMPTY
    #define IN_E                prhs[3] // EDGES OF TREES
    #define IN_gradient         prhs[4] // GREDIENTS OF TREES
    #define OUT_Ymax            plhs[0] // OUTPUT BEST MULTILABEL
    #define OUT_YmaxVal         plhs[1] // OUTPUT MARGIN
    #define OUT_break_flag      plhs[2] // HIGHEST POSITION OF MULTILABEL IN THE LIST
    #define OUT_Y_pos           plhs[3] // AVERAGE POSITION OF Yi
    
    double * Y_kappa;
    double * Y_kappa_val;
    double * Y;
    double * E;
    double * gradient;
    double * Ymax;
    double break_flag=0;
    double Y_pos=-1;
    double Y_ind;
    
    int Y_kappa_nrow;
    int Y_kappa_ncol;
    int Y_kappa_val_nrow;
    int Y_kappa_val_ncol;
    int nlabel;
    int Y_ncol;
    int Y_nrow;
    
    /* INPUT VARIABLES */
    Y_kappa         = mxGetPr(IN_Y_kappa);
    Y_kappa_nrow    = mxGetM(IN_Y_kappa);
    Y_kappa_ncol    = mxGetN(IN_Y_kappa);
    Y_kappa_val     = mxGetPr(IN_Y_kappa_val);
    Y_kappa_val_nrow    = mxGetM(IN_Y_kappa_val);
    Y_kappa_val_ncol    = mxGetN(IN_Y_kappa_val);
    Y           = mxGetPr(IN_Y);
    E           = mxGetPr(IN_E);
    gradient    = mxGetPr(IN_gradient);
    Y_ncol      = mxGetN(IN_Y);
    Y_nrow      = mxGetM(IN_Y);
    nlabel      = Y_kappa_ncol/Y_kappa_val_ncol;
    
    // OUTPUT STUFFS
    OUT_Ymax        = mxCreateDoubleMatrix(1,nlabel,mxREAL);
    OUT_YmaxVal     = mxCreateDoubleScalar(1);
    OUT_break_flag  = mxCreateDoubleScalar(1);
    OUT_Y_pos       = mxCreateDoubleScalar(1);
    Ymax            = mxGetPr(OUT_Ymax);

    // ASSIGN AN ID TO EACH UNIQUE LABEL IN THE LIST
    double * Y_kappa_ind;
    Y_kappa_ind = (double *) malloc (sizeof(double) * Y_kappa_val_nrow* Y_kappa_val_ncol);
    struct type_arr2id_list * arr2id_head;
    struct type_arr2id_list * arr2id_curpos;
    struct type_arr2id_list * arr2id_prevpos;
    arr2id_head = NULL;
    arr2id_curpos = NULL;
    int num_uelement = 1;
    for(int ii=0;ii<Y_kappa_nrow;ii++)
    {
        for(int jj=0;jj<Y_kappa_val_ncol;jj++)
        {
            double * tmp;
            tmp = (double *) malloc (sizeof(double ) * nlabel);
            for(int kk=0;kk<nlabel;kk++)
            {tmp[kk] = Y_kappa[ii+(jj*nlabel+kk)*Y_kappa_nrow];}
            // EMPTY LIST -> INITIALIZE THE LIST BY THE ELEMENT
            if(!arr2id_head)
            {
                Y_kappa_ind[ii+jj*Y_kappa_nrow] = num_uelement;
                arr2id_head = (struct type_arr2id_list * ) malloc (sizeof(struct type_arr2id_list));
                arr2id_head->arr = tmp;
                arr2id_head->id = num_uelement;
                arr2id_head->next=NULL;
                num_uelement++;
                continue;
            }
            // NOT EMPTY GO THROUGH
            arr2id_curpos = arr2id_head;
            int find=0;
            while(arr2id_curpos)
            {
                int not_equ = 0;
                for(int kk=0;kk<nlabel;kk++)
                {
                    if(tmp[kk]!=arr2id_curpos->arr[kk])
                    {
                        not_equ = 1;
                        break;
                    }
                }
                if(!not_equ)
                {
                    Y_kappa_ind[ii+jj*Y_kappa_nrow] = arr2id_curpos->id;
                    find=1;
                    break;
                }
                arr2id_prevpos = arr2id_curpos;
                arr2id_curpos = arr2id_curpos->next;
            }
            // ELEMENE NOT FOUND, ADD IT TO CURRENT LIST
            if(!find)
            {
                arr2id_curpos           = arr2id_prevpos;
                Y_kappa_ind[ii+jj*Y_kappa_nrow] = num_uelement;
                arr2id_curpos->next     = (struct type_arr2id_list * ) malloc (sizeof(struct type_arr2id_list));
                arr2id_curpos           = arr2id_curpos->next;
                arr2id_curpos->arr      = tmp;
                arr2id_curpos->id       = num_uelement;
                arr2id_curpos->next     = NULL;
                num_uelement ++;
            }
        }
    }
    // ASSIGN ID TO TRUE LABEL IN THE LIST
    Y_ind = 0.0;
    if(Y_ncol>0)
    {
        arr2id_curpos = arr2id_head;
        while(arr2id_curpos)
        {
            int not_equ = 0;
            for(int kk=0;kk<nlabel;kk++)
            {
                if(Y[kk]!=arr2id_curpos->arr[kk])
                {
                    not_equ = 1;
                    break;
                }
            }
            if(!not_equ)
            {
                Y_ind = arr2id_curpos->id;
                break;
            }
            arr2id_curpos = arr2id_curpos->next;
        }
    }
    // get F_Y
    double F_Y=0; 
    double * Ytmp;
    if(Y_ncol>0)
    {
    
    Ytmp = (double *) malloc (sizeof(double) * nlabel);
    for(int kk=0;kk<nlabel;kk++)
    {Ytmp[kk] = Y_kappa[kk];}
       
    for(int tt=0; tt<Y_kappa_val_nrow; tt++)
    {
        double * EEtmp;
        double * ggradienttmp;
        EEtmp = (double *) malloc (sizeof(double) * (nlabel-1)*2);
        ggradienttmp = (double *) malloc (sizeof(double) * (nlabel-1) * 4);
        // E
        for(int ll=0;ll<(nlabel-1)*2;ll++)
        {EEtmp[ll] = E[tt*2*(nlabel-1)+ll];}
        // GRADIENT
        for(int ll=0;ll<(nlabel-1)*4;ll++)
        {ggradienttmp[ll] = gradient[tt*4*(nlabel-1)+ll];}
        // UPDATE F
        F_Y += Y2Fy(Ytmp, EEtmp, ggradienttmp, nlabel); 
        free(ggradienttmp);
        free(EEtmp);
    }
    free(Ytmp);
    }
    // get average position
    double Y_pos_avg = 0;
    for(int ii=0;ii<Y_kappa_val_nrow;ii++)
    {
        int jj;
        for(jj=0;jj<Y_kappa_val_ncol;jj++)
        {
            if(Y_kappa_ind[ii+jj*Y_kappa_val_nrow] == Y_ind)
            {break;}
        }
        Y_pos_avg += jj;
        if(jj==Y_kappa_val_nrow)
        {Y_pos_avg -= 1;}
    }
    Y_pos_avg = Y_pos_avg / Y_kappa_val_nrow;

//     printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printm(Y_kappa_val,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printm(Y_kappa,Y_kappa_nrow,20);
//     printf("finish\n");
    //  DESTROY TEMPORATORY LIST
    while(arr2id_head)
    {
        arr2id_curpos   = arr2id_head;
        arr2id_head     = arr2id_head->next;
        free(arr2id_curpos->arr);
        free(arr2id_curpos);
    }
    // DEFINE THE THRESHOLD
    double theta_K=0;
    for(int ii=0;ii<Y_kappa_val_nrow;ii++)
    {theta_K += Y_kappa_val[ii+(Y_kappa_val_ncol-1)*Y_kappa_val_nrow];}
    theta_K -= nlabel*Y_kappa_val_nrow;
    // DEFINE THE MAXIMUM DEPTH
    int theta_ncol=Y_kappa_val_ncol-1;
    
    
    
    //  LOOP THROUGH MULTILABELS
    //printm(Y_kappa_val,Y_kappa_val_nrow,Y_kappa_val_ncol);
    double cur_F;
    double best_F;
    best_F  = -10000000000;
    cur_F   = -10000000000;
    int cur_row;
    int cur_col;
    cur_row =-1;
    cur_col =-1;
    int find;
    find = 0;
    for(int ii=0; ii<Y_kappa_val_ncol;ii++)
    {
        for(int jj=0; jj<Y_kappa_val_nrow; jj++)
        {
            // if current label is the true label -> skip
            if(Y_kappa_ind[jj+ii*Y_kappa_val_nrow]==Y_ind)
            {
                //printf("skip %d %d \n",jj,ii);
                if(Y_pos==-1 && F_Y>=theta_K)
                {Y_pos = Y_pos_avg;}
                continue;
            }
            // otherwise, Ytmp
            //double * Ytmp;
            Ytmp = (double *) malloc (sizeof(double) * nlabel);
            for(int kk=0;kk<nlabel;kk++)
            {
                Ytmp[kk] = Y_kappa[jj+(ii*nlabel+kk)*Y_kappa_val_nrow];
            }
            // EVALUATE OVER TREE
            cur_F=0;
            for(int tt=0; tt<Y_kappa_val_nrow; tt++)
            {
                double * Etmp;
                double * gradienttmp;
            Etmp = (double *) malloc (sizeof(double) * (nlabel-1)*2);
            gradienttmp = (double *) malloc (sizeof(double) * (nlabel-1) * 4);
                // E;
                for(int ll=0;ll<(nlabel-1)*2;ll++)
                {Etmp[ll] = E[tt*2*(nlabel-1)+ll];}
                // GRADIENT
                for(int ll=0;ll<(nlabel-1)*4;ll++)
                {gradienttmp[ll] = gradient[tt*4*(nlabel-1)+ll];}
                // UPDATE F
                cur_F += Y2Fy(Ytmp, Etmp, gradienttmp, nlabel); 
                // FREE POINTER SPACE    
                
            free(gradienttmp);
            free(Etmp);
            }
            // FREE POINTER SPACE
            free(Ytmp);
            //printf("\t%d %d -> %.4f threshold col:%d val:%.4f  |  True:%.4f\n",jj,ii,cur_F,theta_ncol,theta_K,F_Y);
            // if current score is better than the best we have so far update current we have
            if(best_F<cur_F)    
            {
                cur_col=ii;
                cur_row=jj;
                best_F = cur_F;
                // if find one then shorten the list
                if(best_F > theta_K)
                {
                    find=1;
                    // move threshold upwards
                    double up_K;
                    up_K=0;
                    for(int ii=0;ii<Y_kappa_val_nrow;ii++)
                    {up_K += Y_kappa_val[ii+(theta_ncol-1)*Y_kappa_val_nrow];}
                    up_K -= nlabel*Y_kappa_val_nrow;
                    while(theta_ncol>=0 && best_F>up_K)
                    {
                        //printf("\t\t===== move up from %d:%.4f->%.4f\n",theta_ncol,theta_K,up_K);
                        theta_ncol--;
                        theta_K = up_K;
                        
                        up_K=0;
                        for(int ii=0;ii<Y_kappa_val_nrow;ii++)
                        {up_K += Y_kappa_val[ii+(theta_ncol-1)*Y_kappa_val_nrow];}
                        up_K -= nlabel*Y_kappa_val_nrow;
                        //printf("%.4f\n",up_K);
                    }
                }
            }
        }
        if(ii>=theta_ncol)
        {break;}
    }
    
    
    //printf("%d %d -> %.2f find %d best: %.2f threshold: %.2f\n",cur_row,cur_col,cur_F,find,best_F,theta_K);
    // STORE MULTILABEL THAT ACHIEVE THE BEST SCORE F
    for(int ii=0;ii<nlabel;ii++)
    {
        Ymax[ii] = Y_kappa[cur_row+(cur_col*(int)nlabel+ii)*Y_kappa_nrow];
    }
    // FREE POINTER SPACE
    free(Y_kappa_ind);
    /* COLLECT RESULTS */
    *(mxGetPr(OUT_YmaxVal)) = F_Y-best_F;
    *(mxGetPr(OUT_break_flag)) = 0;
    if(find==1)
    {*(mxGetPr(OUT_break_flag)) = cur_col+1;}
    *(mxGetPr(OUT_Y_pos)) = Y_pos+1;
    //printf("%d %.2f %d : %.3f %.3f %.3f\n",find, Y_pos+1,cur_col+1,F_Y,best_F,F_Y-best_F);
    
}

// GIVEN MULTILABEL AND GRADIENT, COMPUTE THE FUNCTUON VALUE
double Y2Fy(double *Y, double * E, double * gradient, double nlabel)
{
    double * mu;
    double Fy;
    Fy=0.0;
    // GET MU OUT FROM Y
    mu = (double *) calloc (sizeof(double), ((int)nlabel-1)*4);
    for(int i=0;i<(int)nlabel-1;i++)
    {mu[ 4*i+(int)Y[(int)E[i]-1]*2+(int)Y[(int)E[i+(int)(nlabel-1)]-1] ] = 1.0;}
    // COMPUTE MU*GRADIENT
    for(int i=0;i<(nlabel-1)*4;i++)
    {Fy += mu[i]*gradient[i];}
    // RETURN RESULTS
    free(mu);
    return(Fy);
}

// GIVEM POINTER AND DIMENTION, PRINT OUT MATRIX
void printm(double * M, int nrow, int ncol)
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
    for(int i=0; i<nrow; i++)
    {
        for(int j=0; j<ncol; j++)
        {
            printf("%.3f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
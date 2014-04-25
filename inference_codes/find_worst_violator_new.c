
/* 
 * MATLAB C GATEWAY FUNCTON:   
 *      find_worst_violator_new.c
 *
 * Compile wth:
 *      mex find_worst_violator_new.c
 *
 * Ver 0.0
 *
 * March 2014
 *
 * Input:
 *      1. Y_kappa:
 *          matrix of K*|Y||T| dimension, containing K best multilabel from all tree
 *      2. Y_kappa_val:
 *          matrix of K*|T| dimension, contain score of the K best multlabel from all tree
 *      3. Y:
 *          in training, correct multilabel for the example
 *          in testing, empty
 *      4. E:
 *          edge list of the all trees pooled together
 *      5. gradient
 *          gradient of all trees pooled together
 *
 * Output:
 *      1. Ymax:
 *          best multilabel from the k best list
 *      2. YmaxVal:
 *          score of the best multilabel
 *      3. break_flag:
 *          if the best multilabel is found evidentally, meaning the score of the multilabel is higher than the threshold
 *      4. Yi_pos:
 *          median position of best multilabel
 *
 * Note:
 *      1. No memeory lead, last check on 26/03/2014
 *      2. Median function is implemented by qsort, could be improved with a O(n) algorithm
 *      3. Searching K best list for Y* is improved with algorithm that increases threshold
 *      4. Add annotation on 25/04/2014
 *
 */

#include "matrix.h"
#include "mex.h"
#include "find_worst_violator.h"
#include "stdio.h"
#include "time.h"

// MATLAB GATEWAY FUNCTION
void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    #define IN_Y_kappa          prhs[0] // MATRIX OF MULTILABELS
    #define IN_Y_kappa_val      prhs[1] // MATRIX OF MULTILABEL SCORES
    #define IN_Y                prhs[2] // CORRECT MULTILABEL COULD BE EMPTY
    #define IN_E                prhs[3] // EDGES OF TREES
    #define IN_gradient         prhs[4] // GREDIENTS OF TREES
    #define OUT_Ymax            plhs[0] // OUTPUT BEST MULTILABEL
    #define OUT_YmaxVal         plhs[1] // OUTPUT MARGIN
    #define OUT_break_flag      plhs[2] // HIGHEST POSITION OF MULTILABEL IN THE LIST
    #define OUT_Yi_pos           plhs[3] // AVERAGE POSITION OF Yi
    
    double * Y_kappa;
    double * Y_kappa_val;
    double * Y;
    double * E;
    double * gradient;
    double * Ymax;
    double break_flag=0;
    double Yi_pos=-1;
    double Yi_ind;
    
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
    OUT_Yi_pos       = mxCreateDoubleScalar(1);
    Ymax            = mxGetPr(OUT_Ymax);

    // assign id to each unique label in the list from 1 to the number of element
    // assign id to Yi, initial Yi to 0
    // also get Yi position in all row
    Yi_ind = 0.0;
    double * Y_kappa_ind;
    Y_kappa_ind = (double *) malloc (sizeof(double) * Y_kappa_val_nrow* Y_kappa_val_ncol);
    struct type_arr2id_list * arr2id_head;
    struct type_arr2id_list * arr2id_curpos;
    struct type_arr2id_list * arr2id_prevpos;
    arr2id_head = NULL;
    arr2id_curpos = NULL;
    int num_uelement = 1;
    double * Yi_positions;
    Yi_positions = (double *) malloc (sizeof(double) * Y_kappa_val_nrow);
    for(int ii=0;ii<Y_kappa_val_nrow;ii++)
    {Yi_positions[ii] = Y_kappa_val_ncol+1;}
    for(int ii=0;ii<Y_kappa_nrow;ii++)
    {
        for(int jj=0;jj<Y_kappa_val_ncol;jj++)
        {
            // current Y in the top k list
            double * tmp;
            tmp = (double *) malloc (sizeof(double ) * nlabel);
            for(int kk=0;kk<nlabel;kk++)
            {tmp[kk] = Y_kappa[ii+(jj*nlabel+kk)*Y_kappa_nrow];}
            // if Y=Yi, keep the position
            int Yi_find = 1;
            if(Y_ncol)
            {
                for(int kk=0;kk<nlabel;kk++)
                {
                    if(Y[kk]!=tmp[kk])
                    {
                        Yi_find=0;
                        break;
                    }
                }
            }
            else
            {
                Yi_find=0;
            }
            if(Yi_find)
            {
                Yi_positions[ii]=jj+1;
            }
            // EMPTY LIST -> INITIALIZE THE LIST BY THE ELEMENT
            if(!arr2id_head)
            {
                Y_kappa_ind[ii+jj*Y_kappa_nrow] = num_uelement;
                arr2id_head = (struct type_arr2id_list * ) malloc (sizeof(struct type_arr2id_list));
                arr2id_head->arr = tmp;
                arr2id_head->id = num_uelement;
                arr2id_head->next=NULL;
                if(Yi_find==1)
                {Yi_ind = arr2id_head->id;}
                num_uelement++;
                continue;
            }
            // NOT EMPTY GO THROUGH
            arr2id_curpos = arr2id_head;
            int find=0;
            while(arr2id_curpos)
            {
                int Ytmp_find = 1;
                for(int kk=0;kk<nlabel;kk++)
                {
                    if(tmp[kk]!=arr2id_curpos->arr[kk])
                    {
                        Ytmp_find = 0;
                        break;
                    }
                }
                if(Ytmp_find==1)
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
            if(Yi_find==1)
            {Yi_ind = arr2id_curpos->id;}
        }
    }
    qsort(Yi_positions, Y_kappa_val_nrow, sizeof(double), sortcompare);
    Yi_pos = Yi_positions[Y_kappa_val_nrow/2]; 
    //printm(Yi_positions,1,Y_kappa_val_nrow);
    //printf("%.2f\n",Yi_pos);
    free(Yi_positions);
    // ASSIGN ID TO TRUE LABEL IN THE LIST
//     Yi_ind = 0.0;
//     if(Y_ncol>0)
//     {
//         arr2id_curpos = arr2id_head;
//         while(arr2id_curpos)
//         {
//             int not_equ = 0;
//             for(int kk=0;kk<nlabel;kk++)
//             {
//                 if(Y[kk]!=arr2id_curpos->arr[kk])
//                 {
//                     not_equ = 1;
//                     break;
//                 }
//             }
//             if(!not_equ)
//             {
//                 Yi_ind = arr2id_curpos->id;
//                 Yi_pos = 1;
//                 break;
//             }
//             arr2id_curpos = arr2id_curpos->next;
//         }
//     }
    // get F_Y
    double F_Y=0; 
    double * Ytmp;
    if(Y_ncol>0)
    {
        Ytmp = (double *) malloc (sizeof(double) * nlabel);
        for(int kk=0;kk<nlabel;kk++)
        {Ytmp[kk] = Y[kk];}
        //printm(Ytmp,1,10);
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
            //printm(gradient,4,9);
            //printm(EEtmp,9,2);
            F_Y += Y2Fy(Ytmp, EEtmp, ggradienttmp, nlabel); 
            free(ggradienttmp);
            free(EEtmp);
        }
        free(Ytmp);
        //printf("--%.4f\n",F_Y);
    }
    
    //if(Y_ncol>0){printf("----------------------------%.4f\n",F_Y);}
//     // get average position
//     double Yi_pos_avg = 0;
//     for(int ii=0;ii<Y_kappa_val_nrow;ii++)
//     {
//         int jj;
//         for(jj=0;jj<Y_kappa_val_ncol;jj++)
//         {
//             //printf("%.1f %.1f\n",Y_kappa_ind[ii+jj*Y_kappa_val_nrow],Yi_ind);
//             if(Y_kappa_ind[ii+jj*Y_kappa_val_nrow] == Yi_ind)
//             {break;}
//         }
//         Yi_pos_avg += jj;
//         if(jj==Y_kappa_val_nrow)
//         {Yi_pos_avg -= 1;}
//     }
//     Yi_pos_avg = Yi_pos_avg / Y_kappa_val_nrow;

    

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
    //theta_K -= nlabel*Y_kappa_val_nrow;
    // DEFINE THE MAXIMUM DEPTH
    int theta_ncol=Y_kappa_val_ncol-1;
    
//     printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printm(Y_kappa_val,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     //printm(Y_kappa,Y_kappa_nrow,8);
//     printf("Y ind:%.2f\n",Yi_ind);
//     printf("Position of Yi:%.2f\n",Yi_pos_avg);
//     printf("F of Yi:%.2f\n",F_Y);
//     printf("Threshold:%.2f\n\n",theta_K);
    
    
    
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
    //printf("\n");
    for(int ii=0; ii<Y_kappa_val_ncol;ii++)
    {
        for(int jj=0; jj<Y_kappa_val_nrow; jj++)
        {
            // if current label is the true label -> skip
            if(Y_kappa_ind[jj+ii*Y_kappa_val_nrow]==Yi_ind)
            {
                //if(Y_ncol>0){printf("\t%d %d -> skip\n",jj,ii);}
                //printf("-->%.2f %.2f : %.15f %.15f %.15f\n",Y_kappa_ind[jj+ii*Y_kappa_val_nrow],Yi_ind,F_Y,theta_K,F_Y-theta_K);
                //if(Yi_pos==-1 && F_Y-theta_K>=-0.0001)
                //{Yi_pos = Yi_pos_avg;}
                //printf("%.2f\n",Yi_pos);
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
            //if(Y_ncol>0){printf("\t%d %d -> %.4f threshold col:%d val:%.4f  |  True:%.4f\n",jj,ii,cur_F,theta_ncol,theta_K,F_Y);}
            // if current score is better than the best we have so far update current we have
            if(best_F<cur_F)    
            {
                cur_col=ii;
                cur_row=jj;
                best_F = cur_F;
                // if find one then shorten the list
                if(best_F-theta_K >= -1e-8)
                {
                    find=1;
                    // move threshold upwards
                    double up_K;
                    up_K=0;
                    for(int ii=0;ii<Y_kappa_val_nrow;ii++)
                    {up_K += Y_kappa_val[ii+(theta_ncol-1)*Y_kappa_val_nrow];}
                    //up_K -= nlabel*Y_kappa_val_nrow;
                    while(theta_ncol>0 && best_F-up_K>0)
                    {
                        //if(Y_ncol>0){printf("\t\t===== move up from %d:%.4f->%.4f\n",theta_ncol,theta_K,up_K);}
                        theta_ncol--;
                        theta_K = up_K;
                        
                        up_K=0;
                        for(int ii=0;ii<Y_kappa_val_nrow;ii++)
                        {up_K += Y_kappa_val[ii+(theta_ncol-1)*Y_kappa_val_nrow];}
                        //up_K -= nlabel*Y_kappa_val_nrow;
                        //printf("%.4f\n",up_K);
                    }
                }
            }
        }
        if(ii>=theta_ncol)
        {break;}
    }

    //if(Y_ncol>0){printf("%d %d -> find %d best: %.2f threshold: %.2f Fy %.2f\n",cur_row,cur_col,find,best_F,theta_K,F_Y);}
    // STORE MULTILABEL THAT ACHIEVE THE BEST SCORE F
    for(int ii=0;ii<nlabel;ii++)
    {
        Ymax[ii] = Y_kappa[cur_row+(cur_col*(int)nlabel+ii)*Y_kappa_nrow];
    }

    /* COLLECT RESULTS */
    *(mxGetPr(OUT_YmaxVal)) = F_Y-best_F;
//     *(mxGetPr(OUT_YmaxVal)) = best_F;
//     *(mxGetPr(OUT_break_flag)) = 0;
//     if(find==1)
//     {*(mxGetPr(OUT_break_flag)) = cur_col+1;}
    *(mxGetPr(OUT_Yi_pos)) = Yi_pos;
    //printf("%.3f \n",Yi_pos+1);
    //printf("\n%d %.2f %d : %.3f %.3f %.3f\n",find, Yi_pos+1,cur_col+1,F_Y,best_F,F_Y-best_F);
    // Y_position
    
    double * Y_positions;
    Y_positions = (double *) malloc (sizeof(double) * Y_kappa_val_nrow);
    for(int ii=0;ii<Y_kappa_val_nrow;ii++)
    {Y_positions[ii] = Y_kappa_val_ncol+1;}
    for(int ii=0; ii<Y_kappa_val_nrow;ii++)
    {
        for(int jj=0; jj<Y_kappa_val_ncol; jj++)
        {
            if(Y_kappa_ind[ii+jj*Y_kappa_val_nrow] == Y_kappa_ind[cur_row+cur_col*Y_kappa_val_nrow])
            {Y_positions[ii] = jj+1;}
        }
    }
    qsort(Y_positions, Y_kappa_val_nrow, sizeof(double), sortcompare);
    *(mxGetPr(OUT_break_flag)) = Y_positions[Y_kappa_val_nrow/2]; 
    free(Y_positions);
    free(Y_kappa_ind);
    
    
    
}

// GIVEN MULTILABEL AND GRADIENT, COMPUTE THE FUNCTUON VALUE
double Y2Fy ( double *Y, double * E, double * gradient, double nlabel )
{
    double * mu;
    double Fy;
    Fy=0.0;
    // GET MU OUT FROM Y
    mu = (double *) calloc (sizeof(double), ((int)nlabel-1)*4);
    for(int i=0;i<(int)nlabel-1;i++)
    {mu[ 4*i+(int)Y[(int)E[i]-1]*2+(int)Y[(int)E[i+(int)(nlabel-1)]-1] ] = 1.0;}
    // COMPUTE MU*GRADIENT
    //printf("Y2Fy\n");
    //printm(gradient,4,9);
    //printm(Y,1,10);
    //printm(mu,4,9);
    for(int i=0;i<(nlabel-1)*4;i++)
    {
        Fy += mu[i]*gradient[i];
    }
    //printf("%.2f \n",Fy);
    // RETURN RESULTS
    free(mu);
    //printf("Fy: %.2f\n",Fy);
    return(Fy);
}

// GIVEM POINTER AND DIMENTION, PRINT OUT MATRIX
void printm ( double * M, int nrow, int ncol )
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
    for(int i=0; i<nrow; i++)
    {
        for(int j=0; j<ncol; j++)
        {
            printf("%.5f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
int sortcompare (const void * a, const void * b)
{
  return ( *(double*)a - *(double*)b );
}
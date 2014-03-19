

#include "matrix.h"
#include "mex.h"
#include "find_worst_violator.h"
#include "stdio.h"
#include "time.h"



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
    //printf("--> in to worst\n");
    #define IN_Y_kappa          prhs[0]
    #define IN_Y_kappa_val      prhs[1]
    #define IN_Y                prhs[2]
    #define OUT_Ymax            plhs[0]
    #define OUT_YmaxVal         plhs[1]
    #define OUT_break_flag      plhs[2]
    
    double * Y_kappa;
    double * Y_kappa_val;
    double * Y;
    double * Ymax;
    double break_flag=0;
    double Y_ind;
    
    mint Y_kappa_nrow;
    mint Y_kappa_ncol;
    mint Y_kappa_val_nrow;
    mint Y_kappa_val_ncol;
    mint nlabel;
    mint Y_ncol;
    mint Y_nrow;
    
    /* INPUT VARIABLES */
    Y_kappa = mxGetPr(IN_Y_kappa);
    Y_kappa_nrow = mxGetM(IN_Y_kappa);
    Y_kappa_ncol = mxGetN(IN_Y_kappa);
    Y_kappa_val = mxGetPr(IN_Y_kappa_val);
    Y_kappa_val_nrow = mxGetM(IN_Y_kappa_val);
    Y_kappa_val_ncol = mxGetN(IN_Y_kappa_val);
    Y = mxGetPr(IN_Y);
    Y_ncol = mxGetN(IN_Y);
    Y_nrow = mxGetM(IN_Y);
    
    nlabel=Y_kappa_ncol/Y_kappa_val_ncol;
    /* OUTPUT VARIABLES */
    OUT_Ymax = mxCreateDoubleMatrix(1,nlabel,mxREAL);
    OUT_YmaxVal = mxCreateDoubleScalar(1);
    OUT_break_flag = mxCreateDoubleScalar(1);
    Ymax = mxGetPr(OUT_Ymax);

    double * Y_kappa_ind;
    Y_kappa_ind = (double *) malloc (sizeof(double) * Y_kappa_val_nrow* Y_kappa_val_ncol);
//     mxArray * mat_Y_kappa_ind;
//     double *Y_kappa_ind;
//     mat_Y_kappa_ind = mxCreateDoubleMatrix(Y_kappa_val_nrow, Y_kappa_val_ncol,mxREAL);
//     Y_kappa_ind = mxGetPr(mat_Y_kappa_ind);
    
    
    
    /* ASSIGN DECIMAL TO EACH BINARY MULTILABEL */
        

//     for(mint ii=0;ii<Y_kappa_nrow;ii++)
//     {
//         for(mint jj=0;jj<Y_kappa_val_ncol;jj++)
//         {
//             /* printf("%d %d %d\n",ii,jj,nlabel); */
//             Y_kappa_ind[ii+jj*Y_kappa_nrow] = 0;
//             for(mint kk=0;kk<nlabel;kk++)
//             {
//                 double tmp=0;
//                 if((Y_kappa[ii+(jj*nlabel+kk)*Y_kappa_nrow]+1)/2==1)
//                 {tmp=1;}
//                 Y_kappa_ind[ii+jj*Y_kappa_nrow] = Y_kappa_ind[ii+jj*Y_kappa_nrow]*2 + tmp;
//             }
//         }
//     }
//     Y_ind = -1;
//     if(Y_nrow>0)
//     {
//         Y_ind ++;
//         for(mint kk=0;kk<nlabel;kk++)
//         {
//             double tmp = 0;
//             if((Y[kk]+1)/2==1)
//             {tmp=1;}
//             Y_ind = Y_ind*2+tmp;
//             //Y_ind = Y_ind*2 + ((Y[kk]+1)/2==1 ? 1:0);
//         }
//     }
//     printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printf("%.2f\n",Y_ind);

    struct type_arr2id_list * arr2id_head;
    struct type_arr2id_list * arr2id_curpos;
    struct type_arr2id_list * arr2id_prevpos;
    arr2id_head = NULL;
    arr2id_curpos = NULL;
    int num_uelement = 1;
    for(mint ii=0;ii<Y_kappa_nrow;ii++)
    {
        for(mint jj=0;jj<Y_kappa_val_ncol;jj++)
        {
            
            double * tmp;
            tmp = (double *) malloc (sizeof(double ) * nlabel);
            for(int kk=0;kk<nlabel;kk++)
            {tmp[kk] = Y_kappa[ii+(jj*nlabel+kk)*Y_kappa_nrow];}
            
            if(!arr2id_head)
            {
                Y_kappa_ind[ii+jj*Y_kappa_nrow] = num_uelement;
                arr2id_head = (struct type_arr2id_list * ) malloc (sizeof(struct type_arr2id_list));
                arr2id_head->arr = tmp;
                arr2id_head->id = num_uelement;
                //printf("\tadd %d", num_uelement);
                arr2id_head->next=NULL;
                num_uelement++;
                continue;
            }
            
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
            // if the element is not found, add it to the list
            arr2id_curpos = arr2id_prevpos;
            if(!find)
            {
                Y_kappa_ind[ii+jj*Y_kappa_nrow] = num_uelement;
                arr2id_curpos->next = (struct type_arr2id_list * ) malloc (sizeof(struct type_arr2id_list));
                arr2id_curpos = arr2id_curpos->next;
                arr2id_curpos->arr = tmp;
                arr2id_curpos->id = num_uelement;
                //printf("\tadd %d", num_uelement);
                arr2id_curpos->next = NULL;
                num_uelement++;
            }
        }
    }
    
//     arr2id_curpos = arr2id_head;
//     while(arr2id_curpos)
//     {
//        printf("%.2f--",arr2id_curpos->id);
//        arr2id_curpos = arr2id_curpos->next;
//     }
//     printf("link\n");
     
    /* TRUE */
    //printm(Y,1,nlabel);
    Y_ind = 0.0;
    arr2id_curpos = arr2id_head;
    //printf("%.2f\n",Y_ind);
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
    
//     printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printf("%.2f\n",Y_ind);
//     printm(Y_kappa_val,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printf("---->\n");
    
    // destroy arr2id list
    while(arr2id_head)
    {
        arr2id_curpos = arr2id_head;
        arr2id_head = arr2id_head->next;
        free(arr2id_curpos->arr);
        free(arr2id_curpos);
    }
    

    
    //printf("--> middle\n");
    //printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol);
    //printm(Y_kappa_val,Y_kappa_val_nrow,Y_kappa_val_ncol);
    
    /* LOOP THROUGHT KAPPA*/
    struct type_element_list * my_list;
    struct type_element_list * cur_pos;
    my_list=NULL;
    cur_pos=NULL;
    double max_val = -1000000000;
    double max_ind = -1;
    mint max_row=0;
    mint max_col=0;

    for(mint ii=0;ii<Y_kappa_val_ncol;ii++)
    {
        double theta=0;
        /* GET CURRENT LINE THRESHOLD THETA */ 
        for(mint jj=0;jj<Y_kappa_val_nrow;jj++)
        {theta = theta + Y_kappa_val[jj+ii*Y_kappa_val_nrow];}
        /* UPATE SCORE */
        for(mint jj=0;jj<Y_kappa_val_nrow;jj++)
        {
            /* THE EMPTY LIST, BEGINNING STAGE */
            if(!my_list)
            {
                my_list = ( struct type_element_list * ) malloc (sizeof(struct type_element_list));
				my_list->id = Y_kappa_ind[jj+ii*Y_kappa_val_nrow];
				my_list->val = 0;
				my_list->next = NULL;
                //printf("-->init %.2f %.2f\n", my_list->id, my_list->val);
            }
            cur_pos = my_list;
			/* LOOP THE LIST AND UPDATE ELEMENT */
            //printf("-->on %d %d %.2f %.2f\n",jj,ii,Y_kappa_ind[jj+ii*Y_kappa_val_nrow],Y_kappa_val[jj+ii*Y_kappa_val_nrow]);
            mint find_flag=0;
			while(1)
			{
				if(cur_pos->id == Y_kappa_ind[jj+ii*Y_kappa_val_nrow])
				{
					cur_pos->val = cur_pos->val + Y_kappa_val[jj+ii*Y_kappa_val_nrow];
                    find_flag=1;
                    //printf("-->update %.2f %.2f\n", cur_pos->id, cur_pos->val);
					break;
				}
                if(cur_pos->next)
                {cur_pos = cur_pos->next;}
                else
                {break;}
			}
			/* CURRENT ELEMENT IS NOT FOUND IN THE LIST */
			if(!find_flag)
			{
				cur_pos->next = ( struct type_element_list * ) malloc (sizeof(struct type_element_list));
                cur_pos = cur_pos->next;
				cur_pos->id = Y_kappa_ind[jj+ii*Y_kappa_val_nrow];
				cur_pos->val = Y_kappa_val[jj+ii*Y_kappa_val_nrow];
				cur_pos->next = NULL;
                //printf("-->add %.2f %.2f\n", cur_pos->id, cur_pos->val);
			}
            /* ACCESS UPDATED VALUE */
            if(max_val<cur_pos->val & cur_pos->id!=Y_ind)
            {
                max_val = cur_pos->val;
                max_ind = cur_pos->id;
                max_row = jj;
                max_col = ii;
            }
            //printf("-----------%d %d %d %d %.2f %.2f %.2f\n\n", jj,ii,max_row,max_col,max_ind,max_val,theta); 
		}
        if(max_val >= theta)
        {
            break_flag = 1;
            break;
        }
        
    }
    /* DESTROY TEMPORATORY POINTER SPACE */
    
    while(my_list)
    {
        cur_pos = my_list;
        my_list = my_list->next;
        free(cur_pos);
    }  
    
    
    //mxDestroyArray(mat_Y_kappa_ind);
    free(Y_kappa_ind);
    
    /* COLLECT RESULTS */
    *(mxGetPr(OUT_YmaxVal)) = max_val;
    *(mxGetPr(OUT_break_flag)) = break_flag;
    
//     printf("--> in to worst1\n");
//     printf("%d %d %d\n",mxGetN(OUT_Ymax),mxGetN(IN_Y_kappa),nlabel);
//     printf("%d %d \n",max_row,max_col);
//     printm(Y_kappa_ind,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     printm(Y_kappa_val,Y_kappa_val_nrow,Y_kappa_val_ncol);
//     
    for(mint ii=0;ii<nlabel;ii++)
    {
//         printf("%d %d %d %d\n",max_row,max_col,max_row,(max_col*nlabel+ii)*Y_kappa_nrow); 
        Ymax[ii] = Y_kappa[max_row+(max_col*nlabel+ii)*Y_kappa_nrow];
    }
    
    
    /* printf("%d %d %.2f %.2f\n", max_row, max_col,max_ind,max_val); */
    //printf("--| out of worst\n");
}


void printm(double * M, mint nrow, mint ncol)
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
    for(mint i=0; i<nrow; i++)
    {
        for(mint j=0; j<ncol; j++)
        {
            printf("%.1f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
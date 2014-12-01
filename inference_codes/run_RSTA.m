
%%
%
% COMPILE WITH:
%   mex compute_topk_omp.c forward_alg_omp.c backward_alg_omp.c  CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" CC="/usr/local/bin/gcc -std=c99"
%   mex find_worst_violator_new.c
%
%
% PARAMETERS:
%   filename: prefix of the input file
%   graph_type: spanning tree or random pair graph
%   t: number of trees
%   isTest: TRUE to select small portion of data for testing purpose
%   kth_fold: kth fold of five fold CV
%   l_norm: '1'->l1 norm regularization, '2'->l2 norm regularization
%   maxkappa: the length of K best list
%
%
% EXAMPLE:
%   run_RSTA('ArD10','tree','5','1','1','2','2','100')
%   this will run the algorithm
%   on ArD15 dataset, with random spanning tree as output structure,
%   generating 5 random spanning tree, selecting small portion of data for
%   testing, running first fold of five fold CV, with l2 norm
%   regularization, bulding K best list with K=2
%   margin slack parameter equals to 100
%
%

function run_RSTA(filename,graph_type,t,isTest,kth_fold,l_norm,maxkappa,slack_c)

    %% Process input parameters
    if nargin <1
        disp('Not enough input parameters!')
        return;
    end
    if nargin < 2
        graph_type = 'tree';
    end
    if nargin < 3
        t = '1';
    end
    if nargin < 4
        isTest = '0';
    end
    if nargin < 5
        kth_fold = '1';
    end
    if nargin < 6
        l_norm = '2'
    end
    if nargin < 7
        maxkappa = '2';
    end
    
    % Set seed of random number
    rand('twister', 0);
    
    %losstype = 'r'; % 1 loss
    losstype = 's'; % scaled loss
    
    % Set suffix of the result files
    suffix=sprintf('%s_%s_%s_f%s_l%s_k%s_c%s_RSTA%s', filename,graph_type,t,kth_fold,l_norm,maxkappa,slack_c,losstype);
    system(sprintf('rm /var/tmp/%s.log', suffix));
    system(sprintf('rm /var/tmp/Ypred_%s.mat', suffix));
    
    % Convert parameters from string to numerical
    t=eval(t);
    isTest = eval(isTest);
    kth_fold = eval(kth_fold);
    l_norm = eval(l_norm);
    maxkappa = eval(maxkappa);
    slack_c = eval(slack_c);
    
    % Add search path
    addpath('../shared_scripts/');  
    
    % Get current hostname to be able to run on cluster
    [~,comres]=system('hostname');
    
    % Read in X and Y matrix
    if strcmp(comres(1:4),'melk') || strcmp(comres(1:4),'ukko') || strcmp(comres(1:4),'node')
        X=dlmread(sprintf('/cs/taatto/group/urenzyme/workspace/data/%s_features',filename));
        Y=dlmread(sprintf('/cs/taatto/group/urenzyme/workspace/data/%s_targets',filename));
    else
        X=dlmread(sprintf('../shared_scripts/test_data/%s_features',filename));
        Y=dlmread(sprintf('../shared_scripts/test_data/%s_targets',filename));
    end
    
    %% Process input data X and Y matrix
    % select examples which have non-zero features 
    Xsum=sum(X,2);
    X=X(Xsum~=0,:);
    Y=Y(Xsum~=0,:);
    % Select labels which is not constant over examples.
    % Note that after label selection the performance might drop because the easy-to-predict labels are removed.
    Yuniq=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        if size(unique(Y(:,i)),1)>1
            Yuniq(i)=i;
        end
    end
    Y=Y(:,Yuniq(Yuniq~=0));
    
    %Y=Y(:,1:6);
    
    % Feature normalization (tf-idf for text data, scale and centralization for other numerical features).
    if or(strcmp(filename,'medical'),strcmp(filename,'enron')) 
        X=tfidf(X);
    elseif ~(strcmp(filename(1:2),'to'))
        X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    end

    % Change Y from -1 to 0: labeling (0/1).
    Y(Y==-1)=0;

    % Get dot product kernels from normalized features or just read precomputed kernels.
    if or(strcmp(filename,'fpuni'),strcmp(filename,'cancer'))
        if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
            K=dlmread(sprintf('/cs/taatto/group/urenzyme/workspace/data/%s_kernel',filename));
        else
            K=dlmread(sprintf('../shared_scripts/test_data/%s_kernel',filename));
        end
    else
        K = X * X'; % dot product
        K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1
    end

    %% Stratified n fold cross validation index.
    nfold = 5;
    Ind = getCVIndex(Y,nfold);
    
    %% Select part of the data for code sanity check if 'isTest==1'.
    ntrain = 100;
    if isTest==1
        X=X(1:ntrain,:);
        Y=Y(1:ntrain,:);
        K=K(1:ntrain,1:ntrain);
        Ind=Ind(1:ntrain);
    end

    %% Perform parameter selection.
    % TODO: to be better implemented
    % ues results from parameter selection, otherwise use fixed parameters
    para_n=11;
    parameters=zeros(para_n,10);
    for i=1:para_n
        try
            load(sprintf('../parameters/%s_%s_1_f%d_l2_i%d_RSTAp.mat',filename,graph_type,kth_fold,i));
            parameters(i,:) = perf;
        catch err
            parameters(i,:) = [i,10,zeros(1,8)];
        end
    end
    parameters=sortrows(parameters,[3,2]);
    mmcrf_c = parameters(para_n,2);
    
    % currently use following parameters
    mmcrf_c = slack_c;
    mmcrf_g = -1e10;%0.01;
    mmcrf_i = 120;
    mmcrf_maxkappa = maxkappa;
    
    % display parameters
    fprintf('\tC:%d G:%.2f Iteration:%d\n', mmcrf_c,mmcrf_g,mmcrf_i);
    
    %% Generate random graph.
    rand('twister', 0);    
    
    Nrep=t;
    
    Nnode=size(Y,2);
    Elist=cell(Nrep,1);
    for i=1:Nrep
        if strcmp(graph_type,'tree')
            E=randTreeGenerator(Nnode); % generate
        end
        if strcmp(graph_type,'pair')
            E=randPairGenerator(Nnode); % generate
        end
        E=[E,min(E')',max(E')'];E=E(:,3:4); % arrange head and tail
        E=sortrows(E,[1,2]); % sort by head and tail
        Elist{i}=RootTree(E); % put into cell array
      
% construct similar spanning tree        
%         if i~=1
%             Elist{i}=Elist{1};
%             pos = randsample(2:(size(Elist{1},1)),1);
%             Elist{i}(pos,1) = randsample(unique(Elist{1}(1:(pos-1),:)),1);
%             Elist{i}=RootTree(Elist{i});
%         end
        
        
    end
    
    
    % Pick up one random graph.
    E=Elist{t};
    
    %% Variable to keep results.
    Ypred=zeros(size(Y));
    YpredVal=zeros(size(Y));
    running_times=zeros(nfold,1);
    muList=cell(nfold,1);

    %% nfold cross validation of base learner
    for k=kth_fold
        paramsIn.profileiter    = 40;
        paramsIn.losstype       = losstype; % losstype
        paramsIn.mlloss         = 0;        % assign loss to microlabels(0) edges(1)
        paramsIn.profiling      = 1;        % profile (test during learning)
        paramsIn.epsilon        = mmcrf_g;        % stopping criterion: minimum relative duality gap
        paramsIn.C              = mmcrf_c;        % margin slack
        paramsIn.maxkappa       = mmcrf_maxkappa;
        paramsIn.max_CGD_iter   = 1;		% maximum number of conditional gradient iterations per example
        paramsIn.max_LBP_iter   = 3;        % number of Loopy belief propagation iterations
        paramsIn.tolerance      = 1E-10;    % numbers smaller than this are treated as zero
        paramsIn.profile_tm_interval = 10;  % how often to test during learning
        paramsIn.maxiter        = mmcrf_i;        % maximum number of iterations in the outer loop
        paramsIn.verbosity      = 1;
        paramsIn.debugging      = 3;
        paramsIn.l_norm         = l_norm;
        if isTest
            paramsIn.extra_iter     = 0;        % extra iteration through examples when optimization is over
        else
            paramsIn.extra_iter     = 0;        % extra iteration through examples when optimization is over
        end
        paramsIn.filestem       = sprintf('%s',suffix);		% file name stem used for writing output

        % nfold cross validation
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        %Itrain = [Itrain;Itest(1:ceil(numel(Itest)/5))];
        gKx_tr = K(Itrain, Itrain);     % kernel
        gKx_ts = K(Itest,  Itrain)';
        gY_tr = Y(Itrain,:); gY_tr(gY_tr==0)=-1;    % training label
        gY_ts = Y(Itest,:); gY_ts(gY_ts==0)=-1;
        % set input data
        dataIn.Elist = Elist;               % edge
        dataIn.Kx_tr = gKx_tr;      % kernel
        dataIn.Kx_ts = gKx_ts;
        dataIn.Y_tr = gY_tr;        % label
        dataIn.Y_ts = gY_ts;
        % running
        [rtn,~] = RSTA(paramsIn,dataIn);
        % save margin dual mu
        muList{k}=rtn;
        % collecting results
        load(sprintf('/var/tmp/Ypred_%s.mat', paramsIn.filestem));
        
        Ypred(Itest,:)=Ypred_ts;
        %YpredVal(Itest,:)=Ypred_ts_val;
        running_times(k,1) = running_time;
    end

    
    % auc & roc random model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y(Itest,:),(Ypred(Itest,:)==1),YpredVal(Itest));
    perf = [acc,vecacc,pre,rec,f1,auc1,auc2,norm_const_quadratic_list]
    
    %% Close matlab process if it's not test run, otherwise keep matlab process alive
    if ~isTest
        %% need to save: Ypred, YpredVal, running_time, mu for current baselearner t,filename
        save(sprintf('../outputs/%s.mat', paramsIn.filestem), 'perf','Ypred', 'YpredVal', 'running_times', 'muList','norm_const_quadratic_list');
        system(sprintf('mv /var/tmp/%s.log ../outputs/', suffix));    
        exit
    end
end


%% construct a rooted tree always from node 1
function [E] = RootTree(E)
    
    clist=[1];
    nclist=[];
    workingE=[E,ones(size(E,1),1)];
    newE=[];
    while size(clist)~=0
        for j=clist
            for i=1:size(E,1)
                if workingE(i,3)==0
                    continue
                end
                if workingE(i,1)==j
                    nclist=[nclist,workingE(i,2)];
                    newE=[newE;[j,E(i,2)]];
                    workingE(i,3)=0;
                end
                if workingE(i,2)==j
                    nclist=[nclist,workingE(i,1)];
                    newE=[newE;[j,E(i,1)]];
                    workingE(i,3)=0;
                end
            end            
        end
        clist=nclist;
        nclist=[];
    end
    E=newE;
end




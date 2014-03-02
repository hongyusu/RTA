


%% Random spanning tree approximation (RSTA) for structural output prediction
function [rtn, ts_err] = RSTA(paramsIn, dataIn)
    %% define global variables
    global loss_list;   % losses associated with different edge labelings
    global mu_list;     % marginal dual varibles: these are the parameters to be learned
    global E_list;      % edges of the Markov network e_i = [E(i,1),E(i,2)];
    global ind_edge_val_list;	% ind_edge_val{u} = [Ye == u] 
    global Ye_list;             % Denotes the edge-labelings 1 <-- [-1,-1], 2 <-- [-1,+1], 3 <-- [+1,-1], 4 <-- [+1,+1]
    global Kx_tr;   % X-kernel, assume to be positive semidefinite and normalized (Kx_tr(i,i) = 1)
    global Kx_ts;
    global Y_tr;    % Y-data: assumed to be class labels encoded {-1,+1}
    global Y_ts;
    global params;  % parameters use by the learning algorithm
    global m;       % number of training instances
    global l;       % number of labels
    global primal_ub;
    global profile;
    global obj;
    global delta_obj_list;
    global opt_round;
    global Rmu_list;
    global Smu_list;
    global T_size;  % number of trees
    global cc;      % regularization constant
    global Kxx_mu_x_list;
    global kappa;   % K best
    global PAR;     % parallel compuing on matlab with matlabpool
    global kappa_decrease_flags;  
    
    if T_size >= 20
        PAR=1;
    else
        PAR =0;
    end
    
    params=paramsIn;
    Kx_tr=dataIn.Kx_tr;
    Kx_ts=dataIn.Kx_ts;
    Y_tr=dataIn.Y_tr;
    Y_ts=dataIn.Y_ts;
    E_list=dataIn.Elist;
    l = size(Y_tr,2);
    m = size(Kx_tr,1);
    T_size = size(E_list,1);
    loss_list = cell(T_size, 1);
    Ye_list = cell(T_size, 1);
    ind_edge_val_list = cell(T_size, 1);
    Kxx_mu_x_list = cell(T_size, 1);
    %cc  = 1/T_size/size(E_list{1},1);
    cc  = 1/T_size;
    mu_list = cell(T_size);
    
    if T_size == 1
        kappa_MIN = 2;
        kappa_INIT=2;
    else
        kappa_INIT = 8;
        kappa_MIN = 4; 
    end
    kappa_MAX = 64;
    
    kappa=kappa_INIT;
    
    
    for t=1:T_size
        [loss_list{t},Ye_list{t},ind_edge_val_list{t}] = compute_loss_vector(Y_tr,t,params.mlloss);
        mu_list{t} = zeros(4*size(E_list{1},1),m);
        Kxx_mu_x_list{t} = zeros(4*size(E_list{1},1),m);
    end

    
    profile_in_iteration = 1;
    
    %% initialization
    optimizer_init;
    profile_init;


    %% optimization
    print_message('Conditional gradient descend ...',0);
    primal_ub = Inf;
    opt_round = 0;
    if PAR
        par_compute_duality_gap;
    else
        compute_duality_gap;
    end
    profile_update_tr;

       
    
    %% iterate until converge
    % parameters
    obj=0;
    obj_list = zeros(1,T_size);
    prev_obj = 0;
    iter=0; 
    nflip=Inf;
    params.verbosity = 2;
    progress_made = 1;
    profile.n_err_microlbl_prev=profile.n_err_microlbl;

    
    best_n_err_microlbl=Inf;
    best_iter = iter;
    best_kappa = kappa;
    best_mu_list=mu_list;
    best_Kxx_mu_x_list=Kxx_mu_x_list;
    best_Rmu_list=Rmu_list;
    best_Smu_list=Smu_list;
    
    %% loop through examples
    while (primal_ub - obj >= params.epsilon*obj && ... % satisfy duality gap
            progress_made == 1 && ...                   % make progress
            nflip > 0 && ...                            % number of flips
            opt_round < params.maxiter ...              % within iteration limitation
            )
        opt_round = opt_round + 1;
        
        % iterate over examples 
        iter = iter +1;   
        kappa_decrease_flags = zeros(1,m);
        for xi = 1:m
        %for xi = randsample(1:m,ceil(m*0.8))
            print_message(sprintf('Start descend on example %d initial k %d',xi,kappa),3)
            if PAR
                [delta_obj_list,kappa_decrease_flags(xi)] = par_conditional_gradient_descent(xi,kappa);    % optimize on single example
            else
                [delta_obj_list,kappa_decrease_flags(xi)] = conditional_gradient_descent(xi,kappa);    % optimize on single example
            end
            obj_list = obj_list + delta_obj_list;
            obj = obj + sum(delta_obj_list);
            
            if profile_in_iteration && mod(xi,100)==1
                if PAR
                    par_compute_duality_gap;
                else
                    compute_duality_gap;
                end
                profile_update_tr;
            end
        end
        %obj_list
        progress_made = (obj >= prev_obj);  
        prev_obj = obj;
        if PAR
            par_compute_duality_gap;        % duality gap
        else
            compute_duality_gap;
        end
        profile_update_tr;          % profile update for training
        % update flip number
        if profile.n_err_microlbl > profile.n_err_microlbl_prev
            nflip = nflip - 1;
        end
        % update current best solution
        if profile.n_err_microlbl < best_n_err_microlbl
            best_n_err_microlbl=profile.n_err_microlbl;
            best_iter = iter;
            best_kappa = kappa;
            best_mu_list=mu_list;
            best_Kxx_mu_x_list=Kxx_mu_x_list;
            best_Rmu_list=Rmu_list;
            best_Smu_list=Smu_list;
        end
        % update kappa
        if sum(kappa_decrease_flags)<=m*.8
            kappa = min(kappa*2,kappa_MAX);
        else
            kappa = max(ceil(kappa/2),kappa_MIN);
        end
        
    end
    

    %% last optimization iteration
    if paramsIn.extra_iter
        iter = best_iter+1;
        kappa = best_kappa;
        mu_list = best_mu_list;
        Kxx_mu_x_list = best_Kxx_mu_x_list;
        Rmu_list = best_Rmu_list;
        Smu_list = best_Smu_list;
        for xi=1:m
            if PAR
                [~,~] = par_conditional_gradient_descent(xi,kappa);    % optimize on single example
            else
                [~,~] = conditional_gradient_descent(xi,kappa);    % optimize on single example
            end
            profile_update_tr;
            if profile.n_err_microlbl < best_n_err_microlbl
                best_n_err_microlbl=profile.n_err_microlbl;
                best_iter = iter;
                best_kappa = kappa;
                best_mu_list=mu_list;
                best_Kxx_mu_x_list=Kxx_mu_x_list;
                best_Rmu_list=Rmu_list;
                best_Smu_list=Smu_list;
            end
        end
    end
    
    %% final prediction
    iter = best_iter+1;
    kappa = best_kappa;
    mu_list = best_mu_list;
    Kxx_mu_x_list = best_Kxx_mu_x_list;
    Rmu_list = best_Rmu_list;
    Smu_list = best_Smu_list;
    profile_update;
    
    

    rtn = 0;
    ts_err = 0;
end



%% Complete part of gradient for everything
function Kmu = compute_Kmu(Kx,mu,E,ind_edge_val)

    m_oup = size(Kx,2);
    m = size(Kx,1);
    if  0 %and(params.debugging, nargin == 2)
        for x = 1:m
           Kmu(:,x) = compute_Kmu_x(x,Kx(:,x));
        end
        Kmu = reshape(Kmu,4,size(E,1)*m);
    else
        mu = reshape(mu,4,size(E,1)*m);
        Smu = reshape(sum(mu),size(E,1),m);
        term12 =zeros(1,size(E,1)*m_oup);
        Kmu = zeros(4,size(E,1)*m_oup);
        for u = 1:4
            IndEVu = full(ind_edge_val{u});    
            Rmu_u = reshape(mu(u,:),size(E,1),m);
            H_u = Smu.*IndEVu;
            H_u = H_u - Rmu_u;
            Q_u = H_u*Kx;
            term12 = term12 + reshape(Q_u.*IndEVu,1,m_oup*size(E,1));
            Kmu(u,:) = reshape(-Q_u,1,m_oup*size(E,1));
        end
        for u = 1:4
            Kmu(u,:) = Kmu(u,:) + term12;
        end
    end
    %mu = reshape(mu,mu_siz);
    return;
end


%% compute part of gradient for current example x
% Input:
%   x,Kx,t
% Output
%   gradient for current example x
function Kmu_x = compute_Kmu_x(x,Kx,E,ind_edge_val,Rmu,Smu)

    % local
    term12 = zeros(1,size(E,1));
    term34 = zeros(4,size(E,1));
    
    % main
    % For speeding up gradient computations: 
    % store sums of marginal dual variables, distributed by the
    % true edge values into Smu
    % store marginal dual variables, distributed by the
    % pseudo edge values into Rmu
   
    for u = 1:4
        Ind_te_u = full(ind_edge_val{u}(:,x));
        H_u = Smu{u}*Kx-Rmu{u}*Kx;
        term12(1,Ind_te_u) = H_u(Ind_te_u)';
        term34(u,:) = -H_u';
    end
    Kmu_x = reshape(term12(ones(4,1),:) + term34,4*size(E,1),1);
end


%% compute relative duality gap
function compute_duality_gap
    %% parameter
    global T_size;
    global Kx_tr;
    global loss_list;
    global E_list;
    global Y_tr;
    global params;
    global mu_list;
    global ind_edge_val_list;
    global primal_ub;
    global obj;
    global kappa;
    global cc;
    
    m=size(Kx_tr,1);
    Y=Y_tr;
    Ypred = zeros(size(Y));
    Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
    Y_kappa_val = zeros(size(Y,1)*T_size,kappa);
    
    %% get 'k' best prediction from each spanning tree
    Kmu_list_local = cell(1,T_size);
    gradient_list_local = cell(1,T_size);
    for t=1:T_size
        loss = loss_list{t};
        E = E_list{t};
        mu = mu_list{t};
        ind_edge_val = ind_edge_val_list{t};
        loss = reshape(loss,4,size(E,1)*m);
        Kmu_list_local{t} = compute_Kmu(Kx_tr,mu,E,ind_edge_val);
        Kmu_list_local{t} = reshape(Kmu_list_local{t},4,size(E,1)*m);
        Kmu = Kmu_list_local{t};
        gradient_list_local{t} = cc*loss - (1/T_size)*Kmu;
        gradient = gradient_list_local{t};
        [Y_tmp,Y_tmp_val] = compute_topk(gradient,kappa,E);
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp;
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp_val;
    end
    
    %% get top '1' prediction by analyzing predictions from all trees
    for i=1:size(Y,1)
        [Ypred(i,:),~,~] = ...
            find_worst_violator_matlab(...
            Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:),...
            Y_kappa_val((i:size(Y_tr,1):size(Y_kappa_val,1)),:));
    end
    clear Y_kappa;
    clear Y_kappa_val;
    
    %% duality gaps over trees
    dgap = zeros(1,T_size);
    for t=1:T_size
        loss = loss_list{t};
        E = E_list{t};
        mu = mu_list{t};
        ind_edge_val = ind_edge_val_list{t};
        Kmu = Kmu_list_local{t};
        gradient = gradient_list_local{t};
        
        Gmax = compute_Gmax(gradient,Ypred,E);
        mu = reshape(mu,4,m*size(E,1));
        duality_gap = params.C*max(Gmax,0) - sum(reshape(sum(gradient.*mu),size(E,1),m),1)';
        dgap(t) = sum(duality_gap);
    end
    %% primal upper bound
    primal_ub = obj + sum(dgap(t));
    
    return;
end
function par_compute_duality_gap
    %% parameter
    global T_size;
    global Kx_tr;
    global loss_list;
    global E_list;
    global Y_tr;
    global params;
    global mu_list;
    global ind_edge_val_list;
    global primal_ub;
    global obj;
    global kappa;
    global cc;
    
    m=size(Kx_tr,1);
    Y=Y_tr;
    Ypred = zeros(size(Y));
    Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
    Y_kappa_val = zeros(size(Y,1)*T_size,kappa);
    
    nlabel = size(Y_tr,2);

    %% get 'k' best prediction from each spanning tree
    Y_tmp = cell(1,T_size);
    Y_tmp_val = cell(1,T_size);
    Kmu_list_local = cell(1,T_size);
    gradient_list_local = cell(1,T_size);
    parfor t=1:T_size
        pause(0.000);
        loss = loss_list{t};
        mu = mu_list{t};
        E = E_list{t};
        ind_edge_val = ind_edge_val_list{t};
        loss = reshape(loss,4,size(E,1)*m);        
        Kmu_list_local{t} = compute_Kmu(Kx_tr,mu,E,ind_edge_val);
        Kmu_list_local{t} = reshape(Kmu_list_local{t},4,size(E,1)*m);
        Kmu = Kmu_list_local{t};
        gradient_list_local{t} = cc*loss - (1/T_size)*Kmu;
        gradient = gradient_list_local{t};
        [Y_tmp{t},Y_tmp_val{t}] = compute_topk(gradient,kappa,E);
    end
    for t=1:T_size
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp{t};
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp_val{t};
    end
    clear Y_tmp;
    clear Y_tmp_val;
    
    %% get top '1' prediction by analyzing predictions from all trees
    parfor i=1:size(Y,1)
        [Ypred(i,:),~,~] = ...
            find_worst_violator_matlab(Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:),...
            Y_kappa_val((i:size(Y_tr,1):size(Y_kappa_val,1)),:));
    end
    clear Y_kappa;
    clear Y_kappa_val;
    
    %% duality gaps over trees
    dgap = zeros(1,T_size);
    parfor t=1:T_size
        pause(0.000);
        loss = loss_list{t};
        E = E_list{t};
        mu = mu_list{t};
        ind_edge_val = ind_edge_val_list{t};
        Kmu = Kmu_list_local{t};
        gradient = gradient_list_local{t};
        
        Gmax = compute_Gmax(gradient,Ypred,E);
        mu = reshape(mu,4,m*size(E,1));
        duality_gap = params.C*max(Gmax,0) - sum(reshape(sum(gradient.*mu),size(E,1),m),1)';
        dgap(t) = sum(duality_gap);
    end
    %% primal upper bound
    primal_ub = obj + sum(dgap(t));
    
    return;
end




%% conditional gradient optimization, conditional on example x
% input: 
%   x   --> the id of current training example
%   obj --> current objective
%   kappa --> current kappa
function [delta_obj_list,kappa_decrease_flag] = conditional_gradient_descent(x, kappa)
    global loss_list;
    global loss;
    global Ye_list;
    global Ye;
    global E_list;
    global E;
    global mu_list;
    global mu;
    global ind_edge_val_list;
    global ind_edge_val;
    global Rmu_list;
    global Smu_list;
    global Kxx_mu_x_list;
    global cc;
    global l;
    global Kx_tr;
    global Y_tr;
    global T_size;
    global params;
    
    Y_kappa = zeros(T_size,kappa*l);        % label prediction
    Y_kappa_val = zeros(T_size,kappa);      % score
    gradient_list_local = cell(1,T_size);
    Kmu_x_list_local = cell(1,T_size);
    
    %% collect top-K prediction from each tree
    print_message(sprintf('Collect top-k prediction from each tree T-size %d', T_size),3)
    for t=1:T_size
        % variables located for tree t and example x
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};    
        % compute the quantity for tree t
        Kmu_x_list_local{t} = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu); % Kmu_x = K_x*mu_x
        Kmu_x = Kmu_x_list_local{t};
        gradient_list_local{t} =  cc*loss - (1/T_size)*Kmu_x;    % current gradient    
        gradient = gradient_list_local{t};
        % find top k violator
        [Ymax,YmaxVal] = compute_topk(gradient,kappa,E);
        % save resutls
        Y_kappa(t,:) = Ymax;
        Y_kappa_val(t,:) = YmaxVal;
    end
    
    %% get worst violator from top K
    print_message(sprintf('Get worst violator'),3)
    [Ymax, ~, kappa_decrease_flag] = find_worst_violator_matlab(Y_kappa,Y_kappa_val);
    
    
    %% if the worst violator is the correct label, exit without update mu
    if sum(Ymax~=Y_tr(x,:))==0
        delta_obj_list = zeros(1,T_size);
        %kappa_decrease_flag=1;
        return;
    end
    
    
    %% otherwise line serach
    mu_d_list = mu_list;
    nomi=zeros(1,T_size);
    denomi=zeros(1,T_size);
    kxx_mu_0 = cell(1,T_size);
    Gmax = zeros(1,T_size);
    G0 = zeros(1,T_size);
    Kmu_d_list = cell(1,T_size);
    for t=1:T_size
        % variables located for tree t and example x
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};

        %% compute
        Kmu_x = Kmu_x_list_local{t};
        gradient = gradient_list_local{t};
        Gmax(t) = compute_Gmax(gradient,Ymax,E);            % objective under best labeling
        G0(t) = -mu'*gradient;                               % current objective
        
        %% best margin violator into update direction mu_0
        Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
        mu_0 = zeros(size(mu));
        for u = 1:4
            mu_0(4*(1:size(E,1))-4 + u) = params.C*(Umax_e == u);
        end
        % compute Kmu_0
        if sum(mu_0) > 0
            smu_1_te = sum(reshape(mu_0.*Ye,4,size(E,1)),1);
            smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu),1);
            kxx_mu_0{t} = ~Ye*params.C+mu_0-smu_1_te;
        else
            kxx_mu_0{t} = zeros(size(mu));
        end
        
        Kmu_0 = Kmu_x + kxx_mu_0{t} - Kxx_mu_x_list{t}(:,x);

        mu_d = mu_0 - mu;
        Kmu_d = Kmu_0-Kmu_x;
              
        Kmu_d_list{t} = Kmu_d;
        mu_d_list{t} = mu_d;
        nomi(t) = mu_d'*gradient;
        denomi(t) = (1/T_size)*Kmu_d' * mu_d;
        
    end
    
    % decide whether to update or not
    if sum(Gmax)>=sum(G0) %&& sum(Gmax>=G0) >= T_size *1
        tau = min(sum(nomi)/sum(denomi),1);
    else
        tau=0;
    end

    %% update for each tree
    delta_obj_list = zeros(1,T_size);
    for t=1:T_size
        % variables located for tree t and example x
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        gradient =  gradient_list_local{t};
        mu_d = mu_d_list{t};
        Kmu_d = Kmu_d_list{t};
        %
        delta_obj_list(t) = gradient'*mu_d*tau - (1/T_size)*tau^2/2*mu_d'*Kmu_d;
        mu = mu + tau*mu_d;
        Kxx_mu_x_list{t}(:,x) = (1-tau)*Kxx_mu_x_list{t}(:,x) + tau*kxx_mu_0{t};
        % update Smu Rmu
        mu = reshape(mu,4,size(E,1));
        for u = 1:4
            Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
            Rmu_list{t}{u}(:,x) = mu(u,:)';
        end
        
        mu = reshape(mu,4*size(E,1),1);
        mu_list{t}(:,x) = mu;
    end
    
    return;
end
function [delta_obj_list,kappa_decrease_flag] = par_conditional_gradient_descent(x, kappa)
    global loss_list;
    global loss;
    global Ye_list;
    global Ye;
    global E_list;
    global E;
    global mu_list;
    global mu;
    global ind_edge_val_list;
    global ind_edge_val;
    global Rmu_list;
    global Smu_list;
    global Kxx_mu_x_list;
    global cc;
    global l;
    global Kx_tr;
    global Y_tr;
    global T_size;
    global params;
    
    Y_kappa = zeros(T_size,kappa*l);        % label prediction
    Y_kappa_val = zeros(T_size,kappa);      % score    
    gradient_list_local = cell(1,T_size);
    Kmu_x_list_local = cell(1,T_size);

    %% collect top-K prediction from each tree
    print_message(sprintf('Collect top-k prediction from each tree T-size %d', T_size),3)
    parfor t=1:T_size
        pause(0.000);
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};
        % compute the quantity for tree t
        Kmu_x_list_local{t} = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu); % Kmu_x = K_x*mu_x
        Kmu_x = Kmu_x_list_local{t};
        gradient_list_local{t} =  cc*loss - (1/T_size)*Kmu_x;    % current gradient    
        gradient = gradient_list_local{t};
        % find top k violator
        [Ymax,YmaxVal] = compute_topk(gradient,kappa,E);
        % save result
        Y_kappa(t,:) = Ymax;
        Y_kappa_val(t,:) = YmaxVal;
    end
    
    %% get worst violator from top K
    print_message(sprintf('Get worst violator'),3)
    [Ymax, ~, kappa_decrease_flag] = find_worst_violator_matlab(Y_kappa,Y_kappa_val);


    %% if the worst violator is the correct label, exit without update mu
    if sum(Ymax~=Y_tr(x,:))==0
        delta_obj_list = zeros(1,T_size);
        %kappa_decrease_flag=1;
        return;
    end

    
%% otherwise line serach
    mu_d_list = mu_list;
    nomi=zeros(1,T_size);
    denomi=zeros(1,T_size);
    kxx_mu_0 = cell(1,T_size);
    Gmax = zeros(1,T_size);
    G0 = zeros(1,T_size);
    Kmu_d_list = cell(1,T_size);
    parfor t=1:T_size
        pause(0.000);
        % variables located for tree t and example x
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};

        %% 
        Kmu_x = Kmu_x_list_local{t};
        gradient = gradient_list_local{t};
        Gmax(t) = compute_Gmax(gradient,Ymax,E);            % objective under best labeling
        G0(t) = -mu'*gradient;                               % current objective

        %% best margin violator into update direction mu_0
        Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
        mu_0 = zeros(size(mu));
        for u = 1:4
            mu_0(4*(1:size(E,1))-4 + u) = params.C*(Umax_e == u);
        end
        % compute Kmu_0
        if sum(mu_0) > 0
            smu_1_te = sum(reshape(mu_0.*Ye,4,size(E,1)),1);
            smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu),1);
            kxx_mu_0{t} = ~Ye*params.C+mu_0-smu_1_te;
        else
            kxx_mu_0{t} = zeros(size(mu));
        end

        
        Kmu_0 = Kmu_x + kxx_mu_0{t} - Kxx_mu_x_list{t}(:,x);

        mu_d = mu_0 - mu;
        Kmu_d = Kmu_0-Kmu_x;
        
        Kmu_d_list{t} = Kmu_d;
        mu_d_list{t} = mu_d;
        nomi(t) = mu_d'*gradient;
        denomi(t) = (1/T_size)*Kmu_d' * mu_d;

        
    end
    
    
    % decide whether to update or not
    if sum(Gmax)>=sum(G0) %&& sum(Gmax>G0)>T_size/2
        tau = min(sum(nomi)/sum(denomi),1);
    else
        tau=0;
    end
    
    %% update for each tree
    delta_obj_list = zeros(1,T_size);
    parfor t=1:T_size
        % variables located for tree t and example x
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        gradient =  gradient_list_local{t};
        mu_d = mu_d_list{t};
        Kmu_d = Kmu_d_list{t};
        % 
        delta_obj_list(t) = gradient'*mu_d*tau - (1/T_size)*tau^2/2*mu_d'*Kmu_d;
        mu = mu + tau*mu_d;
        Kxx_mu_x_list{t}(:,x) = (1-tau)*Kxx_mu_x_list{t}(:,x) + tau*kxx_mu_0{t};
        % update Smu Rmu
        mu = reshape(mu,4,size(E,1));
        for u = 1:4
            Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
            Rmu_list{t}{u}(:,x) = mu(u,:)';
        end
        
        mu = reshape(mu,4*size(E,1),1);
        mu_list{t}(:,x) = mu;
    end
     
    return;
end


%% Compute Gmax
function [Gmax] = compute_Gmax(gradient,Ymax,E)
    m = size(Ymax,1);
    
    gradient = reshape(gradient,4,size(E,1)*m);
    Umax(1,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
    Umax(2,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
    Umax(3,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
    Umax(4,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
    % sum up the corresponding edge-gradients
    Gmax = reshape(sum(gradient.*Umax),size(E,1),m);
    Gmax = reshape(sum(Gmax,1),m,1);
    
    return;
end





%% train profile and test profile update
function profile_update
    global params;
    global profile;
    global E;
    global Ye;
    global Y_tr;
    global Kx_tr;
    global Y_ts;
    global Kx_ts;
    global mu;
    global obj;
    global PAR;
    global primal_ub;
    m = size(Ye,2);
    tm = cputime;
    print_message(sprintf('tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f',...
    round(tm-profile.start_time),profile.iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj),5,sprintf('/var/tmp/%s.log',params.filestem));
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlbl_prev = profile.n_err_microlbl;

        %% train
        if PAR
            [Ypred_tr,~] = par_compute_error(Y_tr,Kx_tr);
        else
            [Ypred_tr,~] = compute_error(Y_tr,Kx_tr);
        end
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlbl = sum(profile.microlabel_errors);
        profile.p_err_microlbl = profile.n_err_microlbl/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);

        %% test
        if PAR
            [Ypred_ts,~] = par_compute_error(Y_ts,Kx_ts);
        else
            [Ypred_ts,~] = compute_error(Y_ts,Kx_ts);
        end
        profile.microlabel_errors_ts = sum(abs(Ypred_ts-Y_ts) > 0,2);
        profile.n_err_microlbl_ts = sum(profile.microlabel_errors_ts);
        profile.p_err_microlbl_ts = profile.n_err_microlbl_ts/numel(Y_ts);
        profile.n_err_ts = sum(profile.microlabel_errors_ts > 0);
        profile.p_err_ts = profile.n_err_ts/length(profile.microlabel_errors_ts);

        print_message(...
            sprintf('tm: %d 1_er_tr: %d (%3.2f) er_tr: %d (%3.2f) 1_er_ts: %d (%3.2f) er_ts: %d (%3.2f)',...
            round(tm-profile.start_time),...
            profile.n_err,...
            profile.p_err*100,...
            profile.n_err_microlbl,...
            profile.p_err_microlbl*100,...
            round(profile.p_err_ts*size(Y_ts,1)),...
            profile.p_err_ts*100,sum(profile.microlabel_errors_ts),...
            sum(profile.microlabel_errors_ts)/numel(Y_ts)*100),...
            0,sprintf('/var/tmp/%s.log',params.filestem));

        running_time = tm-profile.start_time;
        sfile = sprintf('/var/tmp/Ypred_%s.mat',params.filestem);
        save(sfile,'Ypred_tr','Ypred_ts','params','running_time');
        Ye = reshape(Ye,4*size(E,1),m);
    end
end

%% training profile
function profile_update_tr
    global params;
    global profile;
    global Y_tr;
    global Kx_tr;
    global obj;
    global primal_ub;
    global PAR;
    global kappa;
    global opt_round;
    global kappa_decrease_flags;

    tm = cputime;
    
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlbl_prev = profile.n_err_microlbl;
        % compute training error
        if PAR
            [Ypred_tr,~] = par_compute_error(Y_tr,Kx_tr);
        else
            [Ypred_tr,~] = compute_error(Y_tr,Kx_tr);
        end
        %[Ypred_tr,~] = compute_error(Y_tr,Kx_tr);
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlbl = sum(profile.microlabel_errors);
        profile.p_err_microlbl = profile.n_err_microlbl/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);
        print_message(...
            sprintf('tm: %d iter: %d 1_er_tr: %d (%3.2f) er_tr: %d (%3.2f) K: %d (%3.2f) obj: %.2f gap: %.2f %%',...
            round(tm-profile.start_time),...
            opt_round,...
            profile.n_err,...
            profile.p_err*100,...
            profile.n_err_microlbl,...
            profile.p_err_microlbl*100,...
            kappa,...
            sum(kappa_decrease_flags)/size(Y_tr,1)*100,...
            obj,...
            (primal_ub-obj)/obj*100),...
            0,sprintf('/var/tmp/%s.log',params.filestem));
    end
end

%% test error
function [Ypred,YpredVal] = compute_error(Y,Kx) 
    %% global variable
    global T_size;
    global E_list;
    global Ye_list;
    global mu_list;
    global kappa;
    Ypred = zeros(size(Y));
    YpredVal = zeros(size(Y,1),1);
    Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
    Y_kappa_val = zeros(size(Y,1)*T_size,kappa);
    %% compute 'k' best from each random spanning tree
    for t=1:T_size
        E = E_list{t};
        Ye = Ye_list{t};
        mu = mu_list{t};
        w_phi_e = compute_w_phi_e(Kx,E,Ye,mu);
        [Y_tmp,Y_tmp_val] = compute_topk(w_phi_e,kappa,E);
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp;
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp_val;
    end
    %% compute top '1' for all tree
    for i=1:size(Y,1)
        [Ypred(i,:),YpredVal(i,:),~] = ...
            find_worst_violator_matlab(...
            Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:),...
            Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:));
    end
    
    return;
end
function [Ypred,YpredVal] = par_compute_error(Y,Kx) 
    %% global variable
    global T_size;
    global E_list;
    global Ye_list;
    global mu_list;
    global kappa;
    Ypred = zeros(size(Y));
    YpredVal = zeros(size(Y,1),1);
    Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
    Y_kappa_val = zeros(size(Y,1)*T_size,kappa); 
    %% compute 'k' best from each random spanning tree
    Y_tmp = cell(1,T_size);
    Y_tmp_val = cell(1,T_size);
    parfor t=1:T_size
        pause(0.000);
        E = E_list{t};
        Ye = Ye_list{t};
        mu = mu_list{t};
        w_phi_e = compute_w_phi_e(Kx,E,Ye,mu);
        [Y_tmp{t},Y_tmp_val{t}] = compute_topk(w_phi_e,kappa,E);
    end
    for t=1:T_size
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp{t};
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp_val{t};
    end
    clear Y_tmp;
    clear Y_tmp_val;
    %% compute top '1' for all tree
    input_labels = cell(1,size(Y,1));
    input_scores = cell(1,size(Y,1));
    for i=1:size(Y,1)
        input_labels{i} = Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:);
        input_scores{i} = Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:);
    end
    parfor i=1:size(Y,1)
        pause(0.000);
        [Ypred(i,:),YpredVal(i,:)] = find_worst_violator_matlab(input_labels{i},input_scores{i});
    end
    
    return;
end

%% for testing
% Input: test kernel, tree index
% Output: gradient
function w_phi_e = compute_w_phi_e(Kx,E,Ye,mu)
    m = numel(mu)/size(E,1)/4;
    Ye = reshape(Ye,4,size(E,1)*m);   
    mu = reshape(mu,4,size(E,1)*m);
    m_oup = size(Kx,2);

    % compute gradient
    if isempty(find(mu,1))
        w_phi_e = zeros(4,size(E,1)*m_oup);
    else  
        w_phi_e = sum(mu);
        w_phi_e = w_phi_e(ones(4,1),:);
        w_phi_e = Ye.*w_phi_e;
        w_phi_e = w_phi_e-mu;
        w_phi_e = reshape(w_phi_e,4*size(E,1),m);
        w_phi_e = w_phi_e*Kx;
        w_phi_e = reshape(w_phi_e,4,size(E,1)*m_oup);
    end
    
    return;
end

%% compute loss vector
function [loss,Ye,ind_edge_val] = compute_loss_vector(Y,t,scaling)
    % scaling: 0=do nothing, 1=rescale node loss by degree
    global E_list;
    global m;
    ind_edge_val = cell(4,1);
    print_message(sprintf('Computing loss vector for %d_th Tree.',t),0);
    E = E_list{t};
    loss = ones(4,m*size(E,1));
    Te1 = Y(:,E(:,1))'; % the label of edge tail
    Te2 = Y(:,E(:,2))'; % the label of edge head
    NodeDegree = ones(size(Y,2),1);
    if scaling ~= 0 % rescale to microlabels by dividing node loss among the adjacent edges
        for v = 1:size(Y,2)
            NodeDegree(v) = sum(E(:) == v);
        end
    end
    NodeDegree = repmat(NodeDegree,1,m);    
    u = 0;
    for u_1 = [-1, 1]
        for u_2 = [-1, 1]
            u = u + 1;
            loss(u,:) = reshape((Te1 ~= u_1).*NodeDegree(E(:,1),:)+(Te2 ~= u_2).*NodeDegree(E(:,2),:),m*size(E,1),1);
        end
    end    
    loss = reshape(loss,4*size(E,1),m);
      
    Ye = reshape(loss==0,4,size(E,1)*m);
    for u = 1:4
        ind_edge_val{u} = sparse(reshape(Ye(u,:)~=0,size(E,1),m));
    end
    Ye = reshape(Ye,4*size(E,1),m);
    loss = loss*0+1;
    
    return;
end

%% initialize profile
function profile_init
    global profile;
    profile.start_time = cputime;
    profile.next_profile_tm = profile.start_time;
    profile.n_err = 0;
    profile.p_err = 0; 
    profile.n_err_microlbl = 0; 
    profile.p_err_microlbl = 0; 
    profile.n_err_microlbl_prev = 0;
    profile.microlabel_errors = [];
    profile.iter = 0;
    profile.err_ts = 0;
end

%% initialize optimizer
function optimizer_init
    clear;
    global T_size;
    global Rmu_list;
    global Smu_list;
    global obj;
    global delta_obj;
    global opt_round;
    global E_list;
    global m;
    
    Rmu_list = cell(T_size,1);
    Smu_list = cell(T_size,1);
    
    for t=1:T_size
        Rmu_list{t} = cell(1,4);
        Smu_list{t} = cell(1,4);
        for u = 1:4
            Smu_list{t}{u} = zeros(size(E_list{t},1),m);
            Rmu_list{t}{u} = zeros(size(E_list{t},1),m);
        end
    end
    
    obj=0;
    delta_obj=0;
    opt_round=0;
end

%%
function print_message(msg,verbosity_level,filename)
    global params;
    if params.verbosity >= verbosity_level
        fprintf('%s: %s\n',datestr(clock,13),msg);
        if nargin == 3
            fid = fopen(filename,'a');
            fprintf(fid,'%s: %s\n',datestr(clock,13),msg);
            fclose(fid);
        end
    end
end

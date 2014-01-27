%% base learner MMCRF

%% Debug log
% learning_MMCRFmissing:
%               1. while condition
%               2. get previous optimal mu
%               3. continuous search after get optimal mu
% optimize_x:
%               1. G0>Gmax, keep previous solution, change from
%               mu_l=zero(mu_x) to mu_l=mu_x
%               2. G0>Gmax, keep previous solution, change from
%               Kxx_mu_l=zero(Kxx_mu_x) to Kxx_mu_l=Kxx_mu_x
% compute_kmux:
%               1. initialize term12 before assigning values

%% TODO
% put nflip variable into paramsIn

%%
function [rtn,ts_err] = RSTA(paramsIn,dataIn,muIn)
    % Input data assumed by the algorithm
    global Kx_tr;   % X-kernel, assume to be positive semidefinite and normalized (Kx_tr(i,i) = 1)
    global Kx_ts;
    global Y_tr;    % Y-data: assumed to be class labels encoded {-1,+1}
    global Y_ts;
    global E;       % edges of the Markov network e_i = [E(i,1),E(i,2)];
    global params;  % parameters use by the learning algorithm
    global loss;    % losses associated with different edge labelings
    global mu;      % marginal dual varibles: these are the parameters to be learned
    global m;       % number of training instances
    global l;       % number of labels
    global Ye;      % Denotes the edge-labelings 1 <-- [-1,-1], 2 <-- [-1,+1], 3 <-- [+1,-1], 4 <-- [+1,+1]
    global IndEdgeVal;  % IndEdgeVal{u} = [Ye == u] 
    global Kmu;     % Kx_tr*mu
    global primal_ub;
    global profile;
    global obj;
    global opt_round;
    
    
    
    params=paramsIn;
    Kx_tr=dataIn.Kx_tr;
    Kx_ts=dataIn.Kx_ts;
    Y_tr=dataIn.Y_tr;
    Y_ts=dataIn.Y_ts;
    E=dataIn.E;
    
    % initialize optimization
    optimizer_init;
    
    global Rmu;
    global Smu;
    
    profile_init;

    l = size(Y_tr,2);
    m = size(Kx_tr,1);
    if nargin > 2
        mu=muIn;
    else
        mu = zeros(4*size(E,1),m);
    end

    % loss ..
    loss = compute_loss_vector(Y_tr, params.mlloss);

    % Matrices for speeding up gradient computations ..
    Ye = reshape(loss==0,4,size(E,1)*m);
    for u = 1:4
        IndEdgeVal{u} = sparse(reshape(Ye(u,:)~=0,size(E,1),m));
    end
    Ye = reshape(Ye,4*size(E,1),m);
    Kxx_mu_x = zeros(4*size(E,1),m);
    Kmu = zeros(numel(mu),1);

    % optimization
    print_message('Starting descent...',0);
    obj = 0;
    primal_ub = Inf;
    iter = 0;
    opt_round = 0;
    % for known mu do only prediction
    if nargin >2
        profile_update;
        if nargout >= 2
            ts_err=profile.n_err_microlbl_ts;
        end
        rtn=mu;
        return
    else
        profile_update_tr;
    end
    compute_duality_gap;
    profile.n_err_microlbl_prev=profile.n_err_microlbl;
    progress_made = 1;
    
    % trace the optimal solution
    prev_n_err_microlbl=profile.n_err_microlbl;
    prev_mu=mu;
    prev_obj=obj;
    prev_Kxx_mu_x=Kxx_mu_x;
    prev_Rmu=Rmu;
    prev_Smu=Smu;
    
    % allowed number of flips
    nflip=5;
    
    %
    while (primal_ub - obj >= params.epsilon*obj & ... % satisfy duality gap
            progress_made == 1 & ...   % make progress
            nflip > 0 & ...
            opt_round < params.maxiter ... % within iteration limitation
            )
        
        opt_round = opt_round + 1;
        progress_made = 0; 
        
        print_message('Conditional gradient optimization...',3)
        for x = 1:m
            % obtain initial gradient for index-x
            Kmu_x = compute_Kmu_x(x,Kx_tr(:,x));
            % conditional gradient optimization on index-x
            [mu(:,x),Kxx_mu_x(:,x),obj,x_iter] = optimize_x(x, obj, mu(:,x), Kmu_x, Kxx_mu_x(:,x),loss(:,x),Ye(:,x),params.C,params.max_CGD_iter);
            %obj0 = mu(:)'*loss(:) - (mu(:)'*reshape(compute_Kmu(Kx_tr),4*size(E,1)*m,1))/2;
            iter = iter + x_iter;
            profile.iter = iter;
        end
		progress_made =  obj > prev_obj; 
		print_message('Duality gap and primal upper bound',3);
		compute_duality_gap;
		profile.next_profile_tm = 0;
        profile_update_tr;
        
        % update flip number
        if profile.n_err_microlbl > profile.n_err_microlbl_prev
            nflip = nflip - 1;
        end
        % update current best solution
        if profile.n_err_microlbl < prev_n_err_microlbl
            prev_n_err_microlbl=profile.n_err_microlbl;
            prev_mu=mu;
            prev_obj=obj;
            prev_Kxx_mu_x=Kxx_mu_x;
            prev_Rmu=Rmu;
            prev_Smu=Smu;
        end
        
    end     % end while
    
    % restore best solution 
    mu=prev_mu;
    obj=prev_obj;
    Rmu=prev_Rmu;
    Smu=prev_Smu;
    Kxx_mu_x=prev_Kxx_mu_x;
    
    % after achieve current optimal solution, continue searching for the
    % $mu$ that minimize the training error in the next 2 iteration
    if params.extra_iter
        opt_mu=0;
        ts_err=1e10;
        tr_err=1e10;
        for x = repmat(1:m,1,params.extra_iter)
            % obtain initial gradient for index-x
            Kmu_x = compute_Kmu_x(x,Kx_tr(:,x));            
            % conditional gradient optimization on index-x
            [mu(:,x),Kxx_mu_x(:,x),obj,~] = optimize_x(x, obj, mu(:,x), Kmu_x, Kxx_mu_x(:,x),loss(:,x),Ye(:,x),params.C,params.max_CGD_iter);
            profile_update_tr;
            if tr_err>profile.n_err_microlbl
                tr_err = profile.n_err_microlbl;
                opt_mu=mu;
            end
        end
        mu=opt_mu;
    end
    profile_update;
    if nargout >= 2
        ts_err=profile.n_err_microlbl_ts;
    end
    rtn = mu;
end

%%    
function Kmu_x = compute_Kmu_x(x,Kx)
    global E;
    global IndEdgeVal;
    global Rmu;
    global Smu;
    global term12;
    global term34;
    global m;
    
    % For speeding up gradient computations: 
    % store sums of marginal dual variables, distributed by the
    % true edge values into Smu
    % store marginal dual variables, distributed by the
    % pseudo edge values into Rmu
    
    if isempty(Rmu)
        Rmu = cell(1,4);
        Smu = cell(1,4);
        for u = 1:4
            Smu{u} = zeros(size(E,1),m);
            Rmu{u} = zeros(size(E,1),m);
        end
    end
    term12=zeros(1,size(E,1));
    for u = 1:4
        Ind_te_u = full(IndEdgeVal{u}(:,x));
        H_u = Smu{u}*Kx-Rmu{u}*Kx;
        term12(1,Ind_te_u) = H_u(Ind_te_u)';
        term34(u,:) = -H_u';
    end
    Kmu_x = reshape(term12(ones(4,1),:) + term34,4*size(E,1),1);
end
 
    
function compute_duality_gap
    global E;
    global m;
    global params;
    global mu;
    global Kmu;
    global loss;
    global obj;
    global primal_ub;
    global duality_gap;
    global opt_round;
    l_siz = size(loss);
    loss = reshape(loss,4,size(E,1)*m);
    kmu_siz = size(Kmu);
    Kmu = reshape(Kmu,4,size(E,1)*m);
    gradient = loss - Kmu;
    mu_siz = size(mu);
    mu = reshape(mu,4,size(E,1)*m); gradient = reshape(gradient,4,size(E,1)*m);
    dgap = Inf; LBP_iter = 1;Gmax = -Inf;
    while LBP_iter <= size(E,1)
        LBP_iter = LBP_iter*2; % no of iterations = diameter of the graph
        if 1==1
            [~,~,G] = BestKDP(gradient); 
        else
            %[~,~,G] = max_gradient_labeling(gradient,LBP_iter); 
        end
        Gmax = max(Gmax,G);

        duality_gap = params.C*max(Gmax,0) - sum(reshape(sum(gradient.*mu),size(E,1),m),1)';
        dgap = sum(duality_gap);

        if obj+dgap < primal_ub+1E-6
            break;
        end
    end
    %primal_ub = min(obj+dgap,primal_ub);
    if primal_ub == Inf
         primal_ub = obj+dgap;
    else
         primal_ub = (obj+dgap)/min(opt_round,10)+primal_ub*(1-1/min(opt_round,10)); % averaging over a few last rounds
    end
    loss= reshape(loss,l_siz);
    Kmu = reshape(Kmu,kmu_siz);
    mu = reshape(mu,mu_siz);
end

% Conditional gradient optimizer for a single example
% mu_x, Kxx_mu_x -> a column in the matrix
function [mu_x,Kxx_mu_x,obj,iter] = optimize_x(x,obj,mu_x,Kmu_x,Kxx_mu_x,loss_x,te_x,C,maxiter)
    global E;
    global Rmu;
    global Smu;
    global IndEdgeVal;
    global params;
    global Y_tr;
    iter = 0;
    while iter < maxiter
        % calculate gradient for current example
        gradient =  loss_x - Kmu_x;
        % terminate if gradient is too small
        if norm(gradient) < params.tolerance
            break;
        end
        % find maximum gradient labeling, Ymax-labeling, Gmax-global maxima
        % under gradient labeling
        
        if 1==1
            [Ymax,~,Gmax] = BestKDP(gradient);
            YmaxK=Ymax;
            Ymax = Ymax(1,1:size(Y_tr,2));
        else   
            [Ymax,~,Gmax] = max_gradient_labeling(gradient);
        end
        [Ymax,~,Gmax] = max_gradient_labeling(gradient);
        [reshape(YmaxK,10,10)';Ymax]
        
        %if sum(sum(Ymax == Ymax_b))~=10
            %[reshape(YmaxK,10,10);Ymax_b]
        %end
        


        % gradient towards zero, current maxima
        G0 = -mu_x'*gradient;
                
        % convert labeling to update direction
        Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
        mu_1 = zeros(size(mu_x));
       
        
        if Gmax >=G0% max(params.tolerance,G0) % keep current solution
			for u = 1:4
		        mu_1(4*(1:size(E,1))-4 + u) = C*(Umax_e == u);
            end
			if sum(mu_1) > 0
			    smu_1_te = sum(reshape(mu_1.*te_x,4,size(E,1)),1);
			    smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu_x),1);
			    kxx_mu_1 = ~te_x*C+mu_1-smu_1_te;
    			%kxx_mu_1 = ones(size(te_x))*C-te_x*C-smu_1_te+mu_1;
			else
	    		kxx_mu_1 = zeros(size(mu_x));
			end
			Kmu_1 = Kmu_x + kxx_mu_1 - Kxx_mu_x;
        else % G0>Gmax, no change
            if G0 < params.tolerance
                break;
            else % keep last solution
                %kxx_mu_1 = zeros(size(mu_x));
                kxx_mu_1 = Kxx_mu_x;
                %mu_1 = zeros(size(mu_x));
                mu_1=mu_x;
                Kmu_1 = Kmu_x + kxx_mu_1 - Kxx_mu_x;
            end
        end
        d_x = mu_1 - mu_x;
        Kd_x = Kmu_1 - Kmu_x;
        l = gradient'*d_x;
        q = d_x'*Kd_x;
        alpha = min(l/q,1);
        
        delta_obj = gradient'*d_x*alpha - alpha^2/2*d_x'*Kd_x;
        if or(delta_obj <= 0,alpha <= 0)
            break;
        end
        
        mu_x = mu_x + d_x*alpha;
        Kmu_x = Kmu_x + Kd_x*alpha;
        obj = obj + delta_obj;
        Kxx_mu_x = (1-alpha)*Kxx_mu_x + alpha*kxx_mu_1;
        iter = iter + 1;
    end
    % For speeding up gradient computations: 
    % store sums of marginal dual variables, distributed by the
    % true edge values into Smu
    % store marginal dual variables, distributed by the
    % pseudo edge values into Rmu
    mu_x = reshape(mu_x,4,size(E,1));
    for u = 1:4
        Smu{u}(:,x) = (sum(mu_x)').*IndEdgeVal{u}(:,x);
        Rmu{u}(:,x) = mu_x(u,:)';
    end
    mu_x = reshape(mu_x,4*size(E,1),1);
end


% Complete gradient
function Kmu = compute_Kmu(Kx,mu0)
    global E;
    global mu;
    global IndEdgeVal;
    global params;

    if nargin < 2
        mu0 = mu;
    end
    m_oup = size(Kx,2);
    m = size(Kx,1);
    if  0 %and(params.debugging, nargin == 2)
        for x = 1:m
           Kmu(:,x) = compute_Kmu_x(x,Kx(:,x));
        end
        Kmu = reshape(Kmu,4,size(E,1)*m);
    else
        mu_siz = size(mu0);
        mu0 = reshape(mu0,4,size(E,1)*m);
        Smu = reshape(sum(mu0),size(E,1),m);
        term12 =zeros(1,size(E,1)*m_oup);
        Kmu = zeros(4,size(E,1)*m_oup);
        for u = 1:4
            IndEVu = full(IndEdgeVal{u});    
            Rmu_u = reshape(mu0(u,:),size(E,1),m);
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
end


function profile_update
    global params;
    global profile;
    global E;
    global Ye;
    global Y_tr;
    global Kx_tr;
    global Y_ts;
    global Kx_ts;
    global Y_pred;
    global Y_predVal;
    global mu;
    global obj;
    global primal_ub;
    m = size(Ye,2);
    tm = cputime;
    print_message(sprintf('alg: M3LBP tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f',...
    round(tm-profile.start_time),profile.iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj),5,sprintf('/var/tmp/%s.log',params.filestem));
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlbl_prev = profile.n_err_microlbl;

        [Ypred_tr,Ypred_tr_val] = compute_error(Y_tr,Kx_tr);
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlbl = sum(profile.microlabel_errors);
        profile.p_err_microlbl = profile.n_err_microlbl/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);

        [Ypred_ts,Ypred_ts_val] = compute_error(Y_ts,Kx_ts);
        profile.microlabel_errors_ts = sum(abs(Ypred_ts-Y_ts) > 0,2);
        profile.n_err_microlbl_ts = sum(profile.microlabel_errors_ts);
        profile.p_err_microlbl_ts = profile.n_err_microlbl_ts/numel(Y_ts);
        profile.n_err_ts = sum(profile.microlabel_errors_ts > 0);
        profile.p_err_ts = profile.n_err_ts/length(profile.microlabel_errors_ts);

        print_message(sprintf('td: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) err_ts: %d (%3.2f) ml.loss ts: %d (%3.2f) obj: %d',...
        round(tm-profile.start_time),profile.n_err,profile.p_err*100,profile.n_err_microlbl,profile.p_err_microlbl*100,round(profile.p_err_ts*size(Y_ts,1)),profile.p_err_ts*100,sum(profile.microlabel_errors_ts),sum(profile.microlabel_errors_ts)/numel(Y_ts)*100, obj),0,sprintf('/var/tmp/%s.log',params.filestem));
        %print_message(sprintf('%d here',profile.microlabel_errors_ts),4);

        running_time = tm-profile.start_time;
        sfile = sprintf('/var/tmp/Ypred_%s.mat',params.filestem);
        save(sfile,'Ypred_tr','Ypred_ts','params','Ypred_ts_val','running_time');
        Ye = reshape(Ye,4*size(E,1),m);
    end
end


function profile_update_tr
    global params;
    global profile;
    global E;
    global Ye;
    global Y_tr;
    global Kx_tr;
    global Y_ts;
    global Kx_ts;
    global Y_pred;
    global Y_predVal;
    global mu;
    global obj;
    global primal_ub;
    m = size(Ye,2);
    tm = cputime;
    print_message(sprintf('alg: M3LBP tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f',...
    round(tm-profile.start_time),profile.iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj),5,sprintf('/var/tmp/%s.log',params.filestem));
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlbl_prev = profile.n_err_microlbl;

        [Ypred_tr,Ypred_tr_val] = compute_error(Y_tr,Kx_tr);
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlbl = sum(profile.microlabel_errors);
        profile.p_err_microlbl = profile.n_err_microlbl/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);

        print_message(sprintf('td: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) obj: %d',...
        round(tm-profile.start_time),profile.n_err,profile.p_err*100,profile.n_err_microlbl,profile.p_err_microlbl*100,obj),0,sprintf('/var/tmp/%s.log',params.filestem));
        %print_message(sprintf('%d here',profile.microlabel_errors_ts),4);

        Ye = reshape(Ye,4*size(E,1),m);
    end
end


function [Ypred,YpredVal] = compute_error(Y,Kx) 
    global profile;
    global Ypred;
    global YpredVal;
    
    if isempty(Ypred)
        Ypred = zeros(size(Y));
    end
    w_phi_e = compute_w_phi_e(Kx);
    if 1==1
        [Ypred,YpredVal] = BestKDP(w_phi_e);
        Ypred = Ypred(:,1:size(Y,2));
        YpredVal = Ypred;
    else
        [Ypred,YpredVal] = max_gradient_labeling(w_phi_e);
    end
    
end


function w_phi_e = compute_w_phi_e(Kx)
    global E;
    global m;
    global Ye;
    global mu;

    Ye_siz = size(Ye);
    Ye = reshape(Ye,4,size(E,1)*m);   
    mu_siz = size(mu);
    mu = reshape(mu,4,size(E,1)*m);
    m_oup = size(Kx,2);

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
    mu = reshape(mu,mu_siz);
    Ye = reshape(Ye,Ye_siz);
end


%% linear time max sum
% calculate P(v) + \sum_{u\in chi(v)}M_{u->v}(v)
% time complexity is not O(K) now
% TODO better sorting algorithm with O(K)
function [res_sc,res_pt] = LinearMaxSum(data)
    %% parameters
    K = size(data,1); % top K
    max_nb_num = size(data,2); % maximum number of neighbor neighbors
    res_sc = zeros(K,1); % results that contain score
    res_pt = zeros(K,max_nb_num); % results that contain pointer value
    
    %% start from the first column
    cur_col = [(1:size(data,1))',data(:,1)]; % current column [index, score]
    cur_col = cur_col(cur_col(:,size(cur_col,2))~=0,:); % remove zero rows
    % if there is no score from tree below then return (meaning leave nodes)
    if numel(cur_col) == 0
        res_sc = zeros(size(data,1),1);
        res_pt = zeros(size(data,1),size(data,2));
        res_sc(1) = res_sc(1) + 1; 
        return
    end
    
    %% combine children score one at a time
    for i=2:max_nb_num
        next_col = [(1:size(data,1))',data(:,i)];
        next_col = next_col(next_col(:,size(next_col,2))~=0,:);
        % return if there is no leaves
        if numel(next_col)==0
            break
        end
        max_cur_row = min(size(cur_col,1),K);
        max_next_row = min(size(next_col,1),K);
        %res = [];
        res = zeros(max_cur_row*max_next_row,size(cur_col,2)+1);
        for p = 1:max_cur_row
            for q=1:max_next_row
                %res = [res;[cur_col(p,1:(size(cur_col,2)-1)),next_col(q,1),cur_col(p,size(cur_col,2))+next_col(q,2)]];
                res((p-1)*max_next_row+q,:) = [cur_col(p,1:(size(cur_col,2)-1)),next_col(q,1),cur_col(p,size(cur_col,2))+next_col(q,2)];
            end
        end
        res = sortrows(res,[-size(res,2)]);
        cur_col = res(1:min(K,size(res,1)),:);
    end
    
    %% collect results
    res_sc(1:size(cur_col,1),1) = cur_col(1:size(cur_col,1),size(cur_col,2)) + 1;
    res_pt(1:size(cur_col,1),1:(size(cur_col,2)-1)) = cur_col(1:size(cur_col,1),1:(size(cur_col,2)-1));
    return
end

%%


%% 
% dynamic programming fashion algorithm for top-K best inference
% assume rooted tree[par,chi;par,chi], score on edges
% get top k maximum score
% space complexity: 2*(K*|V|*max_degree)
% time complexity: 
function [Ymax,YmaxVal,Gmax] = BestKDP(gradient,K)

    %gradient = randsample(1:numel(gradient),numel(gradient));
    global E;
    
    if nargin < 2
        K=10;
    end

    %% 
    m = numel(gradient)/(4*size(E,1));
    nlabel = max(max(E));
    gradient = reshape(gradient,4,size(E,1)*m);
    min_gradient_val = min(min(gradient));
    gradient = gradient - min_gradient_val + 1e-5;
    node_degree = zeros(1,2);
    for i=1:nlabel
        node_degree(i) = sum(sum(E==i));
    end
    Ymax = zeros(m,nlabel*K);
    YmaxVal = zeros(m,K);
    if numel(gradient(gradient~=1)) == 0
            Ymax = Ymax +1;
            return
    end
    
    %% iteration throught examples
    for training_i = 1:m

        training_gradient = gradient(1:4,(training_i-1)*size(E,1)+1:(training_i*size(E,1)));
        if numel(training_gradient(training_gradient~=1)) == 0
            Ymax(training_i,:) = Ymax(training_i,:)+1;
            continue
        end        
        P_node = zeros(K*nlabel,2*max(node_degree)); % score matrix
        T_node = zeros(K*nlabel,2*max(node_degree)); % tracker matrix
        %% iterate on each edge from leave to node to propagation messages
        for i=size(E,1):-1:1
            if i==0
                break
            end

            p = E(i,1);
            c = E(i,2);
            %[training_i,i,p,c]
            % row block index for current edge (child, parent)
            row_block_chi_ind = ((c-1)*K+1):c*K;
            row_block_par_ind = ((p-1)*K+1):p*K;

            % update node score, calculate node top K list score P(v) + sum_{v'\in chi(v)}M_{v'->v}(v)
            col_block_ind = 3:2:size(P_node,2);
            [P_node(row_block_chi_ind,1),T_node(row_block_chi_ind,col_block_ind)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind));
            [P_node(row_block_chi_ind,2),T_node(row_block_chi_ind,col_block_ind+1)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind+1));

            % combine edge potential and send message to parent
            % S
            S_e = reshape(training_gradient(:,i),2,2);
            S_e = [repmat(S_e(1,:),K,1);repmat(S_e(2,:),K,1)];
            % M=S+P
            M = repmat(reshape(P_node(row_block_chi_ind,1:2),2*K,1),1,2);
            M = (M + S_e) .* (M & M);
            T = M;
            [u,v] = sort(M(:,1),'descend');
            M(:,1) = u;T(:,1) = v .* (u & u);
            [u,v] = sort(M(:,2),'descend');
            M(:,2) = u;T(:,2) = v .* (u & u);
            M = M(1:K,:);
            T = T(1:K,:);
            % put into correspond blocks
            j = sum(E(i:size(E,1),1) == p)+1;
            P_node(row_block_par_ind,(j-1)*2+1:j*2) = M;
            T_node(row_block_chi_ind,1:2) = T;
        end

        % one more iteration on root node
        row_block_chi_ind = ((p-1)*K+1):p*K;
        %disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node])
        %disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node])
        % update node score, calculate node top K list score P(v) + sum_{v'\in chi(v)}M_{v'->v}(v)
        col_block_ind = 3:2:size(P_node,2);
        [P_node(row_block_chi_ind,1),T_node(row_block_chi_ind,col_block_ind)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind));
        [P_node(row_block_chi_ind,2),T_node(row_block_chi_ind,col_block_ind+1)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind+1));
        T_node(row_block_chi_ind,1:2) = [1:K;(1:K)+K]';

        %disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node])
        %disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node])


        %% trace back
        node_degree(E(1,1)) = node_degree(E(1,1))+1;

        % root node
        for k=1:K
            if k==0
                break
            end
            % k'th best multilabel
            Y=zeros(nlabel,1);

            Q_node = P_node;
            zero_mask = zeros(K,2);

            % pick up the k'th best value from root
            p = E(1,1);
            row_block_par_ind = ((p-1)*K+1):p*K;
            [~,v] = sort(reshape(P_node(row_block_par_ind,1:2),2*K,1),'descend');
            col_pos = ceil((v(k)-1e-5)/K);
            row_pos = ceil(mod(v(k)-1e-5,K));
            YmaxVal(training_i,k) = P_node(row_pos,col_pos);
            zero_mask(row_pos,col_pos) = 1;
            Q_node(row_block_par_ind,1:2) = Q_node(row_block_par_ind,1:2) .* zero_mask;
            zero_mask(row_pos,col_pos) = 0;

            % now everthing is standardized, then we do loop
            p=0;
            for i=1:size(E,1)
                if p == E(i,1)
                    continue
                end
                p = E(i,1);
                c = E(i,2);
                %[i,p,c]
                % get block of the score matrix
                row_block_par_ind = ((p-1)*K+1):p*K;
                % get current optimal score position
                if m==1
                %[reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node]
                %[reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),Q_node]
                end
                index = find(Q_node(row_block_par_ind,1:2)~=0);
                col_pos = ceil((index-1e-5)/K);
                row_pos = ceil(mod(index-1e-5,K));
                
                %[index,col_pos,row_pos]
                % emit a label
                T_par_block = T_node(row_block_par_ind,1:2);
                Y(p) = ceil((T_par_block(row_pos, col_pos)-1e-5)/K);
                % number of children
                n_chi = node_degree(p)-1;
                %disp(sprintf('%d->%d, n_chi %d, index %d -->(%d,%d)', p,c,n_chi,index,row_pos,col_pos))
                % children in order
                cs = zeros(n_chi,1);
                j=0;
                while E(i+j,1)==p
                    cs(j+1) = E(i+j,2);
                    j=j+1;
                    if i+j>size(E,1)
                        break;
                    end
                end          
                % loop through children
                for j=size(cs,1):-1:1
                    c = cs(j);
                    %disp(sprintf('%d->%d',p,c))
                    c_pos = (n_chi-j+2);
                    col_block_c_ind = ((c_pos-1)*2+1):c_pos*2;
                    block = T_node(row_block_par_ind,col_block_c_ind);
                    index = block(row_pos,col_pos) + (col_pos-1)*K;
                    c_col_pos = ceil((index-1e-5)/K);
                    c_row_pos = ceil(mod(index-1e-5,K));
                    row_block_c_ind = ((c-1)*K+1):c*K;
                    T_chi_block = T_node(row_block_c_ind,1:2);
                    c_index = T_chi_block(c_row_pos,c_col_pos);
                    cc_col_pos = ceil((c_index-1e-5)/K);
                    cc_row_pos = ceil(mod(c_index-1e-5,K));
                    %disp(sprintf('chi index %d -->(%d,%d) %d -->(%d,%d)',index,c_row_pos,c_col_pos,c_index,cc_row_pos,cc_col_pos))
                    zero_mask(cc_row_pos,cc_col_pos) = 1;
                    Q_node(row_block_c_ind,1:2) = Q_node(row_block_c_ind,1:2) .* zero_mask;
                    zero_mask(cc_row_pos,cc_col_pos) = 0;
                    % leave node: emit directly
                    if node_degree(c) == 1
                        Y(c) = cc_col_pos;
                    end
                end
            end
            Ymax(training_i,(k-1)*nlabel+1:k*nlabel) = Y'*2-3;
        end
        node_degree(E(1,1)) = node_degree(E(1,1))-1;
    end
    
    %%
    if nargout > 2
        % find out the max gradient for each example: pick out the edge labelings
        % consistent with Ymax
        Ymax_1 = Ymax(:,1:nlabel);
        Umax(1,:) = reshape(and(Ymax_1(:,E(:,1)) == -1,Ymax_1(:,E(:,2)) == -1)',1,size(E,1)*m);
        Umax(2,:) = reshape(and(Ymax_1(:,E(:,1)) == -1,Ymax_1(:,E(:,2)) == 1)',1,size(E,1)*m);
        Umax(3,:) = reshape(and(Ymax_1(:,E(:,1)) == 1,Ymax_1(:,E(:,2)) == -1)',1,size(E,1)*m);
        Umax(4,:) = reshape(and(Ymax_1(:,E(:,1)) == 1,Ymax_1(:,E(:,2)) == 1)',1,size(E,1)*m);
        % sum up the corresponding edge-gradients
        Gmax = reshape(sum(gradient.*Umax),size(E,1),m);
        Gmax = reshape(sum(Gmax,1),m,1);
    end
    return
end


%%
function [Ymax,YmaxVal,Gmax] = max_gradient_labeling(gradient,max_iter)
    % gradient is length 4*|E| column vector containing the gradient for each edge-labeling 
    global E;
    global MBProp; % 2|E|x2|E| direction-specific adjacency matrix
    global MBPropEdgeNode;
    global params;
    if params.debugging == 1
        [Ymax,Gmax] = max_gradient_labeling_brute_force(gradient);
    else
        ineg = 1;
        ipos = 2;
        if isempty(MBProp)
            [MBProp,MBPropEdgeNode] = buildBeliefPropagationMatrix(E);
        end
        if nargin < 2
            max_iter = params.max_LBP_iter;
        end
        m = numel(gradient)/(4*size(E,1));
        g_siz = size(gradient);
        gradient = reshape(gradient,4,size(E,1)*m);
        
        
        
        % Edge-labeling specific gradient matrices m x |E|
        Gnn = reshape(gradient(1,:),size(E,1),m)'; % edge-gradients for labeling [-1,-1]
        Gnp = reshape(gradient(2,:),size(E,1),m)'; % edge-gradients for labeling [-1,+1]
        Gpn = reshape(gradient(3,:),size(E,1),m)'; % edge-gradients for labeling [+1,-1]
        Gpp = reshape(gradient(4,:),size(E,1),m)'; % edge-gradients for labeling [+1,+1]
        
        % SumMsg_*_*: mx|E| matrices storing the sums of neighboring messages from
        % the head and tail of the edge, respectively, on the condition that
        % the head (resp. tail) is labeled with -1 --> neg or +1 --> pos.
        SumMsg_head_neg = zeros(m,size(E,1));
        SumMsg_head_pos = zeros(m,size(E,1));
        SumMsg_tail_neg = zeros(m,size(E,1));
        SumMsg_tail_pos = zeros(m,size(E,1));
        
        iTail = 1:size(E,1);
        iHead = size(E,1)+iTail;
        
        % Iterate until messages have had time to go accros the whole graph: at
        % most this takes O(|E|) iterations (i.e. when the graph is a chain)
        for iter = 1:max_iter
            % find max-gradient configuration and propage gradient value over the edge
            Msg_head_neg = max(SumMsg_tail_pos+Gpn,SumMsg_tail_neg+Gnn);
            Msg_head_pos = max(SumMsg_tail_pos+Gpp,SumMsg_tail_neg+Gnp);
            Msg_tail_neg = max(SumMsg_head_pos+Gnp,SumMsg_head_neg+Gnn);
            Msg_tail_pos = max(SumMsg_head_pos+Gpp,SumMsg_head_neg+Gpn);
            
            % Sum up gradients of consistent configurations and propage to neighboring
            % edges
            SumMsg_tail_neg = [Msg_tail_neg,Msg_head_neg]*MBProp(:,iTail);
            SumMsg_tail_pos = [Msg_tail_pos,Msg_head_pos]*MBProp(:,iTail);
            SumMsg_head_neg = [Msg_tail_neg,Msg_head_neg]*MBProp(:,iHead);
            SumMsg_head_pos = [Msg_tail_pos,Msg_head_pos]*MBProp(:,iHead);
        end
        
        % find out the labeling: sum up the edge messages coming towards each node
        M_max1 = [Msg_tail_neg,Msg_head_neg]*MBPropEdgeNode;
        M_max2 = [Msg_tail_pos,Msg_head_pos]*MBPropEdgeNode;
        % pick the label of maximum message value
        Ymax = (M_max1 <= M_max2)*2-1;
        % get predicted value
        YmaxVal = (M_max2 - M_max1);

        normModel=1;
        if normModel==1 % normalize by edge degree
            NodeDegree = ones(size(YmaxVal,2),1);
            for v = 1:size(YmaxVal,2)
                NodeDegree(v) = sum(E(:) == v);
            end
            YmaxVal=YmaxVal./repmat(NodeDegree',size(YmaxVal,1),1);
        end
        if normModel==2 % normailze into unit vector length
            if size(YmaxVal,1) > 1
                YmaxValNorm=[];
                for i=1:size(YmaxVal,1)
                    YmaxValNorm=[YmaxValNorm;norm(YmaxVal(i,:))];
                end
                YmaxVal=YmaxVal./repmat(YmaxValNorm,1,size(YmaxVal,2));
            end
        end
        

        if nargout > 2
            % find out the max gradient for each example: pick out the edge labelings
            % consistent with Ymax
            Umax(1,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
            Umax(2,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
            Umax(3,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
            Umax(4,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
            % sum up the corresponding edge-gradients
            Gmax = reshape(sum(gradient.*Umax),size(E,1),m);
            Gmax = reshape(sum(Gmax,1),m,1);
        end
        gradient = reshape(gradient,g_siz);
    end
end


% Construct a matrix containing the neighborhood information of the edges.
% The matrix consists of four blocks, corresponding to the edges that merge 
% (e(2) = e'(2)), branch (e(1) = e'(1)), form a chain forward (e(2) =
% e'(1)) or backward (e(1) = e'(2))
function [MBProp,MBPropEdgeNode] = buildBeliefPropagationMatrix(E)
    MBProp = zeros(size(E,1)*2); % for edge to edge propagation
    MBPropEdgeNode = zeros(size(E,1)*2,max(max(E))); % for edge to node propagation

    numEdges = size(E,1);

    iTail = 1:numEdges;
    iHead = iTail+numEdges;

    for node = 1:max(max(E))

      eTail = find(E(:,1) == node);
      eHead = find(E(:,2) == node);

      % Edges that meet node with their tail
      MBPropEdgeNode(iTail(eTail),node) = 1;
      % Edges that meet node with the head
      MBPropEdgeNode(iHead(eHead),node) = 1;

      % Matrix block for progating messages from edges that meet with their
      % tails at node (eTail); 
      Link = MBProp(iTail,iTail); 
      Link(eTail,eTail) = 1;
     % remove diagonal; we do not propage messages back to self 
      MBProp(iTail,iTail) = Link-diag(diag(Link));

      % Matrix block for progating messages via a backward chain (eTail meeting eHead) at node;
      % messages will go from iTail to iTail (excluding self loops)
      Link = MBProp(iTail,iHead);
      Link(eTail,eHead) = 1;
      % remove diagonal; in case there are self loops e = (v,v) in the graph
      MBProp(iTail,iHead) = Link-diag(diag(Link));

      % Matrix block for progating messages from edges that meet with their
      % heads at node (eHead)
      Link = MBProp(iHead,iHead);
      Link(eHead,eHead) = 1; 
      % remove diagonal; we do not propage messages back to self
      MBProp(iHead,iHead) = Link-diag(diag(Link));

      % Matrix block for progating messages  via a forward chain (eHead meeting
      % eTail) at node;
      Link = MBProp(iHead,iTail);
      Link(eHead,eTail) = 1;
      % remove diagonal; in case there are self loops e = (v,v) in the graph
      MBProp(iHead,iTail) = Link-diag(diag(Link));

    end
end

%%
function loss = compute_loss_vector(Y,scaling)
    % saling: 0=do nothing, 1=rescale node loss by degree
    global E;
    global m;

    print_message('Computing loss vector...',0);

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
    return
end

%%
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

%%
function optimizer_init
    clear global MBProp;
    clear global MBPropEdgeNode;
    clear global Rmu;
    clear global Smu;
    clear global term12;
    clear global term34;
end

%%
function print_message(msg,verbosity_level,filename)
    global params;
    if params.verbosity >= verbosity_level
        fprintf('%s: %s\n',datestr(clock),msg);
        if nargin == 3
            fid = fopen(filename,'a');
            fprintf(fid,'%s: %s\n',datestr(clock),msg);
            fclose(fid);
        end
    end
end






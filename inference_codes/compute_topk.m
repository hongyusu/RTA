
%% compute topk labels
function [Ymax,YmaxVal,Gmax] = compute_topk(gradient,K,E)
    %tic
    if nargin < 3
        disp('Wrong input parameters!');
        return
    end
    
    %% 
    m = numel(gradient)/(4*size(E,1));
    nlabel = max(max(E));
    gradient = reshape(gradient,4,size(E,1)*m);
    min_gradient_val = min(min(gradient));
    gradient = gradient - min_gradient_val + 1e-5;
    node_degree = zeros(1,nlabel);
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
        %% get training gradient
        training_gradient = gradient(1:4,((training_i-1)*size(E,1)+1):(training_i*size(E,1)));
        %% forward algorithm to get P_node and T_node
        %[P_node1,T_node1] = forward_alg_matlab(training_gradient,K,E,nlabel,node_degree,max(max(node_degree)));
%         disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node])
%         disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node])
        [P_node,T_node] = forward_alg(training_gradient,K,E,nlabel,node_degree,max(max(node_degree)));
%         disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node])
%         disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node])

%         if sum(sum(T_node~=T_node1))>0
%             disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node])
%             disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node1])
%             disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node-P_node1])
%             disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node])
%             disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node1])
%             disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node-T_node1])
%             afdsfsd
%         end
        

  
        %% backward algorithm to get Ymax and YmaxVal
%         try
%         [Ymax_single, YmaxVal_single] = backward_alg_matlab(P_node, T_node, K, E, nlabel, node_degree);
%         catch
%        if node_degree(1)==2
%             a=[reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node];
%             b=[reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node];
%             disp(a(550:640,:))
%             disp(b(550:640,:))
%             [P_node,T_node] = forward_alg_matlab(training_gradient,K,E,nlabel,node_degree,max(max(node_degree)));
%             a=[reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node];
%             b=[reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node];
%             disp(a(550:640,:))
%             disp(b(550:640,:))
%          afdsdf
%        end
%         end

        [Ymax_single, YmaxVal_single] = backward_alg_matlab(P_node, T_node, K, E, nlabel, node_degree);
        %[Ymax_single, YmaxVal_single] = backward_alg(P_node, T_node, K, E, nlabel, node_degree);
        %asfsdf
        clear P_node;
        clear Q_node;
            
            
        
        Ymax(training_i,:) = Ymax_single;
        YmaxVal(training_i,:) = YmaxVal_single;       
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
    
    %toc
    return
end
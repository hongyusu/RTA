%% 
function [Ymax, YmaxVal] = majority_voting(Y_kappa,Y_kappa_val,nlabel)

    use_n_label = 1;
    reshape( Y_kappa(:,1:(nlabel*use_n_label)),size(Y_kappa,1)*use_n_label,nlabel) .* repmat(reshape(Y_kappa_val(:,1:use_n_label),size(Y_kappa,1)*use_n_label,1),1,nlabel) 
    Ymax = sum(reshape( Y_kappa(:,1:(nlabel*use_n_label)),size(Y_kappa,1)*use_n_label,nlabel) .* repmat(reshape(Y_kappa_val(:,1:use_n_label),size(Y_kappa,1)*use_n_label,1),1,nlabel) );
    Ymax = (Ymax>=0)*2-1;
    YmaxVal = mean(reshape(Y_kappa_val(:,1:(use_n_label)),size(Y_kappa,1)*use_n_label,1));
    
    return
end
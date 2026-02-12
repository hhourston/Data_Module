%% Make scree plots of all data sets: spontaneous instability, dual pycnocline, and collision for PLOS

%% Spontaneous Instability Data Set
clear all, close all;
load spont_instability_data.mat
rho = reshape(myrhos,3001*156,131);
data = rho; 

[M,N] = size(data);
N_spont=N;

        %% EOF
        data_temp = bsxfun(@minus, data, mean(data,2)); % remove mean

        %% Compute EOFs by svds
        [u,s,v]=svds(data_temp/sqrt(N-1),N); % perform the SVD, v is not used.
        % the columns of u are the eigenvectors.
        lambda_spont = diag(s).^2; % the singular values need to be squared to match the eigenvalues from pca.
        % but svd ensures they're already in decreasing order.
        lambda_spont = lambda_spont./sum(lambda_spont); %normalize scree
        
%% Dual Pycnocline Data Set
load dual_pyc_data.mat
rho = reshape(myrhos,3001*128,100);
data = rho; 

[M,N] = size(data);
N_core = N;
        %% EOF
        data_temp = bsxfun(@minus, data, mean(data,2)); % remove mean

        %% Compute EOFs by svds
        [u,s,v]=svds(data_temp/sqrt(N-1),N); % perform the SVD, v is not used.
        % the columns of u are the eigenvectors.
        lambda_core = diag(s).^2; % the singular values need to be squared to match the eigenvalues from pca.
        % but svd ensures they're already in decreasing order.
        lambda_core = lambda_core./sum(lambda_core); %normalize scree
        
%% Collision Data Set
load collision_data.mat
rho = reshape(rhostrim,3072*81,150);
data = rho; 

[M,N] = size(data);
N_coll = N;
        %% EOF
        data_temp = bsxfun(@minus, data, mean(data,2)); % remove mean

        %% Compute EOFs by svds
        [u,s,v]=svds(data_temp/sqrt(N-1),N); % perform the SVD, v is not used.
        % the columns of u are the eigenvectors.
        lambda_coll = diag(s).^2; % the singular values need to be squared to match the eigenvalues from pca.
        % but svd ensures they're already in decreasing order.
        lambda_coll = lambda_coll./sum(lambda_coll); %normalize scree
        
%% Scree        
fig_w = 1500; % figure width
fig_h = 900; % figure height
make_figs = 1;

   figure('position', [200, 200, fig_w , fig_h]); 
    set(gcf,'DefaultLineLineWidth',3,'DefaultTextFontSize',24,...
        'DefaultTextFontWeight','bold','DefaultAxesFontSize',30,...
          'DefaultAxesFontWeight','bold');
      
      last_eig = 30;
      plot(1:last_eig,lambda_spont(1:last_eig),'r')
      hold on
      plot(1:last_eig,lambda_core(1:last_eig),'b')      
      plot(1:last_eig,lambda_coll(1:last_eig),'k')
      xlim([1 last_eig])
      legend('spontaneous instability','dual pycnocline','collision')
      grid on
      xlh = xlabel('$k$','Interpreter', 'LaTeX')
      xlh.Position(1) = xlh.Position(1) - 0.5;
      ylh=ylabel('$\frac{\lambda_k}{\sum_{i} \lambda_i}$','Interpreter', 'LaTeX','FontSize',40)
      set(get(gca,'YLabel'),'Rotation',0)
      ylh.Position(2) = ylh.Position(2) + 0.023;  %left, bottom, width, height
      ylh.Position(1) = ylh.Position(1) - 2;  %left, bottom, width, height      

      
                figname = strcat('PLOS_scree');
                style = hgexport('factorystyle');
                hgexport(gcf,figname,style,'Format','eps')
             
            
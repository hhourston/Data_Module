
clear all
load Monterey_Bay.mat

x = 0:1:90;

evecs1 = zeros(35,length(x));
evecs2 = zeros(35,length(x));
evecs3 = zeros(35,length(x));
evecs4 = zeros(35,length(x));


j = 1;
for i = x
    i
    mydata = myden;
    
    
    percentnans = i;
    [m,n] = size(mydata);
    numnans = round(n*percentnans/100);
    nanvec = randperm(n,numnans);
    
    mydata(15,nanvec) = NaN;   
    [evecs,score,evals]=pca(mydata','Rows','complete');
        
    evecs1(:,j) = evecs(:,1);
    evecs2(:,j) = evecs(:,2);
    evecs3(:,j) = evecs(:,3);
    evecs4(:,j) = evecs(:,4);
    
    k = j + 1;
    j = k;
end

for ii=2:26
    mydiff1(ii-1)=norm(evecs1(:,ii)-evecs1(:,1))/norm(evecs1(:,1));
    mydiff2(ii-1)=norm(evecs2(:,ii)-evecs2(:,1))/norm(evecs2(:,1));
    mydiff3(ii-1)=norm(evecs3(:,ii)-evecs3(:,1))/norm(evecs3(:,1));
    mydiff4(ii-1)=norm(evecs4(:,ii)-evecs4(:,1))/norm(evecs4(:,1));
end

figure(1)
clf
betterplots
plot(1:25,mydiff1,'bo-',1:25,mydiff2,'m*-',1:25,mydiff3,'rs-',1:25,mydiff4,'kp-')
xlabel('percent NaN at grid point 15')
ylabel('Norm of difference in evector')
legend('EOF 1','EOF2','EOF 3','EOF 4','Location','NorthWest')
grid on
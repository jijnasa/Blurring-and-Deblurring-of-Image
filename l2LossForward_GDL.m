function y = l2LossForward_GDL(x,r)
delta = (x - r) ;
y = sum(delta(:).^2);
y = y / (size(x,1) * size(x,2)) ;  % normalize by image size

%beta = 1; reg=0; lambda =1e-2;%size(r)
%for i=1:size(r,4)
%   reg = reg + tv(r(:,:,:,i),beta);
%end
%y = y + lambda*reg;
alpha = 2;lambda = 1e-1;
p1 = abs(r(2:(size(r,1)),:)-r(1:(size(r,1)-1),:)) - abs(x(2:(size(x,1)),:)-x(1:(size(x,1)-1),:));
p2 = abs(r(:,1:(size(r,2)-1))-r(:,2:(size(r,2)))) - abs(x(:,1:(size(x,2)-1))-x(:,2:(size(x,2))));
reg = sum(p1(:).^alpha)+sum(p2(:).^alpha);
reg = reg/ (size(x,1) * size(x,2)) ;
y = y + lambda*reg;

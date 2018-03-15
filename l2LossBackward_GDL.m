function dx = l2LossBackward_GDL(x,r,p)
dy = 2 * p * (x - r) ;
dy = dy / (size(x,1) * size(x,2)) ;  % normalize by image size

%for finding gradient of gdl regularization 
term1 = zeros(size(x));
term2 = zeros(size(x));
term3 = zeros(size(x));
term4 = zeros(size(x));

term1(1:(size(x,1)-1),:) = 2 * (abs(r(2:(size(r,1)),:)-r(1:(size(r,1)-1),:))-abs(x(2:(size(x,1)),:)-x(1:(size(x,1)-1),:))) .* sign(x(2:(size(x,1)),:)-x(1:(size(x,1)-1),:));
term2(2:(size(x,1)),:) =-2 * (abs(r(2:(size(r,1)),:)-r(1:(size(r,1)-1),:))-abs(x(2:(size(x,1)),:)-x(1:(size(x,1)-1),:))) .* sign(x(2:(size(x,1)),:)-x(1:(size(x,1)-1),:));
term3(:,2:(size(x,2))) = 2 * (abs(r(:,1:(size(r,1)-1))-r(:,2:(size(r,1))))-abs(x(:,1:(size(x,2)-1))-x(:,2:(size(x,2))))) .* sign(x(:,1:(size(x,2)-1))-x(:,2:(size(x,2))));
term4(:,1:(size(x,2)-1)) =-2 * (abs(r(:,1:(size(r,1)-1))-r(:,2:(size(r,1))))-abs(x(:,1:(size(x,2)-1))-x(:,2:(size(x,2))))) .* sign(x(:,1:(size(x,2)-1))-x(:,2:(size(x,2))));
dreg = p * (term1 + term2 + term3 + term4);
dreg = dreg / (size(x,1) * size(x,2)) ;

lambda = 1e-1;
dx = dy + lambda * dreg;
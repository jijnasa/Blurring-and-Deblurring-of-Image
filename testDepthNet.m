function testDepthNet
net = load('H:\Research\Projects\Blur\Encoder-decoder-BCL-GDL\practical-cnn-reg-master\data\train_Blur_BCEupX16\net-epoch-2000.mat');
net = net.net;
net.layers(end) = [] ;

%load('E:\Ganesh\Codes\Gopi\SALICON\validate.mat');
dataDir = 'H:\Research\Datasets\Blur_Segmentation\Test\images\';
labelDir = 'H:\Research\Datasets\Blur_Segmentation\Test\gt\';
outdir = 'H:\Research\Projects\Blur\Encoder-decoder-results\';
gtdir = 'E:\Ganesh\Datasets\Fence\Imgs\ground_truth\';
for i = 2:2:296
    filename = strcat(dataDir,'motion (',num2str(i),').jpg');
    im = im2single(imread(filename));
    gt = imread(strcat(labelDir,'motion (',num2str(i),').png'));
    %im = imresize(im,[224,224]) ;
    %lab = load([labelDir validate(i).image '.mat']);
    %lab = imresize(lab.I,[224,224]) ;
    %label = single(lab) ;
    res = vl_simplenn(net, im);   
    blurmap = res(end).x;
    %     imagesc(fence,[-1,0]);
    
    figure ; set(gcf, 'name', 'results') ; clf ;
    subplot(1,3,1) ; imagesc(im) ;
    axis off image ; title('Input') ;
    subplot(1,3,2) ; imagesc(blurmap) ;
    axis off image ; title('Desired output (sharp)') ;
    subplot(1,3,3) ; imagesc(gt) ;
    axis off image ; title('GT') ;
    colormap gray ;
%     subplot(1,3,3) ;
%     imagesc(salmap) ; axis image off  ;
%     title('CNN output (trained 150 epoch)') ;
     outName = strcat(strcat(outdir,'motion (',num2str(i),').jpg'));%[outdir validate(i).image '.png'];
%     gtName = [gtdir validate(i).image '.png'];
     imwrite(blurmap,outName);
%     imwrite(label, gtName);
    pause(2);
    close all;
end
end


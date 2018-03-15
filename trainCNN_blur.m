setup() ;
% setup('useGpu', true); % Uncomment to initialise with a GPU support

%% Part 3.1: Prepare the data

% Load a database of blurred images to train from
imdb = load('H:\Research\Projects\Jijnasa\Encoder-decoder-BCL-GDL\practical-cnn-reg-master\data\blur_imdb.mat') ;dataDir = 'H:\Research\Datasets\Blur_224\images\';
labelDir = 'H:\Research\Datasets\Blur_224\gt\';

imdb = imdb.imdb;
net = initializeVGGCNN() ;
%net = initializeSalN16();


% Visualize the first image in the database

%imdb = load('data/saliency_imdb.mat') ;
%imdb = imdb.imdb;
batch = 5;

path = strcat(dataDir,imdb.images.data(batch));
path = path{1};
im_ = im2single(imread(path)) ;
%im = single(im) ;% note: 255 range

im = single(im_) ; % note: 255 range
%im = imresize(im, net.meta.normalization.imageSize(1:2)) ;
%im = bsxfun(@minus,im,net.meta.normalization.averageImage) ;

% im = imresize(im,[224,224]) ;
% pathl = strcat(labelDir,imdb.images.label(batch));
% pathl = pathl{1};
% lab = load(pathl);
% label = imresize(lab.I,[224,224]) ;
% label = single(label) ;
% figure(31) ; set(gcf, 'name', 'Part 3.1: Data') ; clf ;
% 
% subplot(1,2,1) ; imagesc(im_) ;
% axis off image ; title('Input') ;
% 
% subplot(1,2,2) ; imagesc(lab.I) ;
% axis off image ; title('Desired output (sharp)') ;
% 
% colormap gray ;


%% Part 3.2: Create a network architecture
%
% The expected input size (a single 64 x 64 x 1 image patch). This is
% used for visualization purposes.

%net = initializeVGGCNN() ;
%net = initializeLargeCNN() ;

% Display network
%vl_simplenn_display(net) ;

%Evaluate network on an image
res = vl_simplenn(net, im) ;

figure(32) ; clf ; colormap gray ;
set(gcf,'name', 'Part 3.2: network input') ;
subplot(1,2,1) ;
imagesc(im) ; axis image off  ;
title('CNN input') ;

set(gcf,'name', 'Part 3.2: network output') ;
subplot(1,2,2) ;
result = res(end).x;
imagesc(result) ; axis image off  ;
title('CNN output (not trained yet)') ;

 %% Part 3.3: learn the model

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward_GDL, @l2LossBackward_GDL) ;

% Extra: uncomment the following line to use your implementation
% of the L1 loss
%net = addCustomLossLayer(net, @l1LossForward, @l1LossBackward) ;

% Train
trainOpts.expDir = 'data/train_Blur_BCEupX16_softmax' ;
trainOpts.gpus = [] ;
% Uncomment for GPU training:
%trainOpts.expDir = 'data/text-small-gpu' ;
%trainOpts.gpus = [1] ;
trainOpts.batchSize = 32 ;
trainOpts.learningRate = 1e-3 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 3000;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 3.4: evaluate the model

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

figure(33) ; set(gcf, 'name', 'Part 3.4: Results on the training set') ;
showDeblurringResult(net, imdb, train(1:30:151)) ;

figure(34) ; set(gcf, 'name', 'Part 3.4: Results on the validation set') ;
showDeblurringResult(net, imdb, val(1:30:151)) ;

figure(35) ;
set(gcf, 'name', 'Part 3.4: Larger example on the validation set') ;
colormap gray ;
subplot(1,2,1) ; imagesc(imdb.examples.blurred{1}, [-1, 0]) ;
axis image off ;
title('CNN input') ;
res = vl_simplenn(net, imdb.examples.blurred{1}) ;
subplot(1,2,2) ; imagesc(res(end).x, [-1, 0]) ;
axis image off ;
title('CNN output') ;

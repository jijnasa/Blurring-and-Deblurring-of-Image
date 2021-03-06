function net = initializeVGGCNN()
%INITIALIZESMALLCNN  Initialize a small CNN for text deblurring
%   NET = INITIALIZESMALLCNN() returns the SimpleNN model NET.

net1 = load('H:\Research\Projects\Jijnasa\Encoder-decoder-BCL-GDL\practical-cnn-reg-master\data\imagenet-vgg-verydeep-16.mat') ;
net.layers = net1.layers(1:30) ;
net.meta = {};
net.meta.inputs = net1.meta.inputs;
net.meta.normalization = net1.meta.normalization;

net.layers{end+1} = struct(...
  'name', 'conv6_1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,512,512)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu6_1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv6_2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,512,512)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu6_2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv6_3', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,512,512)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu6_3', ...
  'type', 'relu') ;


net.layers{end+1} = struct(...
  'name', 'upsample1', ...
  'type', 'convt', ...
  'weights', {bilinearW(4,512,512)}, ...
  'upsample', 2, ...
  'crop', 1) ;

%net.layers{end+1} = struct(...
%  'name', 'relu6_1', ...
%  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv7_1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,512,512)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu7_1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv7_2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,512,512)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu7_2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv7_3', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,512,256)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu7_3', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'upsample2', ...
  'type', 'convt', ...
  'weights', {bilinearW(4,256,256)}, ...
  'upsample', 2, ...
  'crop', 1) ;

%net.layers{end+1} = struct(...
%  'name', 'relu7_1', ...
% 'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv8_1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,256,256)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu8_1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv8_2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,256,256)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu8_2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv8_3', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,256,128)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu8_3', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'upsample3', ...
  'type', 'convt', ...
  'weights', {bilinearW(4,128,128)}, ...
  'upsample', 2, ...
  'crop', 1) ;

%net.layers{end+1} = struct(...
%  'name', 'relu8_1', ...
%  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv9_1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,128,128)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu9_1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv9_2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,128,64)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu9_2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'upsample4', ...
  'type', 'convt', ...
  'weights', {bilinearW(4,64,64)}, ...
  'upsample', 2, ...
  'crop', 1) ;

%net.layers{end+1} = struct(...
%  'name', 'relu10_1', ...
%  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv10_1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,64,64)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;


net.layers{end+1} = struct(...
  'name', 'relu9_2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv10_2', ...
  'type', 'conv', ...
  'weights', {xavier(1,1,64,1)}, ...
  'pad', 0, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

% net.layers{end+1} = struct(...
%   'name', 'softmax', ...
%   'type', 'softmax') ;

net.layers{end+1} = struct(...
  'name', 'sigmoid', ...
  'type', 'sigmoid') ;


%net.layers{end+1} = struct('type', 'softmaxloss') ;
% Consolidate the network, fixing any missing option
% in the specification above

net = vl_simplenn_tidy(net) ;
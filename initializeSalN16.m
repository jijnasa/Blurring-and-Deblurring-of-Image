function net = initializeSalN16()
%INITIALIZESMALLCNN  Initialize a small CNN for text deblurring
%   NET = INITIALIZESMALLCNN() returns the SimpleNN model NET.

net1 = load('data/imagenet-vgg-verydeep-16.mat') ;
net.layers = net1.layers(1:30) ;
net.meta = {};
net.meta.inputs = net1.meta.inputs;
net.meta.normalization = net1.meta.normalization;

net.layers{end+1} = struct(...
  'name', 'upsampleq16x', ...
  'type', 'convt', ...
  'weights', {bilinearW(8, 512, 512)}, ...
  'upsample', 2, ...
  'crop', 3) ;

net.layers{end+1} = struct(...
  'name', 'conv10_2', ...
  'type', 'conv', ...
  'weights', {xavier(1,1,512,1)}, ...
  'pad', 0, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'sigmoid', ...
  'type', 'sigmoid') ;


net = vl_simplenn_tidy(net) ;
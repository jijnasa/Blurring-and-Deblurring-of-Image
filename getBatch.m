function [im, label] = getBatch(imdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.

dataDir = 'H:\Research\Datasets\Blur_224\images\';
labelDir = 'H:\Research\Datasets\Blur_224\gt\';

%imdb = load('data/saliency_imdb.mat') ;
%imdb = imdb.imdb;
%batch = 3;

path = strcat(dataDir,imdb.images.data(batch));
path = path{1};
im_ = im2single(imread(path)) ;
%im = single(im) ;% note: 255 range

im = single(im_) ; % note: 255 range
%im = imresize(im, net.meta.normalization.imageSize(1:2)) ;
%im = bsxfun(@minus,im,net.meta.normalization.averageImage) ;

im = imresize(im,[224,224]) ;
pathl = strcat(labelDir,imdb.images.label(batch));
pathl = pathl{1};
lab = im2single(imread(pathl));
label = imresize(lab,[224,224]) ;
if size(lab,3) == 3
    label = rgb2gray(label) ;
end
%label = label.*(255/max(max(label)));
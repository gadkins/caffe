addpath(genpath('/home/sdsuvision3/caffe/matlab/maxflow'));
addpath(genpath('/home/sdsuvision3/caffe/matlab/UGM'));
seg_path = '/home/sdsuvision3/caffe/models/pascalpart-fcn32s/person/larm/segmentation_results';
seg_imgs = dir([seg_path, '/', '*.png']);
crf_dir = '/home/sdsuvision3/caffe/models/pascalpart-fcn32s/person/larm/optimal_decoding/';

for ii = 1:numel(seg_imgs)
    imname = seg_imgs(ii).name;
    img = imread([seg_path, '/', imname]);
    [X,map] = rgb2ind(img, 60);
    % no segmentation result
    if (numel(unique(img)) == 1)
        imwrite(img, fullfile(crf_dir,imname));
        continue
    end
    [nRows,nCols,~] = size(img);
    [nodePot, edgePot, edgeStruct] = get_potentials(img);
    optimalDecoding = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
    opt = reshape(optimalDecoding,nRows,nCols);
    opt8 = uint8(opt);
    imwrite(opt8, colormap(jet), fullfile(crf_dir,imname));
end
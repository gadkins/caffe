imagesFolder = dir(['/home/cv/DeconvNet/data/Pascal-Part/images/person/rleg','/*.jpg']);
N_files = length(imagesFolder(not([imagesFolder.isdir])));

intial_file = imread(imagesFolder(1).name);
[h,w,c] = size(intial_file);
trainset = zeros(h,w,c,N_files,'uint8');

for ii = 1:N_files
    trainset(:,:,:,ii) = imread(imagesFolder(ii).name);    
end

meanR = mean(squeeze(trainset(:,:,1,:)),3);
meanR = mean(meanR(:));

meanG = mean(squeeze(trainset(:,:,2,:)),3);
meanG = mean(meanG(:));

meanB = mean(squeeze(trainset(:,:,3,:)),3);
meanB = mean(meanB(:));


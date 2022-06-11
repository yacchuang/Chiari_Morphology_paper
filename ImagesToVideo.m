
%% Read images sequence
workingDir = "/Users/kurtlab/Desktop/Chiari_Morphometric/screenshot";
imageNames = dir(fullfile(workingDir,'3Dseg','*.png'));
imageNames = {imageNames.name}';

%% Create New Video with the Image Sequence
outputVideo = VideoWriter(fullfile(workingDir,'shuttle_out.avi'));
open(outputVideo)

for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,'3Dseg',imageNames{ii}));
   writeVideo(outputVideo,img)
end

close(outputVideo);

%% View the Final Video
shuttleAvi = VideoReader(fullfile(workingDir,'shuttle_out.avi'));

ii = 1;
while hasFrame(shuttleAvi)
   mov(ii) = im2frame(readFrame(shuttleAvi));
   ii = ii+1;
end

figure 
imshow(mov(1).cdata, 'Border', 'tight')

movie(mov,1,shuttleAvi.FrameRate)
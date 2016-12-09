clear all;
tdir = dir('test*');
imageName = [];
for i = 1:size(tdir,1)
    imageName = char(imageName, fullfile(tdir(i).name));
end
imageName = imageName(2:end,:);
mkdir('invert');
for i = 1:size(imageName,1)
    imname = strrep(imageName(i,:),' ','');
    img = imread(imname);
    img = double(img);
    img = 256-img;
    img(img<=25) = 0;
    img = uint8(round(img));
    imwrite(img,fullfile('invert',imname));
end
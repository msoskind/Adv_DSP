% Finding circles in an image of an eye

close all;

newD = 250;
filename = 'eye_60.jpg';

A = imread(['cropped/', filename]);
A = imresize(A, [newD NaN]);
A3(:,:,3) = 128; A3(:,:,2) = 128; 
A3 = rgb2gray(imresize(A3, [newD NaN]));
A1 = rgb2gray(A);

% figure()
% imshow(A1)
B1 = imbinarize(A(:,:,1), .06);
B2 = imbinarize(A(:,:,2), .45);
A2 = imsharpen(A1, 'Radius',50,'Amount',1);
% B2 = imbinarize(A2, .15);
figure()
imshow(A)

p_max = 15;
p_min = 9;

pup_range_1 = [floor(size(B1,1)/12),floor(size(B1,1)/7)];
% pup_range_1 = [5,20];
pup_range_2 = [floor(size(B2,1)/4),floor(size(B2,1)/1.7)];
[center_p, radii_p] = imfindcircles(B1, pup_range_1, 'ObjectPolarity', 'dark')
% [center_i, radii_i] = imfindcircles(B2, pup_range_2, 'ObjectPolarity', 'dark');
[center_i, radii_i] = imfindcircles(B2, pup_range_2, 'EdgeThreshold', .01, 'ObjectPolarity', 'dark')
viscircles(center_p(1,:), radii_p(1,:),'EdgeColor','r');
viscircles(center_i(1,:), radii_i(1,:),'EdgeColor','b');
center_p = center_p(1,:); radii_p = radii_p(1,:);
center_i = center_i(1,:); radii_i = radii_i(1,:);

fprintf('%s\t%6.2f\t%6.2f\t%6.2f\t%6.2f\t%6.2f\t%6.2f\n',filename, center_p(1), center_p(2), radii_p, center_i(1), center_i(2), radii_i);
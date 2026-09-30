clear;
% Resolve the data directory relative to this script's location
% (repo_root/src/original_single_split/<this file>), so it runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
data_dir = fullfile(repo_root, 'data', 'processed_combined');
% Figures are likewise written under repo_root, not the caller's
% current directory, so output lands in the same place regardless of
% where this script is invoked from.
output_dir = fullfile(repo_root, 'results', 'original_single_split', 'triplet');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

target_bodipy = [1,3,9];
% 
% target_bodipy = [1:2,4:8];
% target_bodipy = [target_bodipy, target_bodipy + 9];

wl_min = 600;
wl_max = 700;

cutoff_val = 1500;

n_bodipy = length(target_bodipy);

misidentification_sc = ones(n_bodipy,n_bodipy,n_bodipy);
accuracy_matrix_cnn = nan(n_bodipy,n_bodipy,n_bodipy);

n_molecules_total = zeros(n_bodipy,1);
mean_spectral_centroid = zeros(n_bodipy,1);

rng(9);

% plot the n_photons vs. spectral centroids
for i_bodipy = 1:n_bodipy
    i_curr_bodipy = target_bodipy(i_bodipy);
    load(fullfile(data_dir, sprintf('bodipy%d.mat',i_curr_bodipy)));
    i_n_photons = n_photons_combined;
    i_spectral_rois = spectral_rois_combined;
    i_spectral_centroid = spectral_centroids_combined;
    i_spectral_profiles = spectral_profiles_combined;

    % sort the spectral profiles by n_photons, descending
    [i_n_photons, sort_idx] = sort(i_n_photons,'descend');
    i_spectral_rois = i_spectral_rois(:,:,sort_idx);
    i_spectral_centroid = i_spectral_centroid(sort_idx);
    i_spectral_profiles = i_spectral_profiles(:,sort_idx);

    % cutoff the data at n_photons = cutoff_val
    i_spectral_rois = i_spectral_rois(:,:,i_n_photons > cutoff_val);
    i_spectral_profiles = i_spectral_profiles(:,i_n_photons > cutoff_val);
    i_spectral_centroid = i_spectral_centroid(i_n_photons > cutoff_val);
    i_n_photons = i_n_photons(i_n_photons > cutoff_val);

    % remove outliers of the spectral centroids
    [i_spectral_centroid,i_rm_rows] = rmoutliers(i_spectral_centroid,'mean','thresholdfactor',4);
    % update the spectral profiles
    i_spectral_rois = i_spectral_rois(:,:,~i_rm_rows);
    i_spectral_profiles = i_spectral_profiles(:,~i_rm_rows);
    i_n_photons = i_n_photons(~i_rm_rows);
    
    n_molecules_total(i_bodipy) = length(i_n_photons);
    mean_spectral_centroid(i_bodipy) = mean(i_spectral_centroid);
    
    for j_bodipy = (1+i_bodipy):n_bodipy
        j_curr_bodipy = target_bodipy(j_bodipy);
        load(fullfile(data_dir, sprintf('bodipy%d.mat',j_curr_bodipy)));
        j_n_photons = n_photons_combined;
        j_spectral_rois = spectral_rois_combined;
        j_spectral_centroid = spectral_centroids_combined;
        j_spectral_profiles = spectral_profiles_combined;

        % sort the spectral profiles by n_photons, descending
        [j_n_photons, sort_idx] = sort(j_n_photons,'descend');
        j_spectral_rois = j_spectral_rois(:,:,sort_idx);
        j_spectral_centroid = j_spectral_centroid(sort_idx);
        j_spectral_profiles = j_spectral_profiles(:,sort_idx);

        % cutoff the data at n_photons = cutoff_val
        j_spectral_rois = j_spectral_rois(:,:,j_n_photons > cutoff_val);
        j_spectral_profiles = j_spectral_profiles(:,j_n_photons > cutoff_val);
        j_spectral_centroid = j_spectral_centroid(j_n_photons > cutoff_val);
        j_n_photons = j_n_photons(j_n_photons > cutoff_val);

        % remove outliers of the spectral centroids
        [j_spectral_centroid,j_rm_rows] = rmoutliers(j_spectral_centroid,'mean','thresholdfactor',4);
        % update the spectral profiles
        j_spectral_rois = j_spectral_rois(:,:,~j_rm_rows);
        j_spectral_profiles = j_spectral_profiles(:,~j_rm_rows);
        j_n_photons = j_n_photons(~j_rm_rows);
        
        for k_bodipy = (1+j_bodipy):n_bodipy
            k_curr_bodipy = target_bodipy(k_bodipy);
            load(fullfile(data_dir, sprintf('bodipy%d.mat',k_curr_bodipy)));
            k_n_photons = n_photons_combined;
            k_spectral_rois = spectral_rois_combined;
            k_spectral_centroid = spectral_centroids_combined;
            k_spectral_profiles = spectral_profiles_combined;

            % sort the spectral profiles by n_photons, descending
            [k_n_photons, sort_idx] = sort(k_n_photons,'descend');
            k_spectral_rois = k_spectral_rois(:,:,sort_idx);
            k_spectral_centroid = k_spectral_centroid(sort_idx);
            k_spectral_profiles = k_spectral_profiles(:,sort_idx);

            % cutoff the data at n_photons = cutoff_val
            k_spectral_rois = k_spectral_rois(:,:,k_n_photons > cutoff_val);
            k_spectral_profiles = k_spectral_profiles(:,k_n_photons > cutoff_val);
            k_spectral_centroid = k_spectral_centroid(k_n_photons > cutoff_val);
            k_n_photons = k_n_photons(k_n_photons > cutoff_val);

            % remove outliers of the spectral centroids
            [k_spectral_centroid,k_rm_rows] = rmoutliers(k_spectral_centroid,'mean','thresholdfactor',4);
            % update the spectral profiles
            k_spectral_rois = k_spectral_rois(:,:,~k_rm_rows);
            k_spectral_profiles = k_spectral_profiles(:,~k_rm_rows);
            k_n_photons = k_n_photons(~k_rm_rows);
            
            % perform classification test based on spectral centroids
            % we select the number of bodipy molecules in the mean spectral centroid for each dye, plus minus 3
            sc_window = 3;
            % first we check if the mean spectral centroid is within the sc_window of the other dye
            if ~(abs(mean(i_spectral_centroid) - mean(j_spectral_centroid)) < sc_window) && ...
               ~(abs(mean(i_spectral_centroid) - mean(k_spectral_centroid)) < sc_window) && ...
               ~(abs(mean(j_spectral_centroid) - mean(k_spectral_centroid)) < sc_window)
               
                % if it isn't, we count the number of dyes with spectral centroid in each window
                i_sc_window = mean(i_spectral_centroid) + [-sc_window, sc_window];
                j_sc_window = mean(j_spectral_centroid) + [-sc_window, sc_window];
                k_sc_window = mean(k_spectral_centroid) + [-sc_window, sc_window];

                % total number of molecules
                n_molecules = length(i_spectral_centroid) + length(j_spectral_centroid) + length(k_spectral_centroid);

                % count the number of molecules in each window
                n_i_sc_window = sum(j_spectral_centroid > i_sc_window(1) & j_spectral_centroid < i_sc_window(2)) + ...
                                sum(k_spectral_centroid > i_sc_window(1) & k_spectral_centroid < i_sc_window(2));
                n_j_sc_window = sum(i_spectral_centroid > j_sc_window(1) & i_spectral_centroid < j_sc_window(2)) + ...
                                sum(k_spectral_centroid > j_sc_window(1) & k_spectral_centroid < j_sc_window(2));
                n_k_sc_window = sum(i_spectral_centroid > k_sc_window(1) & i_spectral_centroid < k_sc_window(2)) + ...
                                sum(j_spectral_centroid > k_sc_window(1) & j_spectral_centroid < k_sc_window(2));

                % calculate the fraction of misidentified molecules
                misidentification_sc(i_bodipy,j_bodipy,k_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(i_bodipy,k_bodipy,j_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(j_bodipy,i_bodipy,k_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(j_bodipy,k_bodipy,i_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(k_bodipy,i_bodipy,j_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(k_bodipy,j_bodipy,i_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
            end
        
            % perform classification test based on CNN
            % 400 brightest images for training, 100 images for testing        
            
            % check that we have at least 500 spectral rois for each dye
            if ~(size(i_spectral_rois,3) < 500 || size(j_spectral_rois,3) < 500 || size(k_spectral_rois,3) < 500)
                % % take the brightest 500 images for each dye
                % i_spectral_rois = i_spectral_rois(:,:,1:500);
                % j_spectral_rois = j_spectral_rois(:,:,1:500);
                % k_spectral_rois = k_spectral_rois(:,:,1:500);

                % scramble the indices for training and testing
                i_idx = randperm(size(i_spectral_rois,3),500);
                j_idx = randperm(size(j_spectral_rois,3),500);
                k_idx = randperm(size(k_spectral_rois,3),500);

                i_train_rois = i_spectral_rois(:,:,i_idx(1:400));
                i_test_rois = i_spectral_rois(:,:,i_idx(401:end));
                j_train_rois = j_spectral_rois(:,:,j_idx(1:400));
                j_test_rois = j_spectral_rois(:,:,j_idx(401:end));
                k_train_rois = k_spectral_rois(:,:,k_idx(1:400));
                k_test_rois = k_spectral_rois(:,:,k_idx(401:end));

                train_labels = [ones(400,1); 2*ones(400,1); 3*ones(400,1)];
                test_labels = [ones(100,1); 2*ones(100,1); 3*ones(100,1)];

                train_idx = randperm(1200);
                test_idx = randperm(300);

                X_train = cat(3,i_train_rois,j_train_rois,k_train_rois);
                X_test = cat(3,i_test_rois,j_test_rois,k_test_rois);

                X_train = X_train(:,:,train_idx(1:1200));
                X_test = X_test(:,:,test_idx(1:300));
                
                Y_train = train_labels(train_idx(1:1200));
                Y_test = test_labels(test_idx(1:300));

                training_labels = categorical(Y_train);
                validation_labels = categorical(Y_test);
                % permute the dimensions of the training and testing images
                training_images = permute(X_train,[1 2 4 3]);
                validation_images = permute(X_test,[1 2 4 3]);

                sz = size(training_images);
                n_labels = 3;
                
                % train the CNN
                layers = [
                    imageInputLayer([sz(1) sz(2) sz(3)])
                
                    convolution2dLayer(3,8,'Padding','same')
                    batchNormalizationLayer
                    reluLayer
                
                    maxPooling2dLayer(2,'Stride',2);
                
                    convolution2dLayer(3,16,'Padding','same')
                    batchNormalizationLayer
                    reluLayer
                
                    fullyConnectedLayer(n_labels)
                    softmaxLayer
                    classificationLayer];
                
                options = trainingOptions('sgdm', ...
                    'InitialLearnRate',0.01, ...
                    'MaxEpochs',20, ...
                    'Shuffle','every-epoch', ...
                    'ValidationData',{validation_images,validation_labels}, ...
                    'ValidationFrequency',30, ...
                    'Verbose',false);
                
                net = trainNetwork(training_images,training_labels,layers,options);
                validation_predicted = classify(net,validation_images);
                
                accuracy_matrix_cnn(i_bodipy,j_bodipy,k_bodipy) = sum(validation_predicted == validation_labels)/numel(validation_labels);
                accuracy_matrix_cnn(i_bodipy,k_bodipy,j_bodipy) = sum(validation_predicted == validation_labels)/numel(validation_labels);
                accuracy_matrix_cnn(j_bodipy,i_bodipy,k_bodipy) = sum(validation_predicted == validation_labels)/numel(validation_labels);
                accuracy_matrix_cnn(j_bodipy,k_bodipy,i_bodipy) = sum(validation_predicted == validation_labels)/numel(validation_labels);
                accuracy_matrix_cnn(k_bodipy,i_bodipy,j_bodipy) = sum(validation_predicted == validation_labels)/numel(validation_labels);
                accuracy_matrix_cnn(k_bodipy,j_bodipy,i_bodipy) = sum(validation_predicted == validation_labels)/numel(validation_labels);
            end
        end
    end
end

% flatten the matrices to visualize in 2D
misidentification_sc = min(misidentification_sc,[],3);
accuracy_matrix_cnn_2d = confusionmat(validation_labels, validation_predicted) ./ 100;

% % fix the matrices to be symmetric
% misidentification_sc = misidentification_sc + misidentification_sc';
% accuracy_matrix_cnn = accuracy_matrix_cnn + accuracy_matrix_cnn';

% set the diagonal to 1
misidentification_sc(1:n_bodipy+1:end) = 1;
% accuracy_matrix_cnn_2d(1:n_bodipy+1:end) = nan;

% plot the cnn classification matrix as a heatmap
f4 = figure(4); clf;
imagesc(accuracy_matrix_cnn_2d);
set(gca,'XTick',1:n_bodipy);
set(gca,'XTickLabel',target_bodipy);
set(gca,'YTick',1:n_bodipy);
set(gca,'YTickLabel',target_bodipy);
% cb = colorbar();

% create a colorbar, with worst values in red and best values in green
cmap_g = [161, 230, 170] ./ 255;
cmap_y = [255, 247, 172] ./ 255;
cmap_r = [255, 153, 179] ./ 255;

intervals1 = linspace(0,1,6);
intervals2 = linspace(0,1,6);
customcmap = [interp1([0; 1],[cmap_r; cmap_y],intervals1(1:end-1)); ...
              interp1([0; 1],[cmap_y; cmap_g],intervals2)];
colormap(customcmap);            
cb = colorbar();
% set the colorbar limits
caxis([0,1]+[-0.025,0.025]);
xlabel('Predicted Class');
ylabel('True Class');
title('CNN Accuracy matrix');

for i1 = 1:size(accuracy_matrix_cnn_2d)
    for i2 = 1:size(accuracy_matrix_cnn_2d)
        text(i1,i2,sprintf('%d',100*accuracy_matrix_cnn_2d(i2,i1)),...
            'horizontalalignment','center', ...
            'verticalalignment','middle');
    end
end

formatfig(f4,'presentation',fullfile(output_dir,'cnn_accuracy.eps'));

% plot the spectral centroid misidentification as a heatmap
f5 = figure(5); clf;
imagesc(1-misidentification_sc);
set(gca,'XTick',1:n_bodipy);
set(gca,'XTickLabel',target_bodipy);
set(gca,'YTick',1:n_bodipy);
set(gca,'YTickLabel',target_bodipy);

% create a colorbar, with worst values in red and best values in green
cmap_g = [161, 230, 170] ./ 255;
cmap_y = [255, 247, 172] ./ 255;
cmap_r = [255, 153, 179] ./ 255;

intervals1 = linspace(0,1,6);
intervals2 = linspace(0,1,6);
customcmap = [interp1([0; 1],[cmap_r; cmap_y],intervals1(1:end-1)); ...
              interp1([0; 1],[cmap_y; cmap_g],intervals2)];
colormap(customcmap);            
cb = colorbar();
caxis([0.5,1]+[-0.025,0.025]);

title('Spectral Centroid Accuracy');

%% Get the list of the highest values
f6 = figure(6); clf;
C = confusionmat(validation_labels, validation_predicted);
cm = confusionchart(C,target_bodipy);

cm.DiagonalColor = '#A1E6AA';
cm.OffDiagonalColor = '#FF99B3';

formatfig(f6,'presentation',fullfile(output_dir,'cnn_numbers.eps'));

% print the overall accuracy
accuracy = sum(diag(C))/sum(C(:));
fprintf('Overall accuracy: %.2f%%\n',accuracy*100);

%% ========================================================================
%  Franka Robot: Multi-to-Multi SSIM Skill Retrieval System
%  ========================================================================
clear; clc; close all;

%% 1. Path Definition and Directory Verification
basePath = '~/Documents/zsc_Franka/data/lib'; 
templateDir = fullfile(basePath, 'templates');       
testSampleDir = fullfile(basePath, 'test_samples');  

if ~exist(testSampleDir, 'dir')
    mkdir(testSampleDir);
    error('Created test directory: %s\nPlease place unknown PNG skill images there and rerun the script.', testSampleDir);
end

if ~exist(templateDir, 'dir')
    error('Reference library not found. Please verify the folder exists: %s', templateDir);
end

%% 2. File Scanning and Discovery
templateFiles = dir(fullfile(templateDir, '*.png'));
testFiles = dir(fullfile(testSampleDir, '*.png'));

if isempty(templateFiles)
    error('No reference images detected in templates/ directory.');
end

if isempty(testFiles)
    error('No evaluation images detected in test_samples/ directory.');
end

CONFIDENCE_THRESHOLD = 0.03; 
fprintf('Reference skills count: %d | Unknown test samples count: %d \n', length(templateFiles), length(testFiles));

%% 3. Outer Loop: Process Each Unknown Test Sample
for t = 1:length(testFiles)
    currentTestName = testFiles(t).name;
    currentTestPath = fullfile(testFiles(t).folder, currentTestName);
    
    fprintf('\nAnalyzing test sample %d/%d: %s\n', t, length(testFiles), currentTestName);
    
    % Load test image
    imgTest = imread(currentTestPath);
    if size(imgTest, 3) == 3, imgTest = rgb2gray(imgTest); end
    
    % Temporary storage allocation
    skillMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    globalMaxScore = -1;
    globalBestMatchFile = '';
    
    %% 4. Inner Loop: Match Against All Templates
    for i = 1:length(templateFiles)
        refPath = fullfile(templateFiles(i).folder, templateFiles(i).name);
        imgRef = imread(refPath);
        if size(imgRef, 3) == 3, imgRef = rgb2gray(imgRef); end
        
        % Align template dimension to test image
        if any(size(imgRef) ~= size(imgTest))
            imgRef = imresize(imgRef, size(imgTest));
        end
        
        % Structural Similarity Index measurement
        currentScore = ssim(imgTest, imgRef);
        
        % Filename parser for regex skill extraction
        [~, fileName, ~] = fileparts(templateFiles(i).name);
        tokens = regexp(fileName, '^([a-zA-Z_]+?)(?=\d*$|_?\d*$)', 'tokens');
        if ~isempty(tokens)
            skillClass = tokens{1}{1};
            if endsWith(skillClass, '_'), skillClass = skillClass(1:end-1); end
        else
            skillClass = fileName;
        end
        
        % Organize scores by class
        if isKey(skillMap, skillClass)
            skillMap(skillClass) = [skillMap(skillClass), currentScore];
        else
            skillMap(skillClass) = currentScore;
        end
        
        % Track the absolute best template match
        if currentScore > globalMaxScore
            globalMaxScore = currentScore;
            globalBestMatchFile = fileName;
        end
    end % End of Inner Matching Loop
    
    %% 5. Class Metric Statistics and Decision Making
    skillClasses = keys(skillMap);
    numClasses = length(skillClasses);
    avgScores = zeros(1, numClasses);
    
    for c = 1:numClasses
        avgScores(c) = mean(skillMap(skillClasses{c}));
    end
    
    [~, bestClassIdx] = max(avgScores);
    predictedSkillClass = skillClasses{bestClassIdx};
    
    sortedAvgScores = sort(avgScores, 'descend');
    if length(sortedAvgScores) > 1
        confidenceGap = sortedAvgScores(1) - sortedAvgScores(2);
    else
        confidenceGap = sortedAvgScores(1);
    end
    
    %% 6. Generation of Console Evaluation Report
    fprintf('   - Closest Template Match: %s (SSIM: %.4f)\n', globalBestMatchFile, globalMaxScore);
    fprintf('   - Category Average Score Breakdown:\n');
    
    for c = 1:numClasses
        currentClass = skillClasses{c};
        currentAvg = avgScores(c);
        fprintf('       [%s] Average Score: %.4f\n', currentClass, currentAvg);
    end
    
    fprintf('   - Final Decision Result: Predicted Category is [%s].\n', predictedSkillClass);
    
    if confidenceGap < CONFIDENCE_THRESHOLD
        fprintf('     Warning: Low confidence gap (%.4f) below threshold (%.4f). Structural overlap detected.\n', confidenceGap, CONFIDENCE_THRESHOLD);
    end
    fprintf('--------------------------------------------------\n');
end % End of Outer Sample Iteration Loop

fprintf('\nEvaluation completed across all samples.\n');
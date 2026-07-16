%% ========================================================================
%  Franka Robot: Multi-to-Multi SIFT-Flow Skill Retrieval System
%  ========================================================================
clear; clc; close all;

%% 1. Environment Path Initialization
SIFT_FLOW_ROOT = '/home/yue/Documents/zsc_Franka/similarity/SIFT-Flow-master/SIFTflow';

addpath(fullfile(SIFT_FLOW_ROOT, 'mexDenseSIFT'));
addpath(fullfile(SIFT_FLOW_ROOT, 'mexDiscreteFlow'));
addpath(SIFT_FLOW_ROOT); 

if isempty(which('mexDenseSIFT'))
    error('Core function mexDenseSIFT not found. Please verify SIFT_FLOW_ROOT path.');
end

%% 2. Data Directory Definition
basePath = '/home/yue/Documents/zsc_Franka/data/lib'; 
templateDir = fullfile(basePath, 'templates');       
testSampleDir = fullfile(basePath, 'test_samples');  

templateFiles = dir(fullfile(templateDir, '*.png'));
testFiles = dir(fullfile(testSampleDir, '*.png'));

if isempty(templateFiles) || isempty(testFiles)
    error('Please check that PNG images exist in templates/ or test_samples/ folders.');
end

CONFIDENCE_THRESHOLD = 0.15; 
fprintf('Reference skills count: %d | Unknown test samples count: %d \n', length(templateFiles), length(testFiles));

%% 3. Outer Loop: Process Each Unknown Test Sample
for t = 1:length(testFiles)
    currentTestName = testFiles(t).name;
    currentTestPath = fullfile(testFiles(t).folder, currentTestName);
    
    fprintf('\nAnalyzing test sample %d/%d: %s\n', t, length(testFiles), currentTestName);
    
    % Load test image
    imgTest = imread(currentTestPath);
    imgTest = im2double(imgTest); 
    if size(imgTest, 3) == 3, imgTest = rgb2gray(imgTest); end
    imgTest = imresize(imgTest, [200, NaN]); 
    
    % Temporary storage allocation
    rawEnergies = zeros(1, length(templateFiles));
    tempFileNames = cell(1, length(templateFiles));
    tempSkillClasses = cell(1, length(templateFiles));
    
    %% 4. Inner Loop - Phase 1: Feature Extraction and Raw SIFT-Flow Energy Computation
    for i = 1:length(templateFiles)
        refPath = fullfile(templateFiles(i).folder, templateFiles(i).name);
        imgRef = imread(refPath);
        imgRef = im2double(imgRef);   
        if size(imgRef, 3) == 3, imgRef = rgb2gray(imgRef); end
        imgRef = imresize(imgRef, size(imgTest));
        
        % SIFT-Flow dense feature matching execution
        cellsize = 3; gridspacing = 1;
        siftTest = mexDenseSIFT(imgTest, cellsize, gridspacing);
        siftRef  = mexDenseSIFT(imgRef, cellsize, gridspacing);
        
        SIFTflowpara.alpha = 2 * 255;
        SIFTflowpara.d = 40 * 255;
        SIFTflowpara.gamma = 0.005 * 255;
        SIFTflowpara.nlevels = 4;
        SIFTflowpara.wsize = 2;
        SIFTflowpara.topwsize = 10;
        SIFTflowpara.nTopIterations = 60;
        SIFTflowpara.nIterations = 30;
        
        [~, ~, energy] = SIFTflowc2f(siftTest, siftRef, SIFTflowpara);
        
        % Struct field parsing compatibility check
        if isstruct(energy)
            if isfield(energy, 'total'), val = energy.total;
            elseif isfield(energy, 'energy'), val = energy.energy;
            else, fields = fieldnames(energy); val = energy.(fields{1}); end
        else
            val = energy;
        end
        rawEnergies(i) = double(sum(val(:)));
        
        % Filename parser for regex skill extraction
        [~, fileName, ~] = fileparts(templateFiles(i).name);
        tempFileNames{i} = fileName;
        tokens = regexp(fileName, '^([a-zA-Z_]+?)(?=\d*$|_?\d*$)', 'tokens');
        if ~isempty(tokens)
            skillClass = tokens{1}{1};
            if endsWith(skillClass, '_'), skillClass = skillClass(1:end-1); end
        else
            skillClass = fileName;
        end
        tempSkillClasses{i} = skillClass;
    end % End of Inner Loop Phase 1
    
    %% 5. Phase 2: Open-Set Verification and Adaptive Score Mapping
    minE = min(rawEnergies);
    maxE = max(rawEnergies);
    
    REJECTION_THRESHOLD = 1.45e9;
    
    isUnknownSkill = false;
    if minE > REJECTION_THRESHOLD
        isUnknownSkill = true;
    end
    
    denom = maxE - minE;
    if denom < 1e-6, denom = 1e-6; end
    
    skillMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    globalMaxScore = -1;
    globalBestMatchFile = '';
    
    for i = 1:length(templateFiles)
        % Map total energy inversely to a relative [0, 1] range
        currentScore = 1 - (rawEnergies(i) - minE) / denom;
        
        skillClass = tempSkillClasses{i};
        fileName = tempFileNames{i};
        
        if isKey(skillMap, skillClass)
            skillMap(skillClass) = [skillMap(skillClass), currentScore];
        else
            skillMap(skillClass) = currentScore;
        end
        
        if currentScore > globalMaxScore
            globalMaxScore = currentScore;
            globalBestMatchFile = fileName;
        end
    end % End of Normalization Mapping
    
    %% 6. Class Metric Statistics and Decision Making
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
    
    %% 7. Generation of Console Evaluation Report
    fprintf('   - Raw Alignment Minimum Energy (minE): %.2e\n', minE); 
    
    if isUnknownSkill
        fprintf('   - Category Average Score Breakdown:\n');
        for c = 1:numClasses
            fprintf('       [%s] Average Score: 0.0000 (Skill Rejected)\n', skillClasses{c});
        end
        fprintf('   - Final Decision Result: This is a new skill.\n');
    else
        fprintf('   - Closest Template Match: %s (Score: %.4f)\n', globalBestMatchFile, globalMaxScore);
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
    end
end % End of Outer Sample Iteration Loop
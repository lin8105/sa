%% ========================================================================
%  🤖 FRANKA 机器人：多对多 FSIM 特征相似度批量技能检索识别系统 (安全拦截版)
%  ========================================================================
clear; clc; close all;

%% 1. 环境路径定义
basePath = '/home/yue/Documents/zsc_Franka/data/lib'; 
templateDir = fullfile(basePath, 'templates');       % 存放已知参考图片的文件夹
testSampleDir = fullfile(basePath, 'test_samples');  % 存放多个未知测试样本的文件夹

templateFiles = dir(fullfile(templateDir, '*.png'));
testFiles = dir(fullfile(testSampleDir, '*.png'));

if isempty(templateFiles) || isempty(testFiles)
    error('请检查 templates/ 或 test_samples/ 文件夹中是否存在 PNG 图片！');
end

% 设定置信度判断阈值（最高大类平均分与次高大类平均分的差值）
CONFIDENCE_THRESHOLD = 0.12; 

% =========================================================================
% 🛡️ 【开放集绝对安检门】：FSIM 绝对相似度硬拦截线
% 因为 FSIM 会强行屏蔽全图大面积的纯绿静止底噪，逼迫系统只死磕真正的动作骨架。
% 如果一个未知的新技能丢进来，由于力学相位和梯度完全对不上，最高分也会很低。
% 只要当前测试样本和库里【最像的那张图】对比，FSIM 得分都低于 0.62，直接安全判定为新技能！
% =========================================================================
REJECTION_THRESHOLD = 0.62;  

fprintf('==================================================\n');
fprintf('     🔥 启动多对多批量技能 [FSIM 特征相似度] 检索系统 🔥     \n');
fprintf('  已知参考技能数: %d  |  待识别未知样本数: %d  \n', length(templateFiles), length(testFiles));
fprintf('==================================================\n\n');

%% 2. 【外层循环】逐个遍历每一个未知的测试样本
for t = 1:length(testFiles)
    currentTestName = testFiles(t).name;
    currentTestPath = fullfile(testFiles(t).folder, currentTestName);
    
    fprintf('▶️ [正在分析未知样本 %d/%d]: %s\n', t, length(testFiles), currentTestName);
    
    % 载入图像（读取原始图，fsim 内部会保证双精度转换）
    imgTest = imread(currentTestPath);
    
    % 使用 Map 结构动态自动按大类归纳得分
    skillMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    globalMaxScore = -1;
    globalBestMatchFile = '';
    
    %% 3. 【内层循环】让当前这个未知样本去和库里所有参考图片比对
    for i = 1:length(templateFiles)
        refPath = fullfile(templateFiles(i).folder, templateFiles(i).name);
        imgRef = imread(refPath);
        
        % 尺度严格对齐：fsim 函数要求两张对齐矩阵尺寸完全一致
        if any(size(imgRef) ~= size(imgTest))
            imgRef = imresize(imgRef, [size(imgTest, 1), size(imgTest, 2)]);
        end
        
        % -----------------------------------------------------------------
        % 🧠 FSIM 核心计算模块：提取相位一致性与梯度特征进行骨架比对
        % -----------------------------------------------------------------
        % [currentScore, ~] = FSIM(imageRef, imageDis)
        % 它返回 0-1 之间的标量分值，越接近 1 说明受力/速度变化的骨架越重合
        [currentScore, ~] = FSIM(imgTest, imgRef); 
        % -----------------------------------------------------------------
        
        % 正则表达式自动解析文件名，剥离数字尾缀提取技能大类标签
        [~, fileName, ~] = fileparts(templateFiles(i).name);
        tokens = regexp(fileName, '^([a-zA-Z_]+?)(?=\d*$|_?\d*$)', 'tokens');
        if ~isempty(tokens)
            skillClass = tokens{1}{1};
            if endsWith(skillClass, '_'), skillClass = skillClass(1:end-1); end
        else
            skillClass = fileName;
        end
        
        % 归类存放得分
        if isKey(skillMap, skillClass)
            skillMap(skillClass) = [skillMap(skillClass), currentScore];
        else
            skillMap(skillClass) = currentScore;
        end
        
        % 记录单一最优解
        if currentScore > globalMaxScore
            globalMaxScore = currentScore;
            globalBestMatchFile = fileName;
        end
    end % 💡 内层模板比对循环结束
    
    %% 4. 计算当前样本针对各个大类的统计指标并做出决策
    skillClasses = keys(skillMap);
    numClasses = length(skillClasses);
    avgScores = zeros(1, numClasses);
    
    for c = 1:numClasses
        avgScores(c) = mean(skillMap(skillClasses{c}));
    end
    
    [bestAvgScore, bestClassIdx] = max(avgScores);
    predictedSkillClass = skillClasses{bestClassIdx};
    
    % 计算区分度
    sortedAvgScores = sort(avgScores, 'descend');
    if length(sortedAvgScores) > 1
        confidenceGap = sortedAvgScores(1) - sortedAvgScores(2);
    else
        confidenceGap = sortedAvgScores(1);
    end
    
    %% 5. 判断是否触发新技能拒识（开放集识别决策）
    isUnknownSkill = false;
    if globalMaxScore < REJECTION_THRESHOLD
        isUnknownSkill = true;
    end
    
    %% 6. 打印当前样本的识别决策报告
    fprintf('   ├─ 最优对齐最高 FSIM 得分: %.4f (最佳匹配模板: %s)\n', globalMaxScore, globalBestMatchFile);
    
    if isUnknownSkill
        % 🚨 触发拦截：最高分低于及格线，判定为库里没有的新技能
        fprintf('   ├─ 各大类综合匹配得分列表:\n');
        for c = 1:numClasses
            fprintf('   │      【%s】平均得分: %.4f (已触发新技能拒识)\n', skillClasses{c}, avgScores(c));
        end
        fprintf('   │\n');
        fprintf('   └─ 综合判定结果: ❌ 这是一个【未知新技能（Unknown_Skill）】，已成功拦截！\n');
    else
        % ✅ 正常识别分支
        fprintf('   ├─ 各大类综合匹配得分列表:\n');
        for c = 1:numClasses
            currentClass = skillClasses{c};
            currentAvg = avgScores(c);
            if strcmp(currentClass, predictedSkillClass)
                fprintf('   │   ⭐ 【%s】平均 FSIM: %.4f (最高置信度)\n', currentClass, currentAvg);
            else
                fprintf('   │      【%s】平均 FSIM: %.4f\n', currentClass, currentAvg);
            end
        end
        
        fprintf('   │\n');
        fprintf('   └─ 综合判定结果: 预测归属于 【%s】 技能类别。\n', predictedSkillClass);
        
        if confidenceGap < CONFIDENCE_THRESHOLD
            fprintf('      ⚠️  [决策警报]: 分差(%.4f)小于阈值(%.4f)，不同动作物理骨架存在模糊混淆！\n', confidenceGap, CONFIDENCE_THRESHOLD);
        else
            fprintf('      ✅ [决策成功]: FSIM 相位边缘特征成功拉开清晰分类区间。\n');
        end
    end
    fprintf('--------------------------------------------------\n');
end

fprintf('\n🎉 基于 FSIM 特征相似度的批量交叉比对与新技能拦截系统运行完毕！\n');
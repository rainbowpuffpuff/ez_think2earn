%% MA
% This file is for nirs-data classification
% Most of MATLAB functions are available in BBCI toolbox
% Some minor code modifications might be applied
% We do not guarantee all of functions works properly in your platform
% If you want to see more tutorials, visit BBCI toolbox (https://github.com/bbci/bbci_public)
% Modified by Zenghui Wang, November 8, 2023.
%% MA
% This file is for NIRS-data classification.
% Most MATLAB functions are available in the BBCI toolbox.
% Some minor code modifications might be necessary.
% We do not guarantee that all functions will work properly on your platform.
% For more tutorials, visit the BBCI toolbox (https://github.com/bbci/bbci_public).
% Modified by Zenghui Wang, November 8, 2023.

% specify your NIRS data directory (NirsMyDataDir), temporary directory (TemDir), working directory (WorkingDir), and preprocessed data (MySaveDir)
WorkingDir = '/home/q/Downloads/MATLAB_v2/bin'; % Set your working directory
MyToolboxDir = fullfile(WorkingDir, '../bbci_public-master'); % Relative path to the BBCI toolbox
NirsMyDataDir = fullfile(WorkingDir, '../NIRS'); % Relative path to the NIRS data directory
TemDir = fullfile(WorkingDir, 'temp'); % Temporary directory
MySaveDir = fullfile(WorkingDir, 'MA_fNIRS_data'); % Directory for saving preprocessed data

% Add the BBCI toolbox to MATLAB's path and initialize
cd(MyToolboxDir);
startup_bbci_toolbox('DataDir', NirsMyDataDir, 'TmpDir', TemDir, 'History', 0);

% Loop through the subjects (1 to 29)
for idx = 1:29
    filename = num2str(idx);
    if idx <= 9
        file = strcat('subject 0', filename);
    else
        file = strcat('subject ', filename);
    end
    disp(file);

    % Subject directory and stimulus definition
    subdir_list = {file};
    stimDef.nirs = {1, 2; 'condition1', 'condition2'};

    % Load NIRS data
    loadDir = fullfile(NirsMyDataDir, subdir_list{1});
    cd(loadDir);
    load cnt; load mrk; load mnt; % Load continuous signal (cnt), marker (mrk), and montage (mnt)
    cd(WorkingDir);

    % Merge continuous signals (cnt) in each session, for mental arithmetic (ment)
    cnt_temp = cnt;
    mrk_temp = mrk;
    clear cnt mrk;

    [cnt.ment, mrk.ment] = proc_appendCnt({cnt_temp{2}, cnt_temp{4}, cnt_temp{6}}, {mrk_temp{2}, mrk_temp{4}, mrk_temp{6}}); % Merged mental arithmetic cnts

    % Apply the modified Beer-Lambert law (MBLL)
    cnt.ment = proc_BeerLambert(cnt.ment);

    % Band-pass filtering
    [b, a] = butter(3, [0.01 0.1] / cnt.ment.fs * 2);
    cnt.ment = proc_filtfilt(cnt.ment, b, a);

    % Divide into HbR (deoxyhemoglobin) and HbO (oxyhemoglobin)
    cntHb.ment.oxy = cnt.ment;
    cntHb.ment.deoxy = cnt.ment;

    % Replace data for HbO and HbR
    cntHb.ment.oxy.x = cnt.ment.x(:, 1:end/2);
    cntHb.ment.oxy.clab = cnt.ment.clab(:, 1:end/2);
    cntHb.ment.oxy.clab = strrep(cntHb.ment.oxy.clab, 'oxy', ''); % Remove 'oxy' from clab
    cntHb.ment.oxy.signal = 'NIRS (oxy)';

    cntHb.ment.deoxy.x = cnt.ment.x(:, end/2+1:end);
    cntHb.ment.deoxy.clab = cnt.ment.clab(:, end/2+1:end);
    cntHb.ment.deoxy.clab = strrep(cntHb.ment.deoxy.clab, 'deoxy', ''); % Remove 'deoxy' from clab
    cntHb.ment.deoxy.signal = 'NIRS (deoxy)';

    % Epoching
    ival_epo = [-10 25] * 1000; % Epoch from -10s to 25s relative to task onset
    epo.ment.oxy = proc_segmentation(cntHb.ment.oxy, mrk.ment, ival_epo);
    epo.ment.deoxy = proc_segmentation(cntHb.ment.deoxy, mrk.ment, ival_epo);

    % Baseline correction
    ival_base = [-5 -2] * 1000; % Baseline from -5s to -2s
    epo.ment.oxy = proc_baseline(epo.ment.oxy, ival_base);
    epo.ment.deoxy = proc_baseline(epo.ment.deoxy, ival_base);

    % Moving time windows
    StepSize = 1 * 1000; % 1 second
    WindowSize = 3 * 1000; % 3 seconds
    ival_start = (ival_epo(1):StepSize:ival_epo(end)-WindowSize)';
    ival_end = ival_start + WindowSize;
    ival = [ival_start, ival_end];
    nStep = length(ival);

    for stepIdx = 1:nStep
        segment.ment.deoxy{stepIdx} = proc_selectIval(epo.ment.deoxy, ival(stepIdx, :));
        segment.ment.oxy{stepIdx} = proc_selectIval(epo.ment.oxy, ival(stepIdx, :));
    end

    % Save fNIRS data
    num = num2str(idx);
    mkdir(fullfile(MySaveDir, num));
    for stepIdx = 1:nStep
        path = fullfile(MySaveDir, num, strcat(num2str(stepIdx), '_deoxy.mat'));
        signal = segment.ment.deoxy{stepIdx}.x;
        save(path, 'signal');

        path = fullfile(MySaveDir, num, strcat(num2str(stepIdx), '_oxy.mat'));
        signal = segment.ment.oxy{stepIdx}.x;
        save(path, 'signal');
    end
    path = fullfile(MySaveDir, num, strcat(num, '_desc.mat'));
    label = segment.ment.deoxy{1}.event.desc;
    save(path, 'label');

    disp('MA data finished');
end

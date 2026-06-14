%% filter_design.m
%  Designs the 4th-order Butterworth bandpass filter used for ECG preprocessing.
%
%  KEY CHANGE FROM PREVIOUS VERSION:
%  The filter is now saved in TWO formats:
%
%  1. Direct-form b,a  — used by MATLAB filtfilt() in preprocessing.m (unchanged)
%
%  2. Second-Order Sections (SOS) — used by the FPGA fixed-point implementation
%     WHY: Direct-form a coefficients reach values like 26.3.
%          At Q8.8 (x256)  -> 26.3 x 256 = 6733  (huge, but still fits int16)
%          BUT b max = 0.0066 x 256 = 1.69 -> rounds to 2 (93% error) -> zero output
%          At Q16.16 (x65536) -> a coeff = 26.3 x 65536 = 1,724,000
%          Multiplied by signal -> instant int32 overflow -> +-32768 garbage
%
%          SOS splits the 8th-order polynomial into 4 cascaded 2nd-order stages.
%          Each stage has max |a| ~ 1.9, max |b| ~ 0.02
%          At Q16.16: max a = 1.9 x 65536 = 124,518  (safe)
%                     max b = 0.02 x 65536 = 1310     (preserved, not zeroed)
%          No overflow, coefficients preserved.
%
%  Run this script FIRST before preprocessing.m

clear; clc;

%% ── STEP 1: Set project directory ────────────────────────────────────────
project_dir   = 'C:\Users\User\OneDrive\Desktop\ecg_fpga_classifier\ecg_fpga_classifier\output';
processed_dir = fullfile(project_dir, 'matlab', 'data', 'processed');

fprintf('Project folder : %s\n', project_dir);
fprintf('Processed dir  : %s\n\n', processed_dir);

%% ── STEP 2: Create all required folders ──────────────────────────────────
folders_needed = {
    fullfile(project_dir, 'matlab', 'data'),
    fullfile(project_dir, 'matlab', 'data', 'raw'),
    fullfile(project_dir, 'matlab', 'data', 'processed'),
    fullfile(project_dir, 'matlab', 'data', 'splits'),
    fullfile(project_dir, 'weights'),
    fullfile(project_dir, 'results'),
};
for k = 1:numel(folders_needed)
    if ~exist(folders_needed{k}, 'dir')
        mkdir(folders_needed{k});
        fprintf('Created: %s\n', folders_needed{k});
    end
end
fprintf('All folders ready.\n\n');

%% ── STEP 3: Filter parameters ────────────────────────────────────────────
fs      = 360;   % MIT-BIH sampling rate (Hz)
lowcut  = 0.5;   % Low cutoff  - removes baseline wander
highcut = 45;    % High cutoff - removes muscle noise + 60 Hz hum
order   = 4;     % Filter order (4th-order bandpass = 8th-order polynomial)

%% ── STEP 4: Design filter — both direct-form and SOS ─────────────────────
nyquist = fs / 2;
Wn      = [lowcut/nyquist, highcut/nyquist];

% Direct-form: used ONLY by filtfilt() in preprocessing.m
[b, a] = butter(order, Wn, 'bandpass');

% SOS form: used by FPGA fixed-point IIR
% Each row of sos = [b0 b1 b2 1 a1 a2] for one 2nd-order section
% g = overall gain factor applied once at the input
[sos, g] = tf2sos(b, a);
n_sections = size(sos, 1);

fprintf('=== Direct-Form Coefficients (for filtfilt) ===\n');
fprintf('b = '); fprintf('%.8e  ', b); fprintf('\n');
fprintf('a = '); fprintf('%.8f  ', a); fprintf('\n\n');

fprintf('=== SOS Coefficients (for FPGA) — %d sections ===\n', n_sections);
fprintf('Overall gain g = %.10f\n\n', g);
for s = 1:n_sections
    fprintf('Section %d:  b = [%.8f  %.8f  %.8f]   a = [1  %.8f  %.8f]\n', ...
            s, sos(s,1), sos(s,2), sos(s,3), sos(s,5), sos(s,6));
end

% Show coefficient ranges — this is WHY SOS works for fixed-point
fprintf('\n=== Coefficient Range Check ===\n');
fprintf('Direct-form:  max|b| = %.6f   max|a| = %.6f\n', max(abs(b)), max(abs(a)));
fprintf('SOS sections: max|b| = %.6f   max|a| = %.6f\n', ...
        max(max(abs(sos(:,1:3)))), max(max(abs(sos(:,5:6)))));
fprintf('\nAt Q16.16 (x65536):\n');
fprintf('  Direct a_max -> %.0f  (overflow risk with signal multiply)\n', max(abs(a))*65536);
fprintf('  SOS     a_max -> %.0f  (safe for int32 multiply)\n', max(max(abs(sos(:,5:6))))*65536);

%% ── STEP 5: Fixed-point SOS coefficients (Q16.16) ────────────────────────
SCALE     = 2^16;   % Q16.16
sos_fixed = round(sos * SCALE);
g_fixed   = round(g * SCALE);

fprintf('\n=== SOS Fixed-Point (Q16.16 = x65536) ===\n');
for s = 1:n_sections
    fprintf('Section %d:  B=[%d  %d  %d]   A=[%d  %d  %d]\n', s, ...
            sos_fixed(s,1), sos_fixed(s,2), sos_fixed(s,3), ...
            sos_fixed(s,4), sos_fixed(s,5), sos_fixed(s,6));
end
fprintf('Gain: %d\n', g_fixed);

%% ── STEP 6: Save FPGA coefficient text file ──────────────────────────────
fpga_file = fullfile(processed_dir, 'butterworth_fpga_coeffs.txt');
fid = fopen(fpga_file, 'w');
if fid == -1
    error('Cannot write to: %s', fpga_file);
end

fprintf(fid, '=== Butterworth Bandpass Filter Coefficients for FPGA ===\n');
fprintf(fid, 'Design: %d-order bandpass, %.1f-%.1f Hz, Fs=%d Hz\n\n', order, lowcut, highcut, fs);
fprintf(fid, 'Implementation: Cascaded Second-Order Sections (SOS)\n');
fprintf(fid, 'Format: Q16.16 fixed-point (float x 65536, rounded to integer)\n\n');

fprintf(fid, '--- Floating-point SOS ---\n');
fprintf(fid, 'Overall gain g = %.10f\n\n', g);
for s = 1:n_sections
    fprintf(fid, 'Section %d:  b=[%.8f  %.8f  %.8f]   a=[1  %.8f  %.8f]\n', ...
            s, sos(s,1), sos(s,2), sos(s,3), sos(s,5), sos(s,6));
end

fprintf(fid, '\n--- Fixed-point SOS (Q16.16) ---\n');
fprintf(fid, 'Gain: %d\n\n', g_fixed);
for s = 1:n_sections
    fprintf(fid, 'Section %d:\n', s);
    fprintf(fid, '  B0=%d  B1=%d  B2=%d\n',   sos_fixed(s,1), sos_fixed(s,2), sos_fixed(s,3));
    fprintf(fid, '  A0=%d  A1=%d  A2=%d\n\n', sos_fixed(s,4), sos_fixed(s,5), sos_fixed(s,6));
end

% Also keep direct-form for reference
fprintf(fid, '\n--- Direct-form (reference only, DO NOT use for FPGA) ---\n');
fprintf(fid, 'b: '); fprintf(fid, '%.8e  ', b); fprintf(fid, '\n');
fprintf(fid, 'a: '); fprintf(fid, '%.8f  ', a); fprintf(fid, '\n');

fclose(fid);
fprintf('\nSaved: %s\n', fpga_file);

%% ── STEP 7: Save .mat file ────────────────────────────────────────────────
%  Saves BOTH forms so preprocessing.m (filtfilt) and verify_binary_filter.m
%  (SOS fixed-point) can each load what they need.
mat_file = fullfile(processed_dir, 'filter_coeffs.mat');
save(mat_file, 'b', 'a', 'sos', 'g', 'sos_fixed', 'g_fixed', ...
     'fs', 'lowcut', 'highcut', 'order', 'SCALE');
fprintf('Saved: %s\n', mat_file);

%% ── STEP 8: Verify SOS matches direct-form (sanity check) ────────────────
fprintf('\n=== Sanity Check: SOS vs direct-form frequency response ===\n');
[H_direct, f] = freqz(b, a, 4096, fs);

% Reconstruct frequency response from SOS manually
H_sos = ones(4096, 1);
for s = 1:n_sections
    [H_s, ~] = freqz(sos(s,1:3), sos(s,4:6), 4096, fs);
    H_sos = H_sos .* H_s;
end
H_sos = H_sos * g;

max_diff = max(abs(abs(H_direct) - abs(H_sos)));
fprintf('Max magnitude difference between SOS and direct-form: %.2e\n', max_diff);
if max_diff < 1e-6
    fprintf('Confirmed: SOS and direct-form are equivalent.\n');
end

%% ── STEP 9: Plot frequency response ──────────────────────────────────────
mag_dB  = 20 * log10(abs(H_direct) + eps);
phase_d = unwrap(angle(H_direct)) * 180 / pi;

figure('Color', 'black', 'Position', [100 100 860 580]);

subplot(2,1,1);
semilogx(f, mag_dB, 'b-', 'LineWidth', 2);
hold on;
xline(lowcut,  'r--', 'LineWidth', 1.5, 'Label', '0.5 Hz');
xline(highcut, 'r--', 'LineWidth', 1.5, 'Label', '45 Hz');
yline(0,  'k:', 'LineWidth', 1);
yline(-3, 'g--', 'LineWidth', 1.2, 'Label', '-3 dB');
patch([lowcut lowcut highcut highcut], [-90 5 5 -90], ...
      [0 0.6 0], 'FaceAlpha', 0.06, 'EdgeColor', 'none');
hold off;
xlim([0.1 180]); ylim([-90 5]);
xlabel('Frequency (Hz) - log scale', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontSize', 11);
title('4th-Order Butterworth Bandpass (0.5-45 Hz) - SOS Implementation', ...
      'FontSize', 13, 'FontWeight', 'bold');
legend({'Magnitude response','0.5 Hz cutoff','45 Hz cutoff','0 dB','-3 dB'}, ...
       'Location', 'southwest', 'FontSize', 10);
text(1,   -15, 'Passband', 'Color', [0 0.5 0], 'FontSize', 11, 'FontWeight', 'bold');
text(80,  -15, 'Stopband', 'Color', [0.8 0 0], 'FontSize', 11, 'FontWeight', 'bold');
text(0.13,-15, 'Stopband', 'Color', [0.8 0 0], 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;
set(gca, 'FontSize', 10, 'Color', 'white', 'GridColor', [0.8 0.8 0.8], 'XMinorGrid', 'on');

subplot(2,1,2);
semilogx(f, phase_d, 'b-', 'LineWidth', 2);
hold on;
xline(lowcut,  'r--', 'LineWidth', 1.5);
xline(highcut, 'r--', 'LineWidth', 1.5);
hold off;
xlim([0.1 180]);
xlabel('Frequency (Hz) - log scale', 'FontSize', 11);
ylabel('Phase (degrees)', 'FontSize', 11);
title('Phase Response (zero-phase achieved via filtfilt)', 'FontSize', 13, 'FontWeight', 'bold');
grid on; box on;
set(gca, 'FontSize', 10, 'Color', 'white', 'GridColor', [0.8 0.8 0.8], 'XMinorGrid', 'on');

%% ── STEP 10: Add scaled coefficients for direct-form verification ─────────
% After your existing code, add this section

fprintf('\n=== Generating scaled coefficients for direct-form verification ===\n');

% Scale coefficients for Q16.16
COEFF_SCALE = 65536;
b_scaled = round(b * COEFF_SCALE);
a_scaled = round(a * COEFF_SCALE);

% Normalize so a(1) = COEFF_SCALE
if a_scaled(1) ~= COEFF_SCALE
    norm_factor = COEFF_SCALE / a_scaled(1);
    b_scaled = round(b_scaled * norm_factor);
    a_scaled = round(a_scaled * norm_factor);
    fprintf('  Normalized coefficients (a1 was %d)\n', a_scaled(1));
end

fprintf('  b_scaled: '); fprintf('%d ', b_scaled); fprintf('\n');
fprintf('  a_scaled: '); fprintf('%d ', a_scaled); fprintf('\n');

% Save scaled coefficients for verification
save(fullfile(processed_dir, 'filter_coeffs_scaled.mat'), ...
     'b_scaled', 'a_scaled', 'COEFF_SCALE');

fprintf('  Saved scaled coefficients to filter_coeffs_scaled.mat\n');

fprintf('\nfilter_design.m complete.\n');

%% --- STEP 1: EXPORT FOR VERILOG ---

% 1. Print the exact parameters for Verilog Copy-Paste
fprintf('\n--- COPY THESE INTO YOUR VERILOG TOP MODULE ---\n');
fprintf('Gain (g_fixed): %d\n', g_fixed);
for s = 1:n_sections
    fprintf('Section %d: B0=%d, B1=%d, B2=%d, A1=%d, A2=%d\n', ...
            s, sos_fixed(s,1), sos_fixed(s,2), sos_fixed(s,3), ...
            sos_fixed(s,5), sos_fixed(s,6));
end

% 2. Create a Dummy Input Signal (100 samples of a 10Hz sine wave)
% This helps us verify the filter works before using real ECG data.
test_fs = 360;
t = (0:99)'/test_fs;
test_input = sin(2*pi*10*t) * 1000; % Scale up to see movement in fixed-point
test_input_int = round(test_input);

% 3. Save to a text file for the Verilog Testbench
% 'hex' is easier for Verilog to read accurately
fid = fopen('input_stimulus.txt', 'w');
for i = 1:length(test_input_int)
    % Convert to 32-bit hex string
    fprintf(fid, '%08x\n', typecast(int32(test_input_int(i)), 'uint32'));
end
fclose(fid);

fprintf('\nDone! "input_stimulus.txt" created. Copy the coefficients above.\n');

%% --- STEP 10: FINAL HARDWARE VERIFICATION ---
% Load the Verilog results
verilog_out_file = 'C:/ecg_fpga_classifier/ecg_fpga_classifier/matlab/filtered_output.txt';
if exist(verilog_out_file, 'file')
    verilog_data = readmatrix(verilog_out_file);
    
    % Plot comparison
    figure('Color', 'white');
    plot(test_input_int, 'k:', 'LineWidth', 1); hold on;
    plot(verilog_data, 'r', 'LineWidth', 2);
    grid on;
    title('Final Verification: Verilog Hardware vs. Input');
    legend('Input Stimulus', 'Verilog Filtered Output');
    xlabel('Sample Number');
    ylabel('Amplitude (Q16.16)');
    
    fprintf('Verification complete. Check the plot to see the filter in action!\n');
else
    error('filtered_output.txt not found! Check your Verilog testbench path.');
end


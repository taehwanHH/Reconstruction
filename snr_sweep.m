clear; close all; clc;

all_data = table();
scheme_num = 2;
sim_num=12;
while true
    sim_idx = sprintf('%03d', sim_num);  
    % filename = fullfile('sim_result', sprintf('reconstruction_metrics_%s.csv', sim_idx));
    filename = fullfile('data/main_csv/snr_sweep', sprintf('Scheme%d/Scheme%d_results_%s.csv', scheme_num,scheme_num, sim_idx));
    
    if ~isfile(filename)
        sim_num = sim_num-1;
        fprintf("Calculating average value for total %d sim\n", sim_num);
        break;
    end
    
    data = readtable(filename);  % 각 CSV 파일은 num_samples, PSNR, SSIM, MSE 등의 컬럼을 가짐
    all_data = [all_data; data]; % 테이블에 추가

    sim_num = sim_num + 1;
end

avg_table = groupsummary(all_data, 'snr', 'mean', {'outlier_count'});
snr = avg_table.snr(1:5);
metric =(100000-avg_table.mean_outlier_count(1:5))/100000*100;

% figure(1);
% plot(num_sample,PSNR,'-ro', 'LineWidth', 2);
% xlabel('Total sensing, $N$','Interpreter','latex')
% ylabel('Completeness','Interpreter','latex')
% xlim([200,1000]);
% grid on;
figure;
yyaxis left
plot(snr,metric,'-o', 'LineWidth', 2);
ylabel('Reconstruction accuracy (\%)','Interpreter','latex','FontSize',12)
ylim([0,100])
yticks(0:10:100)

avg_table = groupsummary(all_data, 'snr', 'mean', {'stiffness_accuracy'});
snr = avg_table.snr(1:5);
metric = avg_table.mean_stiffness_accuracy(1:5)*100;


yyaxis right
plot(snr,metric,'-o', 'LineWidth', 2);
ylabel('Classification accuracy (\%)','Interpreter','latex','FontSize',12)
xlabel('SNR (dB)','Interpreter','latex','FontSize',12)
title('Rayleigh fading ($N$=500)','Interpreter','latex','FontSize',12)
xticks(0:5:20);
ylim([0,100])
yticks(0:10:100)

ax = gca;
lineWidth=1.3;
ax.YAxis(1).LineWidth = lineWidth;  % left spine
ax.YAxis(2).LineWidth = lineWidth;  % right spine
ax.XAxis(1).LineWidth = lineWidth;



grid on;
% clear; close all; clc;

all_data = table();
scheme_num = 2;
sim_num=16;
while true
    sim_idx = sprintf('%03d', sim_num);  
    filename = fullfile('data/main_csv/smp_sweep', sprintf('Scheme%d/Scheme%d_results_%s.csv', scheme_num,scheme_num, sim_idx));
    
    if ~isfile(filename)
        sim_num = sim_num-1;
        fprintf("Calculating average value for total %d sim\n", sim_num);
        break;
    end
    
    data = readtable(filename);  % 각 CSV 파일은 num_samples, PSNR, SSIM, MSE 등의 컬럼을 가짐
    all_data = [all_data; data]; % 테이블에 추가

    sim_num = sim_num + 1;
end

avg_table = groupsummary(all_data, 'num_samples', 'mean', {'outlier_count'});
num_samples = avg_table.num_samples;
metric =(100000-avg_table.mean_outlier_count)/100000*100;

% figure(1);
% plot(num_sample,PSNR,'-ro', 'LineWidth', 2);
% xlabel('Total sensing, $N$','Interpreter','latex')
% ylabel('Completeness','Interpreter','latex')
% xlim([200,1000]);
% grid on;
figure;
yyaxis left
plot(num_samples,metric,'-o', 'LineWidth', 2);
ylabel('Reconstruction accuracy (\%)','Interpreter','latex','FontSize',12)
ylim([0,100])

yticks(0:10:100)

% xlabel('Num samples, $N$','Interpreter','latex')

% title('AWGN','Interpreter','latex')
% xticks(100:100:500)
% grid on;


avg_table = groupsummary(all_data, 'num_samples', 'mean', {'stiffness_accuracy'});
num_samples = avg_table.num_samples;
metric =avg_table.mean_stiffness_accuracy*100;

yyaxis right
plot(num_samples,metric,'-o', 'LineWidth', 2);
xlabel('Num samples, $N$','Interpreter','latex','FontSize',12)
ylabel('Classification accuracy (\%)','Interpreter','latex','FontSize',12)
title('Rayleigh fading (SNR=20dB)','Interpreter','latex','FontSize',12)
xticks(100:100:500)
ylim([0,100])
yticks(0:10:100)

ax = gca;
lineWidth=1.3;
ax.YAxis(1).LineWidth = lineWidth;  % left spine
ax.YAxis(2).LineWidth = lineWidth;  % right spine
ax.XAxis(1).LineWidth = lineWidth;

grid on;

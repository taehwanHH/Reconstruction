clear; close all; clc;

all_data = table();
sim_num=1;
while true
    sim_idx = sprintf('%03d', sim_num);  
    filename = fullfile('sim_result', sprintf('reconstruction_metrics_%s.csv', sim_idx));
    
    if ~isfile(filename)
        sim_num = sim_num-1;
        fprintf("Calculating average value for total %d sim\n", sim_num);
        break;
    end
    
    data = readtable(filename);  % 각 CSV 파일은 num_samples, PSNR, SSIM, MSE 등의 컬럼을 가짐
    all_data = [all_data; data]; % 테이블에 추가

    sim_num = sim_num + 1;
end

avg_table = groupsummary(all_data, 'num_samples', 'mean', {'PSNR','SSIM','MSE'});
num_sample = avg_table.num_samples;
PSNR = avg_table.mean_PSNR;
SSIM =avg_table.mean_SSIM;
MSE = avg_table.mean_MSE;

subplot(1,3,1);
plot(num_sample,PSNR,'-o', 'LineWidth', 2);
xlim([100,3000]);
subplot(1,3,2);
plot(num_sample,SSIM,'-o', 'LineWidth', 2);
xlim([100,3000]);
subplot(1,3,3);
plot(num_sample,MSE,'-o', 'LineWidth', 2);
xlim([100,3000]);

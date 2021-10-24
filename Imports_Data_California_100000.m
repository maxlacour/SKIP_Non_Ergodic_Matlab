%% Imports Data from California from Nico n = 2000 points!!

[Example_VCM_Table] = readtable('CS15p4_600_events.csv'); % Is a table! So get rid of headers
opts = detectImportOptions('CS15p4_600_events.csv'); %opts.VariableNames gives headers names

num_rows = size(Example_VCM_Table, 1);
num_columns = size(Example_VCM_Table, 2);

Example_VCM = zeros(num_rows, num_columns);
header_names_Example_VCM = cell(1, num_columns);


for i = 1:num_columns
    header_names_Example_VCM{i} = opts.VariableNames(i);
    header_name = header_names_Example_VCM{i};
    eval(strcat('Example_VCM(:, i) = Example_VCM_Table.', header_name{1}, ';'));
end

Region_vector = Example_VCM(:, 1);
Eq_ID_vector = Example_VCM(:, 2);

eqLongitude_vector = Example_VCM(:, 3);
eqLatitude_vector = Example_VCM(:, 4);

staLongitude_vector = Example_VCM(:, 5);
staLatitude_vector = Example_VCM(:, 6);

RJB_vector = Example_VCM(:, 7);
Mag_vector = Example_VCM(:, 8);

log_PGA_vector = Example_VCM(:, 10); %3s!!


VS30_vector = Example_VCM(:, 15);
SOF_vector = Example_VCM(:, 6);




Mag_vector_2 = Mag_vector.^2;

h = 6;
log_RJB_vector = log(sqrt(RJB_vector.^2 + h^2));


n = length(log_PGA_vector);


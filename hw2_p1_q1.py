import csv
import pandas
import numpy

import matplotlib.pyplot as plt

## path for logs
plot_path = "./logs/"

## read log.txt
data = []
with open(plot_path + 'tpbench_log.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=" ")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            index = row
            line_count += 1
        else:
            line = row[0].split("\t")
            line.pop(len(line)-1)
            data.append(line)
            line_count += 1

## create data structures
df = pandas.DataFrame(data,columns = index)
print(df)
freq = 0.2
time = numpy.linspace(freq,freq*len(df),num=len(df))

#path for plots
plot_path = "./plots/question1/"

## Power Consumption Plot
plt.figure("W_bench")
plt.plot(time,df["W"].astype(float), label='Power Consumption')
plt.xlabel('time s')
plt.ylabel('Power W')
plt.legend()
plt.grid(True)
plt.title('Total Power Consumption of Mc1')
plt.savefig(plot_path + 'power_hw2_q1.png')

for i in range(4):
    col = "usage_c" + str(i+4)
    plt.figure(col)
    plt.plot(time,df[col], label='Core utilization')
    plt.xlabel('time s')
    plt.ylabel('utilization %')
    plt.legend()
    plt.grid(True)
    plt.title(col)
    file_name = plot_path + col + "_hw2_q1.png"
    plt.savefig(file_name)


for i in range(4):
    col = "temp" + str(i+4)
    plt.figure(col)
    plt.plot(time, df[col], label='Core temperature')
    plt.xlabel('time s')
    plt.ylabel('Temp ºC')
    plt.legend()
    plt.grid(True)
    plt.title(col)
    file_name = plot_path + col + "_hw2_q1.png"
    plt.savefig(file_name)

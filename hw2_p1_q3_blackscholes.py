import csv
import pandas
import numpy

import matplotlib.pyplot as plt

## path for logs
plot_path = "./logs/"

## read log.txt
data_blackscholes = []
with open(plot_path + 'blackscholes_log.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=" ")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            index = row
            line_count += 1
        else:
            line = row[0].split("\t")
            line.pop(len(line)-1)
            data_blackscholes.append(line)
            line_count += 1

## create data structures
df_blackscholes = pandas.DataFrame(data_blackscholes,columns = index)
print(df_blackscholes)
freq = 0.2
time_bodytrack = numpy.linspace(freq,freq*len(df_blackscholes),num=len(df_blackscholes))

#path for plots
plot_path = "./plots/question1/"

## Power Consumption Plot
plt.figure("W")
plt.plot(time_bodytrack,df_blackscholes["W"], label='Power Consumption')
plt.xlabel('time s')
plt.ylabel('Total Power Consumption')
plt.legend()
plt.grid(True)
plt.title('Total Power Consumption of Mc1')
plt.savefig(plot_path + 'power_hw2_q3_blackscholes.png')


## Max temperature
df_temp = df_blackscholes[['temp4', 'temp5', 'temp6', 'temp7']]
print(df_temp.max())
plt.figure("T")
plt.plot(time_bodytrack, df_blackscholes['temp6'], label='Core max temperature')
plt.xlabel('time s')
plt.ylabel('Temp ºC')
plt.legend()
plt.grid(True)
plt.title('Max Temperature')
file_name = plot_path + 'max_temp' + "_hw2_q3_blackscholes.png"
plt.savefig(file_name)

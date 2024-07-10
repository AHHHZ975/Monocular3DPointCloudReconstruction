import matplotlib.pyplot as plt
import numpy as np
from Config import ROOT_DIR


Data_Categories =       ['Airplane',    'Bottle',   'Car',      'Sofa',     'Bench',    'Cellphone',    'Bike',     'Speaker',    'Mean']
TransCNN3D = {
    'Loss' :            [0.439,         1.255,       0.945,     1.725,       1.368,      1.217,         1.723,       3.410,        1.517],
    'Inference Time' :  [1.157,         4.827,       1.297,     1.355,       1.624,      2.466,         4.287,       1.489,        2.313],
    'Training Time' :   [9.804,         4.581,       9.300,     9.810,       6.433,      3.674,         2.966,       5.445,        6.502],
}
FI2P = {
    'Loss' :            [0.450,         1.354,       0.975,     1.780,       1.438,      1.297,         1.765,       3.482,        1.568],
    'Inference Time' :  [0.302,         2.846,       0.416,     0.453,       0.634,      1.118,         2.473,       0.544,        1.098],
    'Training Time' :   [6.775,         1.297,       2.064,     1.836,       1.043,      0.594,         0.508,       1.321,        1.930],
}
PSGN = {
    'Loss' :            [0.465,         1.482,       0.998,     1.833,       2.170,      1.353,         1.989,       3.594,        1.722],
    'Inference Time' :  [0.801,         6.252,       1.061,     1.095,       1.465,      2.439,         5.088,       1.281,        2.435],
    'Training Time' :   [30.26,         3.010,       9.372,     6.825,       3.015,      4.653,         2.483,       8.320,        8.492],
}
Pixel2Point = {
    'Loss' :            [0.484,         1.495,       0.980,     1.776,       1.690,      1.368,         1.930,       3.790,        1.689],
    'Inference Time' :  [0.324,         2.923,       0.446,     0.487,       0.710,      1.152,         2.643,       0.576,        1.158],
    'Training Time' :   [7.088,         2.120,       5.739,     5.986,       3.531,      2.031,         1.356,       3.296,        3.893],
}


###################### Compare our models to others' ##################

# def showLossPlot():
#     plt.plot(Data_Categories, Ours_New['Loss'], label='Ours', linestyle= '--', linewidth=4.5)    
#     plt.plot(Data_Categories, Pixel2Point['Loss'], label='Pixel2Point (w/o PC)', linestyle=':', linewidth=4.5)
#     plt.plot(Data_Categories, Pixel2Point_InitialPC['Loss'], label='Pixel2Point (w PC)', linestyle=':', linewidth=4.5)
#     plt.plot(Data_Categories, PSGN['Loss'], label='PSGN', linestyle='-.', linewidth=4.5)
#     plt.legend(prop={'size':20})
#     plt.xticks(fontsize=18, rotation=0)
#     plt.yticks(fontsize=18, rotation=0)
#     plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
#     plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize=20)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig(ROOT_DIR + '/PlotLossResult.png')    
#     plt.show()


# def showTimePlot():
#     plt.plot(Data_Categories, Ours_New['Time'], label='Ours', linestyle= '--', linewidth=4.5)    
#     plt.plot(Data_Categories, Pixel2Point['Time'], label='Pixel2Point (w/o PC)', linestyle=':', linewidth=4.5)
#     plt.plot(Data_Categories, Pixel2Point_InitialPC['Time'], label='Pixel2Point (w PC)', linestyle=':', linewidth=4.5)
#     plt.plot(Data_Categories, PSGN['Time'], label='PSGN', linestyle='-.', linewidth=4.5)
#     plt.legend(prop={'size':20})
#     plt.xticks(fontsize=18, rotation=0)
#     plt.yticks(fontsize=18, rotation=0)
#     plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
#     plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize=20)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig(ROOT_DIR + '/PlotTimeResult.png')

#     plt.show()


def showLossHistogram():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(TransCNN3D['Loss']))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, TransCNN3D['Loss'], color ='y', width = barWidth, label ='TransCNN3D', alpha=1)
    plt.bar(br2, FI2P['Loss'], color ='r', width = barWidth, label ='FI2P', alpha=1)
    plt.bar(br3, Pixel2Point['Loss'], color ='b', width = barWidth, label ='Pixel2Point', alpha=1)
    plt.bar(br4, PSGN['Loss'], color ='g', width = barWidth, label ='PSGN', alpha=1)


    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(FI2P['Loss']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(ROOT_DIR + '/HistogramLossResult.png')    
    plt.show()


def showInferenceTimeHistogram():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(TransCNN3D['Inference Time']))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, TransCNN3D['Inference Time'], color ='y', width = barWidth, label ='TransCNN3D', alpha=1)
    plt.bar(br2, FI2P['Inference Time'], color ='r', width = barWidth, label ='FI2P', alpha=1)
    plt.bar(br3, Pixel2Point['Inference Time'], color ='b', width = barWidth, label ='Pixel2Point', alpha=1)
    plt.bar(br4, PSGN['Inference Time'], color ='g', width = barWidth, label ='PSGN', alpha=1)

    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Inference Time (ms)', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(FI2P['Inference Time']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(ROOT_DIR + '/HistogramInferenceTimeResult.png')
    plt.show()


def showTraningTimeHistogram():
    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    br1 = np.arange(len(TransCNN3D['Training Time']))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, TransCNN3D['Training Time'], color ='y', width = barWidth, label ='TransCNN3D', alpha=1)
    plt.bar(br2, FI2P['Training Time'], color ='r', width = barWidth, label ='FI2P', alpha=1)
    plt.bar(br3, Pixel2Point['Training Time'], color ='b', width = barWidth, label ='Pixel2Point', alpha=1)
    plt.bar(br4, PSGN['Training Time'], color ='g', width = barWidth, label ='PSGN', alpha=1)

    # Adding Xticks
    plt.legend(prop={'size':20})
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
    plt.ylabel('Training Time (min)', fontweight ='bold', fontsize = 20)
    plt.xticks([r + 1.5*barWidth for r in range(len(FI2P['Training Time']))], Data_Categories)
    plt.grid(axis='y', linewidth=0.5)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(ROOT_DIR + '/HistogramTrainingTimeResult.png')    
    plt.show()





###################### Compare our models to each other ##################

# def showLossPlot_Ours():
#     plt.plot(Data_Categories, Ours_New['Loss'], label='Ours (w/o Pool)', linestyle= '--', linewidth=4.5)    
#     plt.plot(Data_Categories, ours['Loss'], label='Ours (w Pool)', linestyle= ':', linewidth=4.5)
#     plt.legend(prop={'size':20})
#     plt.xticks(fontsize=18, rotation=0)
#     plt.yticks(fontsize=18, rotation=0)
#     plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
#     plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize=20)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig(ROOT_DIR + '/PlotLossResult_Ours.png')

#     plt.show()

# def showTimePlot_Ours():
#     plt.plot(Data_Categories, Ours_New['Time'], label='Ours (w/o Pool)', linestyle= '--', linewidth=4.5)    
#     plt.plot(Data_Categories, ours['Time'], label='Ours (w Pool)', linestyle= ':', linewidth=4.5)
#     plt.legend(prop={'size':20})
#     plt.xticks(fontsize=18, rotation=0)
#     plt.yticks(fontsize=18, rotation=0)
#     plt.xlabel('Data Category', fontweight ='bold', fontsize=20)
#     plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize=20)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig(ROOT_DIR + '/PlotTimeResult_Ours.png')

#     plt.show()

# def showLossHistogram_Ours():
#     # set width of bar
#     barWidth = 0.15

#     # Set position of bar on X axis
#     br1 = np.arange(len(ours['Loss']))
#     br2 = [x + barWidth for x in br1]

#     # Make the plot
#     plt.bar(br1, Ours_New['Loss'], color ='r', width = barWidth, label ='Ours (w/o Pool)', alpha=0.7)
#     plt.bar(br2, ours['Loss'], color ='b', width = barWidth, label ='Ours (w Pool)', alpha=0.7)

#     # Adding Xticks
#     plt.legend(prop={'size':20})
#     plt.xticks(fontsize=18, rotation=0)
#     plt.yticks(fontsize=18, rotation=0)
#     plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
#     plt.ylabel('Chamfer Loss', fontweight ='bold', fontsize = 20)
#     plt.xticks([r + 1.5*barWidth for r in range(len(Ours_New['Loss']))], Data_Categories)
#     plt.grid(axis='y', linewidth=0.5)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig(ROOT_DIR + '/HistogramLossResult_Ours.png')

#     plt.show()

# def showTimeHistogram_Ours():
#     # set width of bar
#     barWidth = 0.15

#     # Set position of bar on X axis
#     br1 = np.arange(len(Ours_New['Time']))
#     br2 = [x + barWidth for x in br1]

#     # Make the plot
#     plt.bar(br1, Ours_New['Time'], color ='r', width = barWidth, label ='Ours (w/o Pool)', alpha=0.7)
#     plt.bar(br2, ours['Time'], color ='b', width = barWidth, label ='Ours (w Pool)', alpha=0.7)

#     # Adding Xticks
#     plt.legend(prop={'size':20})
#     plt.xticks(fontsize=18, rotation=0)
#     plt.yticks(fontsize=18, rotation=0)
#     plt.xlabel('Data Category', fontweight ='bold', fontsize = 20)
#     plt.ylabel('Computational Time (ms)', fontweight ='bold', fontsize = 20)
#     plt.xticks([r + 1.5*barWidth for r in range(len(Ours_New['Time']))], Data_Categories)
#     plt.grid(axis='y', linewidth=0.5)
#     fig = plt.gcf()
#     fig.set_size_inches(18.5, 10.5)
#     plt.savefig(ROOT_DIR + '/HistogramTimeResult_Ours.png')


#     plt.show()


if __name__ == "__main__":
    # showLossPlot()
    # showTimePlot()
    showLossHistogram()
    showTraningTimeHistogram()
    showInferenceTimeHistogram()
    
    # showLossPlot_Ours()
    # showTimePlot_Ours()
    # showLossHistogram_Ours()
    # showTimeHistogram_Ours()

